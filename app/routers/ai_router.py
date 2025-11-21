import json
import logging
import re
import time

from fastapi import APIRouter
from fastapi import HTTPException

from app.databases.analytics_ai_database import get_model_details, get_database_details
from app.databases.universal_mysql_database import execute_sql_mysql
from app.databases.universal_pg_database import execute_sql_pg
from app.databases.universal_vertica_database import execute_sql_vertica
from app.schemas.ai_schema import QueryRequest
from app.services.ollama_service import run_chat
from app.utils.tokens_utils import trim_messages_to_token_limit

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai",
    tags=["ai"]
)

DB_ID_EGAIS = 'egais'
DB_ID_REDMINE = 'redmine'
DB_ID_EGAIS_UTM = 'egais_utm'

DB_TYPE_POSTGRES = 'postgres'
DB_TYPE_MYSQL = 'mysql'
DB_TYPE_VERTICA = 'vertica'
DB_TYPE_MSSQL = 'mssql'

SQL_MAX_TRY = 2

NUM_CTX = 16000
NUM_CTX_GEMMA = 6000
NUM_CTX_VERTICA = 40000
# NUM_CTX = 64000 # все остальные тянут 64000
# NUM_CTX = 16000 # все остальные оптимально

OLLAMA_THINK = None  # Думает (аналог /think в запросе)
# OLLAMA_THINK = False # Не думает (аналог /no_think в запросе)
# OLLAMA_THINK = True # Думает, но не выводит


router_system_prompt_file = 'router_v4.txt'
with open(router_system_prompt_file, 'r') as file:
    router_system_prompt = file.read()

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction:
    {}
    
    Напиши SQL-код для ответа на вопрос на основе следующей схемы базы данных:
    {}
    
    ### Input:
    {}

    ### Response:
    В ответе выведи свои пояснения, если они необходимы, потом только один SQl обернутый в ```sql ```
    Пример, как нужно обернуть SQL
    ```sql
    SQL-код для ответа на вопрос
    ```
    """


@router.post("/sql")
async def post_ai_sql(request: QueryRequest):
    start_total = time.perf_counter()

    logger.info("POST Запрос получен:\n%s", request.model_dump_json())

    user_query = request.messages[-1].content

    model_details = await get_model_details(request.model)

    if model_details is None:
        raise HTTPException(status_code=500, detail=f"Модель ({request.model}) не поддерживается.")

    result = {'extend': False,
              'query': user_query,
              'sql': '',
              'message': f'Невалидный запрос model: {request.model}, q: {user_query}',
              'data': [],
              'model': model_details['model'],
              'tokens_count': 0,
              'total_duration_sec': 0
              # 'router': 'error'
              }

    if request.messages is not None:

        # ОПРЕДЕЛЕНИЕ РЕЖИМА РАБОТЫ

        mode = request.mode
        ai_router = None
        database = request.mode  # Если mode не auto, то должно совпадать с mode

        if mode == 'auto' and request.extend:
            ai_router = request.messages[-1].router
            database = request.messages[-1].router.database

        if mode == 'auto' and not request.extend:

            filtered_messages = []

            pairs = zip(request.messages, request.messages[1:])
            for user_msg, assistant_msg in pairs:
                if (
                        user_msg.role == "user"
                        and assistant_msg.role == "assistant"
                        and assistant_msg.mode == mode
                ):
                    filtered_messages.extend([
                        {"role": user_msg.role, "content": user_msg.content},
                        {"role": assistant_msg.role,
                         "content": assistant_msg.router.model_dump_json(exclude_none=True)},
                    ])

            # Добавляем последнее user сообщение
            if request.messages[-1].role == "user":
                filtered_messages.append({"role": request.messages[-1].role, "content": request.messages[-1].content})

            messages = [{"role": "system", "content": router_system_prompt}]
            # messages += [m.model_dump() for m in request.messages]
            messages += filtered_messages

            logger.info("ROUTER messages:\n%s", messages)

            max_tokens = NUM_CTX
            if request.model == 'gemma3':
                max_tokens = NUM_CTX_GEMMA

            # Подсчет токенов примерный, поэтому вычитаем 1000 токенов из max_tokens, на всякий случай
            messages, tokens_count = trim_messages_to_token_limit(messages, request.model, max_tokens=max_tokens - 1000)

            think = OLLAMA_THINK
            if request.model == 'qwen3':
                logger.info(f"Это qwen3. Отключаем reasoning")
                think = False

            logger.info(f'----- Запрос в РОУТЕР. Модель: {model_details["model"]}\n')

            response = await run_chat(model_details['model'], messages, think, max_tokens)
            logger.info(f"Ответ модели: {response}")

            response_after_think_splitted = re.split(r'</think>', response)
            response_after_think = response_after_think_splitted[-1].strip()
            ai_router = json.loads(response_after_think)
            # Пример ответа
            # {
            #   "decision": "database" | "general" | "clarification",
            #   "database": "egais" | "redmine" | null,
            #   "confidence": 0.0-1.0,
            #   "reasoning": "краткое объяснение",
            #   "response": "текст ответа (только если type=general или clarification)"
            # }

            if ai_router['decision'] == "database":
                database = ai_router['database']
            else:
                return {'extend': False,
                        'query': user_query,
                        'sql': '',
                        'message': ai_router['response'],
                        'data': [],
                        'model': model_details['model'],
                        'tokens_count': 0,
                        'total_duration_sec': 0,
                        'mode': mode,
                        'router': ai_router
                        }

        # ЗАВЕРШЕНО ОПРЕДЕЛЕНИЕ РЕЖИМА РАБОТЫ

        # ПОДГОТОВКА ДАННЫХ ДЛЯ ЗАПРОСА В МОДЕЛЬ ИЛИ В БД

        database_details = await get_database_details(database)
        instructions = database_details['instructions']
        ddl = database_details['ddl_schema']

        filtered_messages = []
        clarifications_temp = []

        pairs = zip(request.messages, request.messages[1:])
        for user_msg, assistant_msg in pairs:
            if (
                    user_msg.role == "user"
                    and assistant_msg.role == "assistant"
                    and assistant_msg.database == database
            ):
                filtered_messages.extend(clarifications_temp)
                clarifications_temp = []
                filtered_messages.extend([
                    {"role": user_msg.role, "content": user_msg.content},
                    {"role": assistant_msg.role, "content": assistant_msg.content},
                ])
            if (
                    user_msg.role == "user"
                    and assistant_msg.role == "assistant"
                    and assistant_msg.mode == "auto"
                    and assistant_msg.router.decision == "clarification"
            ):
                clarifications_temp.extend([
                    {"role": user_msg.role, "content": user_msg.content},
                    {"role": assistant_msg.role, "content": assistant_msg.content},
                ])

        # Добавляем последнее clarifications_temp и user сообщение
        if request.messages[-1].role == "user":
            filtered_messages.extend(clarifications_temp)
            filtered_messages.append({"role": request.messages[-1].role, "content": request.messages[-1].content})

        messages = [{"role": "system", "content": alpaca_prompt.format(instructions, ddl, '')}]
        # messages += [m.model_dump() for m in request.messages]
        messages += filtered_messages

        logger.info("DATABASE messages:\n%s", messages)

        max_tokens = NUM_CTX
        if request.model == 'gemma3':
            max_tokens = NUM_CTX_GEMMA
        if database == DB_ID_EGAIS_UTM:
            max_tokens = NUM_CTX_VERTICA

        # Подсчет токенов примерный, поэтому вычитаем 1000 токенов из max_tokens, на всякий случай
        messages, tokens_count = trim_messages_to_token_limit(messages, request.model, max_tokens=max_tokens - 1000)

        think = OLLAMA_THINK
        if request.model == 'qwen3':
            logger.info(f"Это qwen3. Отключаем reasoning")
            think = False

        ### GENERATE SQL

        if not request.extend:
            logger.info(f'----- Генерация SQL. Модель: {model_details["model"]}\n')

            response = await run_chat(model_details['model'], messages, think, max_tokens)
            logger.info(f'Ответ модели: {response}')

            response_after_think_splitted = re.split(r'</think>', response)
            response_after_think = response_after_think_splitted[-1].strip()

            # sql = re.search('```sql(.*)```', response, re.S).group(1).strip()
            sqls = re.findall('```sql(.+?)```', response_after_think, re.DOTALL)
            # if len(sqls) != 1:
            #     logger.info("Не удалось найти один ```sql(.+?)```")
            #     if try_idx >= SQL_MAX_TRY:
            #         raise HTTPException(status_code=500, detail="Не удалось сформировать SQL запрос")
            #     # messages.append({"role": "user", "content": request + "\nПокажи только один SQL запрос в формате Markdown SQL"})
            #     try_idx += 1
            #     continue

            messages.append({"role": "assistant", "content": response_after_think})

            if len(sqls) == 1 and 'select' in sqls[0].lower() and 'from' in sqls[0].lower():
                result = {'extend': True,
                          'query': user_query,
                          'sql': sqls[0],
                          'message': response_after_think,
                          'data': [],
                          'model': model_details['model'],
                          'tokens_count': tokens_count,
                          'total_duration_sec': round(time.perf_counter() - start_total, 2),
                          'mode': mode,
                          'database': database,
                          'router': ai_router
                          }

            else:
                result = {'extend': False,
                          'query': user_query,
                          'sql': '',
                          'message': response_after_think,
                          'data': [],
                          'model': model_details['model'],
                          'tokens_count': tokens_count,
                          'total_duration_sec': round(time.perf_counter() - start_total, 2),
                          'mode': mode,
                          'database': database,
                          'router': ai_router
                          }
        else:
            try_idx = 1
            rsp_to_chat = ''
            rsp = request.messages[-1].content
            sql = ''
            while True:
                try:
                    logger.info(f'----- Запрос в БД. Попытка номер: {try_idx} Модель: {model_details["model"]}\n')

                    response_after_think_splitted = re.split(r'</think>', rsp)
                    response_after_think = response_after_think_splitted[-1].strip()
                    logger.info(f'----- SQL:\n{response_after_think}')

                    sqls = re.findall('```sql(.+?)```', response_after_think, re.DOTALL)
                    if len(sqls) == 1 and 'select' in sqls[0].lower() and 'from' in sqls[0].lower():
                        sql = sqls[0].strip()

                        logger.info(f"Готовый SQL для запроса в БД:\n{sql}")

                        database_type = database_details['database_type']
                        if database_type == DB_TYPE_POSTGRES:
                            result_sql = await execute_sql_pg(sql, database_details)
                        elif database_type == DB_TYPE_MYSQL:
                            result_sql = await execute_sql_mysql(sql, database_details)
                        elif database_type == DB_TYPE_VERTICA:
                            result_sql = await execute_sql_vertica(sql, database_details)
                        else:
                            raise HTTPException(status_code=500,
                                                detail=f"База данных ({database_type}) не поддерживается")

                        logger.info(f"Результат запроса в БД count: {len(result_sql)}")
                        # json_string_pretty = json.dumps(result_sql, indent=2, cls=CustomJSONEncoder, ensure_ascii=False)
                        # logger.info(f"Результат запроса в БД:\n{json_string_pretty}")

                        result = {'extend': False,
                                  'query': user_query,
                                  'sql': sql,
                                  'message': rsp_to_chat,
                                  'data': result_sql,
                                  'model': model_details['model'],
                                  'tokens_count': tokens_count,
                                  'total_duration_sec': round(time.perf_counter() - start_total, 2),
                                  'mode': mode,
                                  'database': database,
                                  'router': ai_router
                                  }

                    else:
                        logger.info("Не удалось найти один ```sql(.+?)```")
                        result = {'extend': False,
                                  'query': user_query,
                                  'sql': '',
                                  'message': rsp_to_chat + '\n\n' + 'Не удалось сформировать SQL',
                                  'data': [],
                                  'model': model_details['model'],
                                  'tokens_count': tokens_count,
                                  'total_duration_sec': round(time.perf_counter() - start_total, 2),
                                  'mode': mode,
                                  'database': database,
                                  'router': ai_router
                                  }

                    break
                except Exception as e:
                    logger.error(f"Ошибка при получении данных: {str(e)}")
                    rsp_to_chat = rsp_to_chat + '\n\n' + "Ошибка: " + str(e)
                    if try_idx >= SQL_MAX_TRY:
                        result = {'extend': False,
                                  'query': user_query,
                                  'sql': sql,
                                  'message': rsp_to_chat,
                                  'data': [],
                                  'model': model_details['model'],
                                  'tokens_count': tokens_count,
                                  'total_duration_sec': round(time.perf_counter() - start_total, 2),
                                  'mode': mode,
                                  'database': database,
                                  'router': ai_router
                                  }
                        break

                    messages.append({"role": "user", "content": "Ошибка: " + str(e)})

                    try_idx += 1

                    logger.info(f'----- Генерация SQL. Попытка номер: {try_idx} Модель: {model_details["model"]}\n')

                    # Подсчет токенов примерный, поэтому вычитаем 1000 токенов из max_tokens, на всякий случай
                    messages, tokens_count = trim_messages_to_token_limit(messages, request.model,
                                                                          max_tokens=max_tokens - 1000)

                    error_fix_response = await run_chat(model_details['model'], messages, think, max_tokens)
                    error_fix_response_after_think_splitted = re.split(r'</think>', error_fix_response)
                    error_fix_response_after_think = error_fix_response_after_think_splitted[-1].strip()

                    messages.append({"role": "assistant", "content": error_fix_response_after_think})
                    rsp_to_chat = rsp_to_chat + '\n\n' + error_fix_response_after_think
                    rsp = error_fix_response_after_think

    return result
