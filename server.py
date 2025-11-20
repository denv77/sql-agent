import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncpg
import aiomysql
import asyncio
from concurrent.futures import ThreadPoolExecutor
import vertica_python
import uvicorn
import re
import json
from datetime import date, datetime
from decimal import Decimal
import uuid
import time

from ollama import chat
from ollama import generate
from ollama import ChatResponse
from ollama import AsyncClient


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# app = FastAPI()
# Создаем приложение FastAPI с кастомными параметрами
app = FastAPI(
    title="Analytics AI",
    description="Analytics AI",
    version="3.0",
    docs_url="/docs", # Путь к Swagger UI
)

ollama_client = AsyncClient()

DB_ID_EGAIS = 'egais'
DB_ID_REDMINE = 'redmine'
DB_ID_EGAIS_UTM = 'egais_utm'
DB_ID = DB_ID_EGAIS

DB_TYPE_POSTGRES = 'postgres'
DB_TYPE_MYSQL = 'mysql'
DB_TYPE_VERTICA = 'vertica'
DB_TYPE_MSSQL = 'mssql'

DB_CONFIG_PG_ANALYTICS_AI = ""

SQL_MAX_TRY = 2

NUM_CTX = 16000
NUM_CTX_GEMMA = 6000
NUM_CTX_VERTICA = 40000
# NUM_CTX = 64000 # все остальные тянут 64000
# NUM_CTX = 16000 # все остальные оптимально

OLLAMA_THINK = None # Думает (аналог /think в запросе)
# OLLAMA_THINK = False # Не думает (аналог /no_think в запросе)
# OLLAMA_THINK = True # Думает, но не выводит



router_system_prompt_file = 'router_v4.txt'
with open(router_system_prompt_file, 'r') as file:
    router_system_prompt = file.read()




class Router(BaseModel):
    decision: Optional[str] = None
    database: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    response: Optional[str] = None
class Message(BaseModel):
    role: str
    content: str
    mode: Optional[str] = None
    database: Optional[str] = None
    router: Optional[Router] = None
class QueryRequest(BaseModel):
    messages: list[Message]
    model: str
    extend: bool
    mode: str

class Database(BaseModel):
    id: Optional[int] = None
    databaseType: str
    displayName: str
    internalName: str
    host: str
    port: int
    databaseName: str
    username: str
    password: str
    instructions: Optional[str] = None
    ddlSchema: Optional[str] = None
    ddlSchemaAi: Optional[str] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None
    # numCtx: Optional[int] = None
    # temperature: Optional[float] = None
    # topP: Optional[float] = None




# Custom JSON encoder to handle dates and other special types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle date objects
        if isinstance(obj, date):
            return obj.isoformat()
        # Handle datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()
        # Handle Decimal objects (common in PostgreSQL)
        elif isinstance(obj, Decimal):
            return float(obj)
        # Handle UUID objects
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        # Let the base class handle the rest or raise TypeError
        return super().default(obj)


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




async def execute_sql_analytics_ai(query: str, params: list | tuple = None):
    try:
        conn = await asyncpg.connect(DB_CONFIG_PG_ANALYTICS_AI)
        try:
            if params:
                rows = await conn.fetch(query, *params)
            else:
                rows = await conn.fetch(query)
        finally:
            await conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Ошибка при выполнении SQL-запроса: {query}, ошибка: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


async def execute_sql_pg(query: str, database_details: dict):
    try:
        conn = await asyncpg.connect(user=database_details['username'], password=database_details['password'], host=database_details['host'], port=database_details['port'], database=database_details['database_name'])
        rows = await conn.fetch(query)
        await conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Ошибка при выполнении SQL-запроса: {query}, ошибка: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


async def execute_sql_mysql(query: str, database_details: dict):
    try:
        conn = await aiomysql.connect(user=database_details['username'], password=database_details['password'], host=database_details['host'], port=database_details['port'], db=database_details['database_name'], charset='utf8mb4')
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query)
            rows = await cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Ошибка при выполнении SQL-запроса: {query}, ошибка: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Пул потоков для выполнения синхронных операций
executor = ThreadPoolExecutor(max_workers=10)

async def execute_sql_vertica(query: str, database_details: dict) -> List[Dict[str, Any]]:
    """
    Выполняет SQL-запрос к Vertica асинхронно
    Возвращает список словарей, аналогично DictCursor в MySQL
    """
    try:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(executor, _execute_query_sync, query, database_details)
        return rows
    except Exception as e:
        logger.error(f"Ошибка при выполнении SQL-запроса: {query}, ошибка: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def _execute_query_sync(query: str, database_details: dict) -> List[Dict[str, Any]]:
    """
    Синхронное выполнение запроса (запускается в отдельном потоке)
    """
    conn = vertica_python.connect(user=database_details['username'], password=database_details['password'], host=database_details['host'], port=database_details['port'], database=database_details['database_name'], connection_timeout=300, tlsmode='disable')
    try:
        cursor = conn.cursor()
        cursor.execute(query)

        # Получаем имена колонок
        columns = [desc[0] for desc in cursor.description]

        # Преобразуем результаты в список словарей (как DictCursor)
        rows = cursor.fetchall()
        result = [dict(zip(columns, row)) for row in rows]

        return result
    finally:
        cursor.close()
        conn.close()


# Примерная, не точная оценка количества токенов
def estimate_tokens_ollama(messages: list[dict], model_name: str) -> int:
    total_tokens = 0
    for m in messages:
        total_tokens += estimate_text_tokens(m.get("content", ""), model_name)
    return total_tokens

def estimate_text_tokens(text: str, model_name: str) -> int:
    word_count = len(text.split())
    char_count = len(text)

    if model_name.startswith("qwen"):
        return int(char_count / 2)  # Qwen сильно дробит Unicode
    elif model_name.startswith("mistral") or model_name.startswith("devstral"):
        return int(word_count * 1.5)
    elif model_name.startswith("phi") or model_name.startswith("gemma"):
        return int(word_count * 2)
    else:
        return int(word_count * 1.8)  # по умолчанию


def trim_messages_to_token_limit(messages: list[dict], model_name: str, max_tokens: int, step: int = 2) -> tuple[list[dict], int]:
    """
    Обрезает сообщения, начиная с самых старых (после system), пока суммарное число токенов не станет меньше max_tokens.
    Всегда удалять на количество кратное двойке, чтобы assistant не оказался первым после system
    """
    if not messages:
        return []

    # system сообщение оставляем всегда (если оно есть)
    system_msg = []
    user_msgs = messages

    if messages[0]["role"] == "system":
        system_msg = [messages[0]]
        user_msgs = messages[1:]

    # Идем справа налево (сначала всё оставляем)
    trimmed_msgs = list(user_msgs)

    total_tokens = estimate_tokens_ollama(system_msg + trimmed_msgs, model_name)
    logger.info(f"Оценка токенов: {total_tokens}, max_tokens: {max_tokens}")
    while total_tokens > max_tokens and trimmed_msgs:
        logger.info(f"Удаляем {step} самых старых сообщений")
        trimmed_msgs = trimmed_msgs[step:]
        total_tokens = estimate_tokens_ollama(system_msg + trimmed_msgs, model_name)
        logger.info(f"Оценка токенов: {total_tokens}, max_tokens: {max_tokens}")

    if total_tokens > max_tokens:
        raise HTTPException(status_code=500, detail=f"Число токенов ({total_tokens}) превышает лимит ({max_tokens}), даже после обрезки.")

    return system_msg + trimmed_msgs, total_tokens



async def run_chat(model: str, messages: list[dict], think: bool, num_ctx: int):

    stream_sql = await ollama_client.chat(
        stream=True,
        model=model,
        messages=messages,
        think=think,
        options={
            'num_ctx': num_ctx,
            'temperature': 0.2,
            'top_p': 0.9
        }
    )

    response = ''

    async for chunk in stream_sql:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        response += content

    print("\n")
    return response



@app.post("/ai/sql", response_class=JSONResponse)
async def post_ai_sql(request: QueryRequest):

    start_total = time.perf_counter()

    logger.info("POST Запрос получен:\n%s", request.model_dump_json())

    user_query = request.messages[-1].content

    model_sql = "SELECT * FROM models WHERE name = $1"
    model_params_sql = [request.model]
    model_details_list = await execute_sql_analytics_ai(model_sql, model_params_sql)

    if len(model_details_list) == 0 or model_details_list[0] is None:
        raise HTTPException(status_code=500, detail=f"Модель ({request.model}) не поддерживается.")

    model_details = model_details_list[0]


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


    if request.messages is not None and len(model_details_list) > 0 and model_details_list[0] is not None:

        # ОПРЕДЕЛЕНИЕ РЕЖИМА РАБОТЫ

        mode = request.mode
        router = None
        database = request.mode # Если mode не auto, то должно совпадать с mode

        if mode == 'auto' and request.extend:
            router = request.messages[-1].router
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
                        {"role": assistant_msg.role, "content": assistant_msg.router.model_dump_json(exclude_none=True)},
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
            messages, tokens_count = trim_messages_to_token_limit(messages, request.model, max_tokens=max_tokens-1000)

            think = OLLAMA_THINK
            if request.model == 'qwen3':
                logger.info(f"Это qwen3. Отключаем reasoning")
                think = False

            logger.info(f'----- Запрос в РОУТЕР. Модель: {model_details["model"]}\n')

            response = await run_chat(model_details['model'], messages, think, max_tokens)
            logger.info(f"Ответ модели: {response}")

            response_after_think_splitted = re.split(r'</think>', response)
            response_after_think = response_after_think_splitted[-1].strip()
            router = json.loads(response_after_think)
            # Пример ответа
            # {
            #   "decision": "database" | "general" | "clarification",
            #   "database": "egais" | "redmine" | null,
            #   "confidence": 0.0-1.0,
            #   "reasoning": "краткое объяснение",
            #   "response": "текст ответа (только если type=general или clarification)"
            # }

            if router['decision'] == "database":
                database = router['database']
            else:
                return {'extend': False,
                        'query': user_query,
                        'sql': '',
                        'message': router['response'],
                        'data': [],
                        'model': model_details['model'],
                        'tokens_count': 0,
                        'total_duration_sec': 0,
                        'mode': mode,
                        'router': router
                        }

        # ЗАВЕРШЕНО ОПРЕДЕЛЕНИЕ РЕЖИМА РАБОТЫ



        # ПОДГОТОВКА ДАННЫХ ДЛЯ ЗАПРОСА В МОДЕЛЬ ИЛИ В БД

        database_details_sql = "SELECT * FROM databases WHERE internal_name = $1"
        database_details_params_sql = [database]
        database_details_list = await execute_sql_analytics_ai(database_details_sql, database_details_params_sql)
        database_details = database_details_list[0]
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
        messages, tokens_count = trim_messages_to_token_limit(messages, request.model, max_tokens=max_tokens-1000)

        think = OLLAMA_THINK
        if request.model == 'qwen3':
            logger.info(f"Это qwen3. Отключаем reasoning")
            think = False




        ### GENERATE SQL


        if not request.extend:
            logger.info(f'----- Генерация SQL. Модель: {model_details["model"]}\n')

            response = await run_chat(model_details['model'], messages, think, max_tokens)
            # logger.info(f"Ответ модели: {response}")

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
                          'router': router
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
                          'router': router
                          }
        else:
            try_idx = 1
            rsp_to_chat = ''
            rsp = request.messages[-1].content
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

                        result_sql = None
                        database_type = database_details['database_type']
                        if database_type == DB_TYPE_POSTGRES:
                            result_sql = await execute_sql_pg(sql, database_details)
                        elif database_type == DB_TYPE_MYSQL:
                            result_sql = await execute_sql_mysql(sql, database_details)
                        elif database_type == DB_TYPE_VERTICA:
                            result_sql = await execute_sql_vertica(sql, database_details)
                        else:
                            raise HTTPException(status_code=500, detail=f"База данных ({database_type}) не поддерживается")

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
                                  'router': router
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
                                  'router': router
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
                                  'router': router
                                  }
                        break

                    messages.append({"role": "user", "content": "Ошибка: " + str(e)})

                    try_idx += 1

                    logger.info(f'----- Генерация SQL. Попытка номер: {try_idx} Модель: {model_details["model"]}\n')

                    # Подсчет токенов примерный, поэтому вычитаем 1000 токенов из max_tokens, на всякий случай
                    messages, tokens_count = trim_messages_to_token_limit(messages, request.model, max_tokens=max_tokens-1000)

                    error_fix_response = await run_chat(model_details['model'], messages, think, max_tokens)
                    error_fix_response_after_think_splitted = re.split(r'</think>', error_fix_response)
                    error_fix_response_after_think = error_fix_response_after_think_splitted[-1].strip()

                    messages.append({"role": "assistant", "content": error_fix_response_after_think})
                    rsp_to_chat = rsp_to_chat + '\n\n' + error_fix_response_after_think
                    rsp = error_fix_response_after_think

    return result








def to_camel_case(s: str) -> str:
    """snake_case → camelCase"""
    return re.sub(r'_([a-z])', lambda m: m.group(1).upper(), s)

def snake_to_camel_dict(data):
    """
    Рекурсивно конвертирует ключи dict/списков из snake_case в camelCase.
    Сохраняет типы и структуру данных.
    """
    if isinstance(data, list):
        return [snake_to_camel_dict(item) for item in data]
    elif isinstance(data, dict):
        return {
            to_camel_case(k): snake_to_camel_dict(v)
            for k, v in data.items()
        }
    else:
        return data




@app.get("/settings/models", response_class=JSONResponse)
async def get_settings_models():
    sql = 'SELECT name, display_name, description FROM models ORDER BY display_order'
    models = await execute_sql_analytics_ai(sql)
    return {'models': snake_to_camel_dict(models)}


@app.get("/settings/work-modes", response_class=JSONResponse)
async def get_settings_work_modes():
    sql = 'SELECT internal_name, display_name, enabled FROM databases ORDER BY display_order'
    work_modes = await execute_sql_analytics_ai(sql)
    # work_modes.append({"internalName": "auto", "displayName": "Автоматически", "enabled": true})
    return snake_to_camel_dict(work_modes)


@app.get("/settings/databases", response_class=JSONResponse)
async def get_databases():
    sql = 'SELECT id, database_type, display_name, internal_name, host, port FROM databases ORDER BY display_order'
    databases = await execute_sql_analytics_ai(sql)
    return snake_to_camel_dict(databases)

@app.get("/settings/databases/{id}", response_class=JSONResponse)
async def get_database_by_id(id: int):
    sql = f"SELECT * FROM databases WHERE id = $1"
    params = [id]
    database = await execute_sql_analytics_ai(sql, params)

    if not database or len(database) == 0:
        raise HTTPException(status_code=404, detail="Database not found")

    return snake_to_camel_dict(database[0])


@app.post("/settings/databases", response_class=JSONResponse)
async def create_databases(db: Database):
    sql = """
          INSERT INTO databases (
              database_type,
              display_name,
              internal_name,
              host,
              port,
              database_name,
              username,
              password,
              instructions,
              ddl_schema,
              ddl_schema_ai,
              description
          )
          VALUES (
                     $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                 )
          RETURNING *; \
          """

    params = [
        db.databaseType,
        db.displayName,
        db.internalName,
        db.host,
        db.port,
        db.databaseName,
        db.username,
        db.password,
        db.instructions,
        db.ddlSchema,
        db.ddlSchemaAi,
        db.description,
    ]

    inserted = await execute_sql_analytics_ai(sql, params)

    return snake_to_camel_dict(inserted[0])



@app.put("/settings/databases/{id}", response_class=JSONResponse)
async def update_databases(id: int, db: Database):

    sql = """
          UPDATE databases
          SET database_type = $1,
              display_name = $2,
              internal_name = $3,
              host = $4,
              port = $5,
              database_name = $6,
              username = $7,
              password = $8,
              instructions = $9,
              ddl_schema = $10,
              ddl_schema_ai = $11,
              enabled = $12,
              description = $13
          WHERE id = $14
          RETURNING *; \
          """

    params = [
        db.databaseType,
        db.displayName,
        db.internalName,
        db.host,
        db.port,
        db.databaseName,
        db.username,
        db.password,
        db.instructions,
        db.ddlSchema,
        db.ddlSchemaAi,
        db.enabled,
        db.description,
        id
    ]

    updated = await execute_sql_analytics_ai(sql, params)

    if not updated or len(updated) == 0:
        raise HTTPException(status_code=404, detail="Database not found")

    return snake_to_camel_dict(updated[0])


@app.delete("/settings/databases/{id}")
async def delete_databases(id: int):
    sql = "DELETE FROM databases WHERE id = $1"
    params = [id]
    await execute_sql_analytics_ai(sql, params)






@app.get("/")
def read_root():
    return {"message": "Analytics AI"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
