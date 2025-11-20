import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import asyncpg
import aiomysql
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
    title="Прототип AI",
    description="Прототип AI",
    version="1.0",
    docs_url="/docs-DjbdKsjfbkjs", # Путь к Swagger UI
)

ollama_client = AsyncClient()

DB_ID_EGAIS = 'egais'
DB_ID_REDMINE = 'redmine'
DB_ID = DB_ID_EGAIS

# DB_CONFIG = ""
DB_CONFIG_PG = ""
DB_CONFIG_MYSQL = {
    'host': '',
    'port': ,
    'user': '',
    'password': '',
    'db': '',
    'charset': ''
}

SQL_MAX_TRY = 2

NUM_CTX_GEMMA = 6000 # gemma3
NUM_CTX = 16000
# NUM_CTX = 64000 # все остальные тянут 64000
# NUM_CTX = 16000 # все остальные оптимально

# OLLAMA_MODEL = "qwq"
# OLLAMA_MODEL = "qwen2.5-coder:32b"
# OLLAMA_MODEL = "gemma3:27b"
# OLLAMA_MODEL = "phi4"
OLLAMA_MODEL = "qwen3:32b"
# OLLAMA_MODEL = "phi4:14b-q8_0"
# OLLAMA_MODEL = "hf.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q6_K_L"
# OLLAMA_MODEL = "deepseek-r1:14b"
# OLLAMA_MODEL = "deepseek-r1:32b"
models = {'qwen3':'qwen3:32b',
          'qwen3think':'qwen3:32b',
          'gptoss':'gpt-oss:20b',
          'gemma3':'gemma3:27b'
          # 'qwen3large':'qwen3:235b',
          # 'devstral':'devstral:24b',
          # 'phi4':'phi4',
          # 'mistral3.1':'hf.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q6_K_L',

          }


OLLAMA_THINK = None # Думает (аналог /think в запросе)
# OLLAMA_THINK = False # Не думает (аналог /no_think в запросе)
# OLLAMA_THINK = True # Думает, но не выводит



ddl_instruction_file_redmine = 'ddl_redmine_instructions.txt'
ddl_schema_file_redmine = 'ddl_redmine_schema.txt'
ddl_instruction_file_egais = 'ddl_egais_instructions.txt'
ddl_schema_file_egais = 'ddl_egais_schema.txt'
router_system_prompt_file = 'router_v3.txt'

with open(ddl_instruction_file_redmine, 'r') as file:
    instructions_redmine = file.read()
with open(ddl_schema_file_redmine, 'r') as file:
    ddl_redmine = file.read()
with open(ddl_instruction_file_egais, 'r') as file:
    instructions_egais = file.read()
with open(ddl_schema_file_egais, 'r') as file:
    ddl_egais = file.read()
with open(router_system_prompt_file, 'r') as file:
    router_system_prompt = file.read()


class Message(BaseModel):
    role: str
    content: str
    mode: Optional[str] = None
class QueryRequest(BaseModel):
    messages: list[Message]
    model: str
    extend: bool
    mode: str


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
    В ответе выведи свой смешной ответ со смайликами, потом только один SQl обернутый в ```sql ``` и потом, если хочешь выведи свои смешные дополнения со смайликами
    Пример, как нужно обернуть SQL
    ```sql
    SQL-код для ответа на вопрос
    ```
    """



async def execute_sql_pg(query: str):
    try:
        conn = await asyncpg.connect(DB_CONFIG_PG)
        rows = await conn.fetch(query)
        await conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Ошибка при выполнении SQL-запроса: {query}, ошибка: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


async def execute_sql_mysql(query: str):
    try:
        conn = await aiomysql.connect(**DB_CONFIG_MYSQL)
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query)
            rows = await cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Ошибка при выполнении SQL-запроса: {query}, ошибка: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))




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



@app.post("/query-DjbdKsjfbkjs1", response_class=JSONResponse)
async def queryPost(request: QueryRequest):

    start_total = time.perf_counter()

    logger.info("POST Запрос получен:\n%s", request.model_dump_json())

    user_query = request.messages[-1].content

    mode = request.mode
    if mode == 'auto':
        mode = '' # тут будет функция определения режима работы
        return {'extend': False,
                'query': user_query,
                'sql': '',
                'message': f'Автоматический режим еще не реализован. Выберите ЕГАИС или Redmine.',
                'data': [],
                'model': models[request.model],
                'tokens_count': 0,
                'total_duration_sec': 0,
                'mode': 'error'
                }



    result = {'extend': False,
              'query': user_query,
              'sql': '',
              'message': f'Невалидный запрос model: {request.model}, q: {user_query}',
              'data': [],
              'model': models[request.model],
              'tokens_count': 0,
              'total_duration_sec': 0,
              'mode': 'error'
              }

    if request.messages is not None and models[request.model] is not None:

        if mode == DB_ID_EGAIS:
            instructions = instructions_egais
            ddl = ddl_egais
        else:
            instructions = instructions_redmine
            ddl = ddl_redmine

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
                    {"role": assistant_msg.role, "content": assistant_msg.content},
                ])

        # Добавляем последнее user сообщение
        if request.messages[-1].role == "user":
            filtered_messages.append({"role": request.messages[-1].role, "content": request.messages[-1].content})

        messages = [{"role": "system", "content": alpaca_prompt.format(instructions, ddl, '')}]
        # messages += [m.model_dump() for m in request.messages]
        messages += filtered_messages

        logger.info("messages:\n%s", messages)

        max_tokens = NUM_CTX
        if request.model == 'gemma3':
            max_tokens = NUM_CTX_GEMMA

        # Подсчет токенов примерный, поэтому вычитаем 1000 токенов из max_tokens, на всякий случай
        messages, tokens_count = trim_messages_to_token_limit(messages, request.model, max_tokens=max_tokens-1000)

        think = OLLAMA_THINK
        if request.model == 'qwen3':
            logger.info(f"Это qwen3. Отключаем reasoning")
            think = False


        ### GENERATE SQL


        if not request.extend:
            logger.info(f'----- Генерация SQL. Модель: {models[request.model]}\n')

            response = await run_chat(models[request.model], messages, think, max_tokens)
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
                          'model': models[request.model],
                          'tokens_count': tokens_count,
                          'total_duration_sec': round(time.perf_counter() - start_total, 2),
                          'mode': mode
                          }

            else:
                result = {'extend': False,
                          'query': user_query,
                          'sql': '',
                          'message': response_after_think,
                          'data': [],
                          'model': models[request.model],
                          'tokens_count': tokens_count,
                          'total_duration_sec': round(time.perf_counter() - start_total, 2),
                          'mode': mode
                          }
        else:
            try_idx = 1
            rsp_to_chat = ''
            rsp = request.messages[-1].content
            while True:
                try:
                    logger.info(f'----- Запрос в БД. Попытка номер: {try_idx} Модель: {models[request.model]}\n')

                    response_after_think_splitted = re.split(r'</think>', rsp)
                    response_after_think = response_after_think_splitted[-1].strip()
                    logger.info(f'----- SQL:\n{response_after_think}')

                    sqls = re.findall('```sql(.+?)```', response_after_think, re.DOTALL)
                    if len(sqls) == 1 and 'select' in sqls[0].lower() and 'from' in sqls[0].lower():
                        sql = sqls[0].strip()

                        logger.info(f"Готовый SQL для запроса в БД:\n{sql}")

                        result_sql = None
                        if mode == DB_ID_EGAIS:
                            result_sql = await execute_sql_pg(sql)
                        else:
                            result_sql = await execute_sql_mysql(sql)

                        logger.info(f"Результат запроса в БД count: {len(result_sql)}")
                        # json_string_pretty = json.dumps(result_sql, indent=2, cls=CustomJSONEncoder, ensure_ascii=False)
                        # logger.info(f"Результат запроса в БД:\n{json_string_pretty}")

                        result = {'extend': False,
                                  'query': user_query,
                                  'sql': sql,
                                  'message': rsp_to_chat,
                                  'data': result_sql,
                                  'model': models[request.model],
                                  'tokens_count': tokens_count,
                                  'total_duration_sec': round(time.perf_counter() - start_total, 2),
                                  'mode': mode
                                  }

                    else:
                        logger.info("Не удалось найти один ```sql(.+?)```")
                        result = {'extend': False,
                                  'query': user_query,
                                  'sql': '',
                                  'message': rsp_to_chat + '\n\n' + 'Не удалось сформировать SQL',
                                  'data': [],
                                  'model': models[request.model],
                                  'tokens_count': tokens_count,
                                  'total_duration_sec': round(time.perf_counter() - start_total, 2),
                                  'mode': mode
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
                                  'model': models[request.model],
                                  'tokens_count': tokens_count,
                                  'total_duration_sec': round(time.perf_counter() - start_total, 2),
                                  'mode': mode
                                  }
                        break

                    messages.append({"role": "user", "content": "Ошибка: " + str(e)})

                    try_idx += 1

                    logger.info(f'----- Генерация SQL. Попытка номер: {try_idx} Модель: {models[request.model]}\n')

                    # Подсчет токенов примерный, поэтому вычитаем 1000 токенов из max_tokens, на всякий случай
                    messages, tokens_count = trim_messages_to_token_limit(messages, request.model, max_tokens=max_tokens-1000)

                    error_fix_response = await run_chat(models[request.model], messages, think, max_tokens)
                    error_fix_response_after_think_splitted = re.split(r'</think>', error_fix_response)
                    error_fix_response_after_think = error_fix_response_after_think_splitted[-1].strip()

                    messages.append({"role": "assistant", "content": error_fix_response_after_think})
                    rsp_to_chat = rsp_to_chat + '\n\n' + error_fix_response_after_think
                    rsp = error_fix_response_after_think

    return result








@app.get("/query-DjbdKsjfbkjs1", response_class=JSONResponse)
async def queryGet(q: str | None = None, model: str | None = None):

    start_total = time.perf_counter()

    logger.info(f"Запрос /query model: {model}, real: {models[model]}, q: {q}")

    result = {'query': q,
              'sql': '',
              'message': f'Невалидный запрос model: {model}, q: {q}',
              'data': [],
              'model': models[model],
              'total_duration_sec': round(time.perf_counter() - start_total, 2)
              }

    if q is not None and models[model] is not None:

        messages = [
            {"role": "user", "content": alpaca_prompt.format(instructions, ddl, q)},
        ]


        ### GENERATE SQL

        try_idx = 1
        while True:
            try:
                logger.info(f'----- Генерация SQL. Попытка номер: {try_idx} Модель: {models[model]}\n')

                think = OLLAMA_THINK
                if model == 'qwen3':
                    logger.info(f"Отключаем reasoning")
                    think = False

                stream_sql = chat(stream=True,
                                  model=models[model],
                                  messages=messages,
                                  think=think,
                                  options={
                                      'num_ctx': NUM_CTX,
                                      'temperature': 0.1,
                                      'top_p': 0.95
                                  }
                                  )

                response = ''

                for chunk in stream_sql:
                    print(chunk['message']['content'], end='', flush=True)
                    response += chunk['message']['content']


                # logger.info(f"Ответ нейронки SQL: {response}")

                response_after_think_splitted = re.split(r'</think>', response)
                response_after_think = response_after_think_splitted[-1].strip()

                # sql = re.search('```sql(.*)```', response, re.S).group(1).strip()
                sqls = re.findall('```sql(.+?)```', response_after_think, re.DOTALL)
                if len(sqls) != 1:
                    logger.info("Не удалось найти один ```sql(.+?)```")
                    if try_idx >= SQL_MAX_TRY:
                        raise HTTPException(status_code=500, detail="Не удалось сформировать SQL запрос")
                    # messages.append({"role": "user", "content": request + "\nПокажи только один SQL запрос в формате Markdown SQL"})
                    try_idx += 1
                    continue

                messages.append({"role": "assistant", "content": response})

                sql = sqls[0].strip()

                logger.info(f"Готовый SQL для запроса в БД:\n{sql}")

                result_sql = None
                if DB_ID == DB_ID_EGAIS:
                    result_sql = await execute_sql_pg(sql)
                else:
                    result_sql = await execute_sql_mysql(sql)

                json_string_pretty = json.dumps(result_sql, indent=2, cls=CustomJSONEncoder, ensure_ascii=False)

                logger.info(f"Результат запроса в БД:\n{json_string_pretty}")

                result = {'query': q,
                          'sql': sql,
                          'message': response,
                          'data': result_sql,
                          'model': models[model],
                          'total_duration_sec': round(time.perf_counter() - start_total, 2)
                          }
                break
            except Exception as e:
                logger.error(f"Ошибка при получении данных: {str(e)}")
                if try_idx >= SQL_MAX_TRY:
                    raise HTTPException(status_code=500, detail=str(e))
                messages.append({"role": "user", "content": "Можешь исправить ошибку: " + str(e)})
                try_idx += 1

    return result





@app.get("/test")
async def queryTest(q: str):

    logger.info(f"Запрос /test q: {q}")

    messages = [
        {"role": "user", "content": alpaca_prompt.format(instructions, ddl, q)},
    ]

    stream_sql = chat(stream=True,
                      model='qwen3:32b',
                      messages=messages,
                      options={
                          'num_ctx': NUM_CTX,
                          'temperature': 0.1,
                          'top_p': 0.95
                      }
                      )

    response = ''

    for chunk in stream_sql:
        print(chunk['message']['content'], end='', flush=True)
        response += chunk['message']['content']


    logger.info(f"Ответ нейронки SQL: {response}")

    response_after_think_splitted = re.split(r'</think>', response)
    response_after_think = response_after_think_splitted[-1].strip()

    sqls = re.findall('```sql(.+?)```', response_after_think, re.DOTALL)

    if len(sqls) == 1 and 'select' in sqls[0].lower() and 'from' in sqls[0].lower():
        sql = sqls[0].strip()
        logger.info(f"Готовый SQL для запроса в БД:\n{sql}")
        result_sql = None
        if DB_ID == DB_ID_EGAIS:
            result_sql = await execute_sql_pg(sql)
        else:
            result_sql = await execute_sql_mysql(sql)
        logger.info(f"result_sql:\n{result_sql}")

    return response



@app.get("/")
def read_root():
    return {"message": "Hello, Den!"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
