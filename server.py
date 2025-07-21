import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncpg
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
    version="1.0.0",
    docs_url="/docs-DjbdKsjfbkjs", # Путь к Swagger UI
)

ollama_client = AsyncClient()

DB_CONFIG = ""

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
          'devstral':'devstral:24b',
          'phi4':'phi4', 
          'mistral3.1':'hf.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q6_K_L', 
          'gemma3':'gemma3:27b'
}


OLLAMA_THINK = None # Думает (аналог /think в запросе)
# OLLAMA_THINK = False # Не думает (аналог /no_think в запросе)
# OLLAMA_THINK = True # Думает, но не выводит


with open('ddl_egais_instructions.txt', 'r') as file:
    instructions = file.read()
with open('ddl_egais_schema.txt', 'r') as file:
    ddl = file.read()


class Message(BaseModel):
    role: str
    content: str
class QueryRequest(BaseModel):
    messages: list[Message]
    model: str


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
    В ответ выведи только SQl обернутый в ```sql ```
    Пример
    ```sql
    SQL-код для ответа на вопрос
    ```
    """



async def execute_sql_pg(query: str):
    try:
        conn = await asyncpg.connect(DB_CONFIG)
        rows = await conn.fetch(query)
        await conn.close()
        return [dict(row) for row in rows]
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
            'temperature': 0.1,
            'top_p': 0.95
        }
    )

    response = ''

    async for chunk in stream_sql:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        response += content

    return response

    

@app.post("/query-DjbdKsjfbkjs1", response_class=JSONResponse)
async def queryPost(request: QueryRequest):

    start_total = time.perf_counter()
    
    logger.info("POST Запрос получен:\n%s", request.json())

    user_query = request.messages[-1].content
    
    result = {'query': user_query,
              'sql': '',
              'message': f'Невалидный запрос model: {request.model}, q: {user_query}',
              'data': [],
              'model': models[request.model],
              'tokens_count': 0,
              'total_duration_sec': 0
             }
    
    if request.messages is not None and models[request.model] is not None:

        messages = [{"role": "system", "content": alpaca_prompt.format(instructions, ddl, '')}]
        messages += request.messages

        messages = [{"role": "system", "content": alpaca_prompt.format(instructions, ddl, '')}]
        messages += [m.model_dump() for m in request.messages]

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

        try_idx = 1
        while True:
            try:
                logger.info(f'----- Генерация SQL. Попытка номер: {try_idx} Модель: {models[request.model]}\n')

                response = await run_chat(models[request.model], messages, think, max_tokens)
                # logger.info(f"Ответ модели: {response}")

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
                
                result_sql = await execute_sql_pg(sql)
                
                json_string_pretty = json.dumps(result_sql, indent=2, cls=CustomJSONEncoder, ensure_ascii=False)

                logger.info(f"Результат запроса в БД:\n{json_string_pretty}")

                result = {'query': user_query,
                          'sql': sql,
                          'message': response, 
                          'data': result_sql, 
                          'model': models[request.model],
                          'tokens_count': tokens_count,
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
                
                result_sql = await execute_sql_pg(sql)
                
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









@app.get("/")
def read_root():
    return {"message": "Hello, Den!"}



if __name__ == "__main__":
    uvicorn.run(app, host="46.148.205.148", port=9000)