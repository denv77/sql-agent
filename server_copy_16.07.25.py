import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncpg
import aiomysql
import uvicorn
import re
import json
from datetime import date, datetime
from decimal import Decimal
import uuid

from ollama import chat
from ollama import generate
from ollama import ChatResponse

from anthropic import Anthropic
from openai import OpenAI



from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer



# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# app = FastAPI()

# Создаем приложение FastAPI с кастомными параметрами
app = FastAPI(
    title="Прототип AI",                # Название API
    description="Прототип AI",  # Описание API
    version="1.0.0",                # Версия API
    docs_url="/docs-DjbdKsjfbkjs",            # Путь к Swagger UI
    # openapi_url="/custom_openapi.json"  # Путь к OpenAPI JSON
)

# DB_CONFIG = ""
DB_CONFIG = ""
DB_CONFIG_MYSQL = {
}


SQL_MAX_TRY = 2

NUM_CTX = 6000 # gemma3
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

#OLLAMA_THINK = None # Думает (аналог /think в запросе)
OLLAMA_THINK = False # Не думает (аналог /no_think в запросе)
#OLLAMA_THINK = True # Думает, но не выводит

USE_API = False

API_URL = 'https://api.proxyapi.ru/anthropic'
# API_URL = 'https://api.proxyapi.ru/deepseek'
API_KEY = ''

client = Anthropic(
    base_url=API_URL,
    api_key=API_KEY
)
# client = OpenAI(
#     api_key=API_KEY,
#     base_url=API_URL
# )



# OLLAMA_MODEL = "denv77/phi4_lora_model"
# model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = "denv77/phi4_lora_model", # YOUR MODEL YOU USED FOR TRAINING
#         max_seq_length = 16000,
#         load_in_4bit = True,
# )
# FastLanguageModel.for_inference(model) 



# with open('ddl_only.txt', 'r') as file:
# with open('ddl.txt', 'r') as file:
with open('ddl_egais.txt', 'r') as file:
    ddl = file.read()

with open('chart.txt', 'r') as file:
    chart = file.read()

with open('html.txt', 'r') as file:
    html = file.read()

# if USE_API:
#     messages = []
#     messages_chart = []
    # messages = [
    #     {"role": "user", "content": ddl},
    # ]
    # messages_chart = [
    #     {"role": "user", "content": chart},
    # ]
    # print('--- start init deepseek api')
    # response_sql_start_api = client.chat.completions.create(model='deepseek-reasoner', messages=messages)
    # response_sql_start = response_sql_start_api.choices[0].message.content
    # messages.append({"role": "assistant", "content": response_sql_start})
    # print('--- response_sql_start:', response_sql_start)
    # response_chart_start_api = client.chat.completions.create(model='deepseek-reasoner', messages=messages_chart)
    # response_chart_start = response_chart_start_api.choices[0].message.content
    # messages_chart.append({"role": "assistant", "content": response_chart_start})
    # print('--- response_chart_start:', response_chart_start)
# else:
    # messages = [
    #     {"role": "system", "content": ddl},
    # ]
    # messages_chart = [
    #     {"role": "system", "content": chart},
    # ]

# Модель запроса
class QueryRequest(BaseModel):
    query: str

# Функция выполнения SQL-запроса
async def execute_sql_pg(query: str):
    try:
        conn = await asyncpg.connect(DB_CONFIG)
        rows = await conn.fetch(query)
        await conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Ошибка при выполнении SQL-запроса: {query}, ошибка: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


async def execute_sql_mysql(query: str):
    try:
        # Создаем соединение с базой данных
        conn = await aiomysql.connect(
            host=DB_CONFIG_MYSQL['host'],
            port=DB_CONFIG_MYSQL['port'],
            user=DB_CONFIG_MYSQL['user'],
            password=DB_CONFIG_MYSQL['password'],
            db=DB_CONFIG_MYSQL['database']
        )
        
        # Создаем курсор для выполнения запроса
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            # Выполняем запрос
            await cursor.execute(query)
            # Получаем результаты
            rows = await cursor.fetchall()
        
        # Закрываем соединение
        conn.close()
        
        # Возвращаем результаты в виде списка словарей
        return list(rows)
    except Exception as e:
        logger.error(f"Ошибка при выполнении SQL-запроса: {query}, ошибка: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


import json

def generate_html_table_with_pagination(data, rows_per_page=10):
    # Проверка на пустой список
    if not data:
        return "<p>No data available</p>"

    # Получаем заголовки из первого элемента
    headers = data[0].keys()

    table_headers_html = ""
    for header in headers:
        table_headers_html += f'<th onclick="sortTableBy(\'{header}\')">{header}</th>'

    # Генерируем HTML
    html = f"""
        <table id="dataTable">
            <thead>
                <tr>{table_headers_html}</tr>
            </thead>
            <tbody>
            </tbody>
        </table>
    
        <div class="pagination">
            <button id="prevBtn">Назад</button>
            <span id="pageInfo"></span>
            <button id="nextBtn">Вперёд</button>
        </div>
    
        <script>
            const originalData = {json.dumps(data, cls=CustomJSONEncoder, ensure_ascii=False)};
            let data = [...originalData];
            const rowsPerPage = {rows_per_page};
            let currentPage = 1;
            let currentSortKey = null;
            let sortDirection = 1;
    
            function renderTablePage() {{
                const tableBody = document.querySelector("#dataTable tbody");
                tableBody.innerHTML = "";
    
                const start = (currentPage - 1) * rowsPerPage;
                const end = start + rowsPerPage;
                const pageData = data.slice(start, end);
    
                for (let row of pageData) {{
                    const tr = document.createElement("tr");
                    for (let header of Object.keys(originalData[0])) {{
                        const td = document.createElement("td");
                        td.textContent = row[header] !== undefined ? row[header] : "";
                        tr.appendChild(td);
                    }}
                    tableBody.appendChild(tr);
                }}
    
                const pageCount = Math.ceil(data.length / rowsPerPage);
                document.getElementById("pageInfo").textContent = `Страница ${{currentPage}} из ${{pageCount}}`;
                document.getElementById("prevBtn").disabled = currentPage === 1;
                document.getElementById("nextBtn").disabled = currentPage === pageCount;
            }}
    
            function sortTableBy(key) {{
                if (currentSortKey === key) {{
                    sortDirection *= -1;
                }} else {{
                    currentSortKey = key;
                    sortDirection = 1;
                }}
    
                data.sort((a, b) => {{
                    let valA = a[key] ?? '';
                    let valB = b[key] ?? '';
    
                    const numA = parseFloat(valA);
                    const numB = parseFloat(valB);
    
                    if (!isNaN(numA) && !isNaN(numB)) {{
                        return (numA - numB) * sortDirection;
                    }}
    
                    return valA.toString().localeCompare(valB.toString(), 'ru', {{ sensitivity: 'base' }}) * sortDirection;
                }});
    
                currentPage = 1;
                renderTablePage();
            }}
    
            document.getElementById("prevBtn").addEventListener("click", () => {{
                if (currentPage > 1) {{
                    currentPage--;
                    renderTablePage();
                }}
            }});
    
            document.getElementById("nextBtn").addEventListener("click", () => {{
                const pageCount = Math.ceil(data.length / rowsPerPage);
                if (currentPage < pageCount) {{
                    currentPage++;
                    renderTablePage();
                }}
            }});
    
            renderTablePage();
        </script>
    """
    return html



def unsloth_generate(messages, model ,tokenizer):
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")
    
    text_streamer = TextStreamer (tokenizer, skip_prompt = True)
    # resp = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 256, temperature = 0.1, use_cache = True, pad_token_id=tokenizer.eos_token_id)
    outp = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 2048, temperature = 0.1, top_p = 0.95, pad_token_id=tokenizer.pad_token_id)
    
    # token_ids = outp[0].tolist()
    # resp = tokenizer.decode(token_ids, skip_special_tokens=True)
    # resp = tokenizer.batch_decode(outp, skip_prompt = True, skip_special_tokens=True)

    # Достаем только ответ
    resp = tokenizer.batch_decode(outp[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return resp


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


@app.get("/query-DjbdKsjfbkjs", response_class=HTMLResponse)
async def query(request: str | None = None):
    logger.info(f"Запрос /query: {request}")
    
    result = html



    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction:
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
    
    
    messages = [
        {"role": "user", "content": alpaca_prompt.format(ddl, request, "")},
    ]

    # messages = []
    # messages = [
    #     {"role": "system", "content": ddl},
    # ]
    messages_chart = [
        {"role": "system", "content": chart},
    ]
    
    if request is not None:


        

        ### GENERATE SQL



        # messages.append({"role": "user", "content": request})
        try_idx = 1
        while True:
            try:
                print(f'\n----- SQL Попытка номер: {try_idx}\n')

                response = ''
                if USE_API:
                    logger.info("Запрос на API SQL")
                    response_api = client.messages.create(max_tokens=2048, system=ddl, messages=messages, model="claude-3-7-sonnet-20250219")
                    response = response_api.content[0].text
                    # response_api = client.chat.completions.create(model='deepseek-reasoner', messages=messages)
                    # response = response_api.choices[0].message.content
                else:
                    logger.info(f"Запрос в OLLAMA SQL {OLLAMA_MODEL}")
                    # response: ChatResponse = chat(model=OLLAMA_MODEL, messages=messages, options={'temperature': OLLAMA_MODEL_TEMPERATURE})
                    
                    # response_ollama: ChatResponse = chat(model=OLLAMA_MODEL, messages=messages, options={
                    #                     'num_ctx': 8096, 
                    #                     # 'temperature': 0.1, 
                    #                     # 'top_p': 0.95
                    #                 })

                    # response = response_ollama.message.content

                    stream_sql = chat(stream=True, model=OLLAMA_MODEL, messages=messages, think=OLLAMA_THINK, options={
                                        'num_ctx': NUM_CTX, 
                                        'temperature': 0.1, 
                                        'top_p': 0.95
                                    })
                    
                    for chunk in stream_sql:
                        print(chunk['message']['content'], end='', flush=True)
                        response += chunk['message']['content']
                            

                    # response = unsloth_generate(messages, model, tokenizer)
                

                    
                    # response_ollama = generate(
                    #     model=OLLAMA_MODEL,
                    #     prompt=request,
                    #     system=ddl
                    # )
                    # response = response['response']
          

                    
                
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
                print('\n')
                logger.info(f"SQL:\n{sql}")
                
                result_sql = await execute_sql_pg(sql)
                # result_sql = await execute_sql_mysql(sql)
                
                json_string_pretty = json.dumps(result_sql, indent=2, cls=CustomJSONEncoder, ensure_ascii=False)
                json_message = f'{request}\n{json_string_pretty}'
                print('\n')
                logger.info(f"SQL json_message: {json_message}")
                break
            except Exception as e:
                logger.error(f"Ошибка при получении данных: {str(e)}")
                if try_idx >= SQL_MAX_TRY:
                    raise HTTPException(status_code=500, detail=str(e))
                messages.append({"role": "user", "content": "Исправь ошибку в SQL: " + str(e)})
                # messages = [
                #         {"role": "user", "content": alpaca_prompt.format(ddl, request, "", "")},
                # ]
                try_idx += 1
        

        messages_chart.append({"role": "user", "content": json_message})

        result_sql_len = len(result_sql)
        logger.info(f"len(result_sql): {result_sql_len}")



        
        ### GENERATE HTML


        
        
        # if result_sql_len > 0 and result_sql_len <= 30:
        if result_sql_len < 0:
        
            try_idx = 1
            while True:
                print(f'\n----- CHART Попытка номер: {try_idx}\n')
                
                response_chart = ''
                if USE_API:
                    logger.info("Запрос на API CHART")
                    response_chart_api = client.messages.create(max_tokens=4096, system=chart, messages=messages_chart, model="claude-3-7-sonnet-20250219")
                    response_chart = response_chart_api.content[0].text
                    # response_chart_api = client.chat.completions.create(model='deepseek-reasoner', messages=messages_chart)
                    # response_chart = response_chart_api.choices[0].message.content
                else:
                    logger.info(f"Запрос в OLLAMA CHART {OLLAMA_MODEL}")
                    # response_chart_ollama: ChatResponse = chat(model=OLLAMA_MODEL, messages=messages_chart, options={
                    #                         'num_ctx': 8096, 
                    #                         # 'temperature': 0.1, 
                    #                         # 'top_p': 0.95
                    #                     })
                    # response_chart = response_chart_ollama.message.content
    
                    stream_chart = chat(stream=True, model=OLLAMA_MODEL, messages=messages_chart, options={
                                                'num_ctx': NUM_CTX, 
                                                'temperature': 0.1, 
                                                'top_p': 0.95
                                            })
                    
                    for chunk_chart in stream_chart:
                        print(chunk_chart['message']['content'], end='', flush=True)
                        response_chart += chunk_chart['message']['content']
    
                    
        
                # logger.info(f"Ответ нейронки HTML: {response_chart}")
                
    
                response_chart_after_think_splitted = re.split(r'</think>', response_chart)
                response_chart_after_think = response_chart_after_think_splitted[-1].strip()
                
                # chart_html = re.search('```html(.*)```', response_chart, re.S).group(1).strip()
                chart_htmls =  re.findall('```html(.+?)```', response_chart_after_think, re.DOTALL)
                if len(chart_htmls) != 1:
                    logger.info("Не удалось найти один ```html(.+?)```")
                    if try_idx >= SQL_MAX_TRY:
                        raise HTTPException(status_code=500, detail="Не удалось сформировать HTML")
                    # messages_chart.append({"role": "user", "content": json_message + "\nПокажи только один фрагмент HTML в формате Markdown HTML"})
                    try_idx += 1
                    continue
    
                messages_chart.append({"role": "assistant", "content": response_chart})
                
                chart_html = chart_htmls[0].strip()
        
                result += f'\n<h2 style="text-align: center;margin-bottom: 30px;">{request}</h2>\n<div style="width: 800px;">\n{chart_html}\n</div><div style="margin-top: 30px; width: 800px;"><pre style="white-space: pre-wrap;">{sql}</pre><pre style="margin-top: 20px; white-space: pre-wrap;">{json_string_pretty}</pre></div>'
                break

        elif result_sql_len > 0:
            html_table = generate_html_table_with_pagination(result_sql)
            result += f'\n<h2 style="text-align: center;margin-bottom: 30px;">{request}</h2>\n<div style="width: 800px;">\n{html_table}\n</div><div style="margin-top: 30px; width: 800px;"><pre style="white-space: pre-wrap;">{sql}</pre><pre style="margin-top: 20px;">result_sql_len: {result_sql_len}</pre></div>'
        else:
            result += f'\n<h2 style="text-align: center;margin-bottom: 30px;">{request}</h2>\n<div style="width: 800px;">\nНет данных\n</div><div style="margin-top: 30px; width: 800px;"><pre style="white-space: pre-wrap;">{sql}</pre><pre style="margin-top: 20px;">{json_string_pretty}</pre></div>'
        
    result += '\n</body>\n</html>'
    # logger.info(f"html: {result}")
    return result
    # return response

@app.get("/")
def read_root():
    return {"message": "Hello, Den!"}



if __name__ == "__main__":
    uvicorn.run(app, host="46.148.205.148", port=9000)