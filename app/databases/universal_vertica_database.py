import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import vertica_python
from fastapi import HTTPException

logger = logging.getLogger(__name__)

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
    conn = vertica_python.connect(user=database_details['username'], password=database_details['password'],
                                  host=database_details['host'], port=database_details['port'],
                                  database=database_details['database_name'], connection_timeout=300, tlsmode='disable')
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
