import logging

import aiomysql
from fastapi import HTTPException

logger = logging.getLogger(__name__)


async def execute_sql_mysql(query: str, database_details: dict):
    try:
        conn = await aiomysql.connect(user=database_details['username'], password=database_details['password'],
                                      host=database_details['host'], port=database_details['port'],
                                      db=database_details['database_name'], charset='utf8mb4')
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query)
            rows = await cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Ошибка при выполнении SQL-запроса: {query}, ошибка: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
