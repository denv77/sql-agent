import logging
import os

import asyncpg
from fastapi import HTTPException

logger = logging.getLogger(__name__)

DB_CONFIG_PG_ANALYTICS_AI = os.getenv("DB_CONFIG_PG_ANALYTICS_AI")
print(DB_CONFIG_PG_ANALYTICS_AI)


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


async def get_model_details(model_name: str):
    model_sql = "SELECT * FROM models WHERE name = $1"
    model_params_sql = [model_name]
    model_details_list = await execute_sql_analytics_ai(model_sql, model_params_sql)
    if len(model_details_list) == 0 or model_details_list[0] is None:
        return None
    return model_details_list[0]


async def get_database_details(database_name: str):
    database_details_sql = "SELECT * FROM databases WHERE internal_name = $1"
    database_details_params_sql = [database_name]
    database_details_list = await execute_sql_analytics_ai(database_details_sql, database_details_params_sql)
    return database_details_list[0]
