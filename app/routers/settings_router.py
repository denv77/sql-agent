from fastapi import APIRouter
from fastapi import HTTPException

from app.databases.analytics_ai_database import execute_sql_analytics_ai
from app.schemas.settings_schema import Database
from app.utils.string_utils import snake_to_camel_dict

router = APIRouter(
    prefix="/settings",
    tags=["settings"]
)


@router.get("/models")
async def get_settings_models():
    sql = 'SELECT name, display_name, description FROM models ORDER BY display_order'
    models = await execute_sql_analytics_ai(sql)
    return {'models': snake_to_camel_dict(models)}


@router.get("/work-modes")
async def get_settings_work_modes():
    sql = 'SELECT internal_name, display_name, enabled FROM databases ORDER BY display_order'
    work_modes = await execute_sql_analytics_ai(sql)
    # work_modes.append({"internalName": "auto", "displayName": "Автоматически", "enabled": true})
    return snake_to_camel_dict(work_modes)


@router.get("/databases")
async def get_databases():
    sql = 'SELECT id, database_type, display_name, internal_name, host, port FROM databases ORDER BY display_order'
    databases = await execute_sql_analytics_ai(sql)
    return snake_to_camel_dict(databases)


@router.get("/databases/{id}")
async def get_database_by_id(id: int):
    sql = f"SELECT * FROM databases WHERE id = $1"
    params = [id]
    database = await execute_sql_analytics_ai(sql, params)

    if not database or len(database) == 0:
        raise HTTPException(status_code=404, detail="Database not found")

    return snake_to_camel_dict(database[0])


@router.post("/databases")
async def create_databases(db: Database):
    sql = """
          INSERT INTO databases (database_type,
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
                                 description)
          VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
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


@router.put("/databases/{id}")
async def update_databases(id: int, db: Database):
    sql = """
          UPDATE databases
          SET database_type = $1,
              display_name  = $2,
              internal_name = $3,
              host          = $4,
              port          = $5,
              database_name = $6,
              username      = $7,
              password      = $8,
              instructions  = $9,
              ddl_schema    = $10,
              ddl_schema_ai = $11,
              enabled       = $12,
              description   = $13
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


@router.delete("/databases/{id}")
async def delete_databases(id: int):
    sql = "DELETE FROM databases WHERE id = $1"
    params = [id]
    await execute_sql_analytics_ai(sql, params)
