from pydantic import BaseModel


class Database(BaseModel):
    id: int | None = None
    databaseType: str
    displayName: str
    internalName: str
    host: str
    port: int
    databaseName: str
    username: str
    password: str
    instructions: str | None = None
    ddlSchema: str | None = None
    ddlSchemaAi: str | None = None
    enabled: str | None = None
    description: str | None = None
    # numCtx: int | None = None
    # temperature: float | None = None
    # topP: float | None = None
