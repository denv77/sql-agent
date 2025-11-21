from pydantic import BaseModel


class Router(BaseModel):
    decision: str | None = None
    database: str | None = None
    confidence: float | None = None
    reasoning: str | None = None
    response: str | None = None


class Message(BaseModel):
    role: str
    content: str
    mode: str | None = None
    database: str | None = None
    router: Router | None = None


class QueryRequest(BaseModel):
    messages: list[Message]
    model: str
    extend: bool
    mode: str
