from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import logging

import uvicorn

from fastapi import FastAPI

from app.routers import ai_router, settings_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "4.0"
logger.info(f"Starting SQL Agent version: {VERSION}")

app = FastAPI(
    title="SQL Agent",
    description="Database management",
    version=VERSION,
    docs_url="/docs",  # Путь к Swagger UI
)
app.include_router(ai_router.router)
app.include_router(settings_router.router)


@app.get("/")
def read_root():
    return {"project": "SQL Agent", "version": VERSION}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
