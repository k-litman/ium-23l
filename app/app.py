from fastapi import FastAPI

from app.dependencies.database import connect
from app.routes.router import router


app = FastAPI()
app.include_router(router, tags=["router"], prefix="/")


@app.on_event("startup")
async def initialize_database_connection():
    await connect()
