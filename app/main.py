from fastapi import FastAPI
import uvicorn

from app.dependencies.database import connect
from app.routes.router import router


app = FastAPI()
app.include_router(router, tags=["router"], prefix="/api")


@app.on_event("startup")
async def initialize_database_connection():
    await connect()


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
