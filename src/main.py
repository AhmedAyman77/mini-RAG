from fastapi import FastAPI
from routes import base, data
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings, Settings
from stores.llm.LLMProviderFactory import LLMProviderFactory

app = FastAPI()


# ================================================================
# The @app.on_event("startup") is a decorator that specifies a function
# to run automatically when the application starts up (i.e., when you run the server).
#
# Why use it?
# 1. To load data or initialize settings before the app starts accepting requests.
# 2. To set up things like database connections or load machine learning models.
# ================================================================
@app.on_event("startup")
async def startup_db_client():
    settings = get_settings()

    app.mongo_connection = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_connection[settings.MONGODB_DATABASE]

    llm_provider_factory = LLMProviderFactory(settings)

    # generation client
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    # embedding client
    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )


@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongo_connection.close()



app.include_router(base.base_router)
app.include_router(data.data_router)
