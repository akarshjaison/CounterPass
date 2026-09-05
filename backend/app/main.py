import os
# Unset conflicting SSL certificate bundle from PostgreSQL to allow downloading YOLO weights
os.environ.pop("CURL_CA_BUNDLE", None)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.database import engine, Base
from app.api.endpoints import router as api_router

# Create Database tables automatically on start
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-Based Football Performance Review and Passing Decision Analysis System",
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to CounterPass API",
        "docs_url": "/docs",
        "health_check": f"{settings.API_V1_STR}/health"
    }
