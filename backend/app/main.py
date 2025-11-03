from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import ml_detector, db_manager
from app.routes import detection, challenges

app = FastAPI(
    title="SQL Injection Detection API",
    description="SQL Injection detection with Machine Learning",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detection.router, prefix="/api", tags=["Detection"])
app.include_router(challenges.router, prefix="/api", tags=["Challenges"])

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 Starting SQL Injection Detection API...")
    success = await ml_detector.train_model()
    if success:
        print("✅ ML Model trained successfully!")
    else:
        print("⚠️  Using rule-based detection")
    print("✅ Application started successfully!")

@app.get("/")
async def root():
    return {"message": "SQL Injection Training API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/model-info")
async def model_info():
    return await ml_detector.get_model_info()