from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import ml_detector, db_manager
from app.routes import detection, challenges
from app.routes import sql_routes  # Add this import

app = FastAPI(
    title="SQL Injection Detection API",
    description="SQL Injection detection with Machine Learning",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://localhost:5173",    # ADD THIS
        "http://127.0.0.1:5173",    # ADD THIS
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detection.router, prefix="/api", tags=["Detection"])
app.include_router(challenges.router, prefix="/api", tags=["Challenges"])
app.include_router(sql_routes.router, prefix="/api", tags=["SQL Detection"])

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

@app.get("/api/model-info")
async def model_info():
    return await ml_detector.get_model_info()

@app.get("/api/detection-stats")
async def get_detection_stats():
    """Get comprehensive detection statistics"""
    return await db_manager.get_detection_stats()

@app.get("/api/user/{user_id}/progress")
async def get_user_progress(user_id: int):
    """Get user progress and achievements"""
    return await db_manager.get_user_progress(user_id)