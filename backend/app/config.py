# app/config.py
import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml_detector import SQLInjectionMLDetector
from app.database import DatabaseManager

# Initialize ML detector and database manager
ml_detector = SQLInjectionMLDetector()
db_manager = DatabaseManager()