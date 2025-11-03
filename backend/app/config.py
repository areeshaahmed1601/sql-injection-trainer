from app.ml_detector import SQLInjectionMLDetector
from app.database import DatabaseManager

# Global instances
ml_detector = SQLInjectionMLDetector()
db_manager = DatabaseManager()