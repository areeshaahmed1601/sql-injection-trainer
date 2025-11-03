import uvicorn
import os
import pandas as pd
import sys
from typing import Optional, Dict

def initialize_environment():
    """Initialize the application environment with dataset combination"""
    # Create necessary directories
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("database", exist_ok=True)
    os.makedirs("app/routes", exist_ok=True)
    
    print("🚀 Starting SQL Injection Detection API with ML...")
    print("📊 Initializing and combining datasets...")
    
    # Combine datasets if Thesis 1 dataset exists
    combine_datasets_if_available()
    
    # Ensure we have a dataset
    dataset_info = ensure_dataset_exists()
    
    print("✅ Environment initialized successfully!")
    print("🔗 API: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print(f"📁 Using dataset: {dataset_info['path']} ({dataset_info['size']} queries)")

def combine_datasets_if_available():
    """Combine datasets if Thesis 1 dataset is available"""
    try:
        # Check specifically for Modified_SQL_Dataset.csv in datasets folder
        thesis1_path = "datasets/Modified_SQL_Dataset.csv"
        
        if os.path.exists(thesis1_path):
            print(f"🔗 Found Thesis 1 dataset: {thesis1_path}")
            
            # Check file size to make sure it's valid
            file_size = os.path.getsize(thesis1_path)
            if file_size < 1024:  # Less than 1KB
                print(f"⚠️  Thesis 1 dataset seems too small ({file_size} bytes), skipping combination")
                return
            
            print("🔄 Combining with current dataset...")
            
            # Import and run the combiner
            try:
                # Add current directory to Python path to ensure imports work
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                
                from datasets.combine_datasets import DatasetCombiner
                combiner = DatasetCombiner()
                combined_df = combiner.run_combination()
                
                if combined_df is not None and len(combined_df) > 0:
                    print(f"✅ Successfully combined datasets: {len(combined_df)} total queries")
                    
                    # Show combined dataset statistics
                    benign_count = len(combined_df[combined_df['label'] == 'benign'])
                    malicious_count = len(combined_df[combined_df['label'] == 'malicious'])
                    other_count = len(combined_df) - benign_count - malicious_count
                    
                    print("📊 Combined Dataset Statistics:")
                    print(f"   Benign queries: {benign_count} ({benign_count/len(combined_df)*100:.1f}%)")
                    print(f"   Malicious queries: {malicious_count} ({malicious_count/len(combined_df)*100:.1f}%)")
                    if other_count > 0:
                        print(f"   Other labels: {other_count} ({other_count/len(combined_df)*100:.1f}%)")
                        
                else:
                    print("⚠️  Dataset combination completed but no data was combined")
                    
            except ImportError as e:
                print(f"⚠️  Could not import dataset combiner: {e}")
                print("📊 Using existing dataset without combination")
                create_fallback_dataset_combiner()
            except Exception as e:
                print(f"⚠️  Dataset combination failed: {e}")
                print("📊 Using existing dataset without combination")
                create_fallback_dataset_combiner()
        else:
            print("📊 Using existing dataset (Thesis 1 dataset not found in datasets/ folder)")
            
    except Exception as e:
        print(f"⚠️  Dataset discovery failed: {e}")
        print("📊 Using existing dataset")

def create_fallback_dataset_combiner():
    """Create a simple dataset combiner as fallback"""
    try:
        print("🔄 Creating fallback dataset combination...")
        
        # Load current dataset if exists
        current_df = pd.DataFrame()
        if os.path.exists("datasets/complete_sql_dataset.csv"):
            current_df = pd.read_csv("datasets/complete_sql_dataset.csv")
            print(f"   ✅ Loaded current dataset: {len(current_df)} queries")
        
        # Load Thesis 1 dataset
        thesis1_df = pd.read_csv("datasets/Modified_SQL_Dataset.csv")
        print(f"   ✅ Loaded Thesis 1 dataset: {len(thesis1_df)} queries")
        
        # Simple column standardization
        if 'Query' in thesis1_df.columns and 'query' not in thesis1_df.columns:
            thesis1_df = thesis1_df.rename(columns={'Query': 'query'})
        if 'Label' in thesis1_df.columns and 'label' not in thesis1_df.columns:
            thesis1_df = thesis1_df.rename(columns={'Label': 'label'})
        
        # Combine datasets
        if not current_df.empty:
            combined_df = pd.concat([current_df, thesis1_df], ignore_index=True)
            # Remove duplicates
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['query'], keep='first')
            after_dedup = len(combined_df)
            print(f"   🔄 Removed {before_dedup - after_dedup} duplicate queries")
        else:
            combined_df = thesis1_df
        
        # Save combined dataset
        combined_df.to_csv("datasets/combined_sql_dataset.csv", index=False)
        print(f"   💾 Saved combined dataset: {len(combined_df)} queries")
        
    except Exception as e:
        print(f"❌ Fallback combination failed: {e}")

def ensure_dataset_exists() -> Dict[str, any]:
    """Ensure we have a dataset file, create one if needed"""
    # Priority order for datasets
    dataset_priority = [
        "datasets/combined_sql_dataset.csv",  # Combined dataset first
        "datasets/complete_sql_dataset.csv",  # Then current dataset
        "datasets/Modified_SQL_Dataset.csv"   # Then Thesis 1 dataset alone
    ]
    
    # Find existing dataset
    for dataset_path in dataset_priority:
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path)
                if len(df) >= 10:  # Reasonable minimum
                    return {
                        "path": dataset_path,
                        "size": len(df),
                        "type": "existing"
                    }
            except Exception as e:
                print(f"⚠️  Error reading {dataset_path}: {e}")
                continue
    
    # No suitable dataset found, create one
    print("📝 No suitable dataset found, creating comprehensive dataset...")
    return create_comprehensive_dataset()

def create_comprehensive_dataset() -> Dict[str, any]:
    """Create a comprehensive dataset for training"""
    try:
        # Try to use the Thesis 1 dataset directly
        if os.path.exists("datasets/Modified_SQL_Dataset.csv"):
            df = pd.read_csv("datasets/Modified_SQL_Dataset.csv")
            # Simple standardization
            if 'Query' in df.columns and 'query' not in df.columns:
                df = df.rename(columns={'Query': 'query'})
            if 'Label' in df.columns and 'label' not in df.columns:
                df = df.rename(columns={'Label': 'label'})
            
            output_path = "datasets/combined_sql_dataset.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Using Thesis 1 dataset directly: {len(df)} queries")
            return {
                "path": output_path,
                "size": len(df),
                "type": "thesis1_direct"
            }
        
        # Fallback: create basic dataset
        return create_basic_dataset()
        
    except Exception as e:
        print(f"❌ Error creating comprehensive dataset: {e}")
        return create_basic_dataset()

def create_basic_dataset() -> Dict[str, any]:
    """Create a basic dataset as final fallback"""
    print("📝 Creating basic dataset as fallback...")
    
    data = {
        'query': [
            # Benign queries
            "SELECT * FROM users WHERE username = 'john'",
            "INSERT INTO products (name, price) VALUES ('laptop', 1000)",
            "UPDATE users SET last_login = NOW() WHERE id = 1",
            "DELETE FROM sessions WHERE expires_at < NOW()",
            "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
            "SELECT name, email FROM users WHERE id = 123",
            "SELECT * FROM products WHERE price BETWEEN 10 AND 100",
            "INSERT INTO logs (message, level) VALUES ('test', 'info')",
            "UPDATE profile SET avatar = 'default.jpg' WHERE user_id = 456",
            "SELECT username FROM users WHERE created_at > '2024-01-01'",
            
            # Malicious queries
            "admin' OR '1'='1'",
            "1 UNION SELECT username, password FROM users",
            "1; DROP TABLE users --",
            "1 AND SLEEP(5) --",
            "admin' OR 1=1 --",
            "1 UNION SELECT 1,2,3,4",
            "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a) --",
            "1; WAITFOR DELAY '00:00:05' --",
            "1 UNION SELECT table_name FROM information_schema.tables",
            "1' OR '1'='1' /*",
        ],
        'label': ['benign'] * 10 + ['malicious'] * 10
    }
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_path = "datasets/complete_sql_dataset.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Created basic dataset with {len(df)} samples")
    return {
        "path": output_path,
        "size": len(df),
        "type": "basic_fallback"
    }

def show_dataset_info():
    """Show information about available datasets"""
    print("\n📊 Available Datasets:")
    datasets = [
        "datasets/combined_sql_dataset.csv",
        "datasets/complete_sql_dataset.csv", 
        "datasets/Modified_SQL_Dataset.csv"
    ]
    
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path)
                benign_count = len(df[df['label'].astype(str).str.lower() == 'benign']) if 'label' in df.columns else 'N/A'
                malicious_count = len(df[df['label'].astype(str).str.lower() == 'malicious']) if 'label' in df.columns else 'N/A'
                
                print(f"   📁 {os.path.basename(dataset_path)}:")
                print(f"      Queries: {len(df)}")
                print(f"      Columns: {list(df.columns)}")
                if benign_count != 'N/A':
                    print(f"      Benign: {benign_count}, Malicious: {malicious_count}")
                print()
            except Exception as e:
                print(f"   📁 {os.path.basename(dataset_path)}: Error reading - {e}")

if __name__ == "__main__":
    # Initialize environment first
    initialize_environment()
    
    # Show dataset information
    show_dataset_info()
    
    # Start the FastAPI application
    print("🎯 Starting FastAPI server...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )