


from pathlib import Path
import os
from dotenv import load_dotenv

# Configuration
class Config:
    # Base paths
    ROOT_DIR = Path('/content/drive/MyDrive/')
    DATA_DIR = ROOT_DIR / "agri-llm/data_agri_llm/kvk_pop/LAKSHADWEEP/lakshadweep/lakshadweep"
    VECTOR_STORE_DIR = ROOT_DIR / "vector_store"

    # Model configurations
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "google/flan-t5-large"

    # Search configurations
    TOP_K_RESULTS = 3
    

def setup_directory_structure():
    """Create necessary directories if they don't exist."""
    directories = [
        Config.DATA_DIR,
        Config.VECTOR_STORE_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_environment():
    """Set up the environment, including loading environment variables"""

    # Load environment variables from a .env file if available
    load_dotenv()

    # Validate that the token is set
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise EnvironmentError("Hugging Face API token is missing")

    print("Environment setup complete.")
    

