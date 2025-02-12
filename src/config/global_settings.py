import os
from dotenv import load_dotenv # type: ignore

# Load environment variables từ .env
load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGODB_DB")
MONGO_STRUCTURED_COLLECTION = os.getenv("MONGODB_COLL")
MONGO_VECTOR_COLLECTION = os.getenv("MONGODB_VECTOR_COLL_LANGCHAIN")
MONGO_VECTOR_INDEX = os.getenv("MONGODB_VECTOR_INDEX")

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
DATA_UPLOADS_DIR = os.path.join(os.getcwd(), "data/uploads/")
LOGS_DIR = os.path.join(os.getcwd(), "data/logs/")
VECTOR_STORE_DIR = os.path.join(os.getcwd(), "data/vector_store/")

# Ensure directories exist
os.makedirs(DATA_UPLOADS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

from pymongo import MongoClient # type: ignore
from src.config.global_settings import MONGO_URI, MONGO_DB_NAME, MONGO_STRUCTURED_COLLECTION

# Kết nối với MongoDB
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
structured_collection = db[MONGO_STRUCTURED_COLLECTION]
