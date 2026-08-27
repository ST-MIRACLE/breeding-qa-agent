"""配置管理"""
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    config_path = os.path.join(BASE_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {"api_key": "", "model": "qwen-turbo", "embedding_model": "text-embedding-v2"}

CONFIG = load_config()
API_KEY = CONFIG.get("api_key", "")
CHAT_MODEL = CONFIG.get("model", "qwen-turbo")
EMBEDDING_MODEL = CONFIG.get("embedding_model", "text-embedding-v2")

KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
DOCS_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "docs")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
STRUCTURED_KB_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "breeding_knowledge.json")

BREEDING_DATA_DIR = "D:\\育种数据\\数据"
