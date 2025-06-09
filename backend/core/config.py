from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # API Configuration
    HOST: str = "localhost"
    PORT: int = 8000
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Database
    DATABASE_URL: str = "postgresql://perfect10k:password@localhost:5432/perfect10k"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # External APIs
    SRTM_DATA_PATH: str = "/path/to/srtm/data"
    
    # ML/Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_CACHE_TTL: int = 3600  # 1 hour
    
    # Route Planning
    DEFAULT_TARGET_DISTANCE: int = 8000  # 8km for 10k steps
    DEFAULT_TOLERANCE: int = 1000  # 1km tolerance
    MAX_ROUTE_COMPLEXITY: int = 1000  # Max nodes in route search
    
    class Config:
        env_file = ".env"


settings = Settings()