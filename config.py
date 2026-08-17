from functools import lru_cache 
from pydantic_settings import BaseSettings , SettingsConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY : str
    LLM_MODEL: str = "your_model_name"
    FAQ_file :str
    bank_branch_file :str  
    embedding_model :str 

    model_config = SettingsConfigDict(
        env_file= ".env",
        env_file_encoding= "utf-8",
        case_sensitive= False,
        extra="ignore",
    )
@lru_cache
def get_setting() -> Settings:
    return Settings()