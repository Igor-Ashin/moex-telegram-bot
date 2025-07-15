import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Settings:
    """Настройки приложения"""
    
    # Telegram
    telegram_token: str = os.getenv("TELEGRAM_TOKEN")
    webhook_url: str = os.getenv("WEBHOOK_URL", "https://moex-telegram-bot-sra8.onrender.com")
    
    # API Keys
    tinkoff_token: str = os.getenv("TINKOFF_API_TOKEN")
    
    # Rate Limiting
    max_requests_per_minute: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10"))
    
    # Cache
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL", "300"))  # 5 минут
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Filters
    min_turnover: float = float(os.getenv("MIN_TURNOVER", "50000000"))  # 50 млн
    
    # Pagination
    tickers_per_page: int = 10
    
    def __post_init__(self):
        if not self.telegram_token:
            raise ValueError("TELEGRAM_TOKEN is required")

# Глобальная конфигурация
settings = Settings()