import os
import logging

#注意！！！不同项目要改数据库！！！
# 数据库配置
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "123456"),
    "database": os.environ.get("DB_NAME", "ezserve_db_not_sport"),
    "port": int(os.environ.get("DB_PORT", "3306"))
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("recommend.log")
    ]
)
logger = logging.getLogger("recommend-service")

# 行为权重配置
BEHAVIOR_WEIGHTS = {
    'view': 1.0,
    'click': 2.0,
    'favorite': 3.0,
    'unfavorite': -3.0,
    'order': 5.0
}

# 其他配置
CACHE_UPDATE_INTERVAL = 3600  # 缓存更新时间间隔(秒) 