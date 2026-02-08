from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import time

from recommend.config import logger
from recommend.models.data_cache import DataCache
from recommend.routers import recommend, feedback, debug

# 创建FastAPI应用
app = FastAPI(title="ezServe推荐系统", description="为ezServe平台提供个性化推荐服务")

# 跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建数据缓存实例
data_cache = DataCache()

# 注册路由
app.include_router(recommend.router)
app.include_router(feedback.router)
app.include_router(debug.router)


# 后台定时更新任务
def run_scheduler(cache, interval_seconds=3600):
    """后台线程：定时更新数据缓存"""
    logger.info(f"后台更新调度器已启动，更新间隔: {interval_seconds}秒")
    while True:
        try:
            # 等待指定间隔
            time.sleep(interval_seconds)
            logger.info("后台任务：开始定时更新缓存...")
            cache.update()
            logger.info("后台任务：缓存更新完成")
        except Exception as e:
            logger.error(f"后台更新任务出错: {str(e)}")
            # 出错后等待一分钟再重试，避免死循环刷屏
            time.sleep(60)


@app.get("/")
async def root():
    return {"message": "ezServe推荐系统API", "status": "running"}


# 初始化数据与后台任务
@app.on_event("startup")
async def startup_event():
    logger.info("推荐系统启动...")

    # 1. 初始检查：如果内存中没有有效数据（首次启动且无文件缓存），强制同步执行一次更新
    # 这样可以保证服务启动后立刻可用，虽然启动时间会变长
    if not data_cache.last_update:
        logger.info("未检测到有效缓存，正在执行首次初始化（这可能需要几十秒）...")
        data_cache.update()
    else:
        logger.info(f"检测到历史缓存（最后更新: {data_cache.last_update}），已跳过初始化训练，服务秒级启动")

    # 2. 启动后台守护线程进行定时更新
    # daemon=True 表示主程序退出时，这个线程也会自动退出
    scheduler_thread = threading.Thread(
        target=run_scheduler,
        args=(data_cache, 3600),  # 1小时更新一次
        daemon=True
    )
    scheduler_thread.start()

    logger.info("系统初始化完成")


# 向路由模块提供数据缓存实例
recommend.data_cache = data_cache
feedback.data_cache = data_cache
debug.data_cache = data_cache

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)