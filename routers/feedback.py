from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from recommend.config import logger
from recommend.models.data_cache import DataCache
from recommend.utils.db import execute_query

router = APIRouter(
    prefix="",
    tags=["feedback"]
)

# 依赖函数：获取数据缓存对象
async def get_data_cache(force_update: bool = False):
    if data_cache.needs_update(force_update):
        data_cache.update()
    return data_cache

@router.post("/recommendation/feedback")
async def recommendation_feedback(
    user_id: int,
    service_id: int,
    is_clicked: bool = False,
    feedback_type: Optional[str] = None
):
    """记录推荐反馈"""
    logger.info(f"收到推荐反馈：用户{user_id}, 服务{service_id}, 点击={is_clicked}, 类型={feedback_type}")
    
    try:
        # 先检查是否有最近的推荐记录
        check_query = """
        SELECT id FROM recommendation_log 
        WHERE user_id = %s AND service_id = %s
        ORDER BY created_at DESC LIMIT 1
        """
        existing_record = execute_query(check_query, (user_id, service_id))
        
        if existing_record:
            # 如果有记录，更新它
            record_id = existing_record[0]['id']
            update_query = """
            UPDATE recommendation_log 
            SET is_clicked = %s,
                algorithm = %s,
                reason = %s
            WHERE id = %s
            """
            affected_rows = execute_query(update_query, 
                                        (1 if is_clicked else 0, "feedback", f"用户反馈: {feedback_type}", record_id),
                                        fetch=False)
            logger.info(f"已更新推荐记录 ID: {record_id}")
        else:
            # 如果没有记录，创建一个新记录
            insert_query = """
            INSERT INTO recommendation_log 
            (user_id, service_id, score, is_clicked, algorithm, reason, created_at)
            VALUES (%s, %s, 0.0, %s, %s, %s, NOW())
            """
            reason = f"用户直接反馈: {feedback_type}"
            affected_rows = execute_query(insert_query, 
                                        (user_id, service_id, 1 if is_clicked else 0, "feedback", reason),
                                        fetch=False)
            logger.info(f"已创建新的推荐反馈记录，用户ID: {user_id}, 服务ID: {service_id}")
        
        return {"success": True, "updated_rows": affected_rows}
    except Exception as e:
        logger.error(f"记录推荐反馈失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-behavior")
async def update_behavior(behavior: dict, cache: DataCache = Depends(get_data_cache)):
    """处理用户行为更新"""
    logger.info(f"收到用户行为更新：{behavior}")
    
    try:
        user_id = behavior.get('userId')
        service_id = behavior.get('serviceId')
        behavior_type = behavior.get('behaviorType')
        duration = behavior.get('duration', 0)
        
        if not user_id or not service_id or not behavior_type:
            logger.error("行为数据不完整")
            return {"success": False, "error": "缺少必要参数"}
        
        # 将行为数据保存到数据库（可选，因为Java端已经保存）
        try:
            # 检查是否已经存在相同记录（避免重复）
            query = """
            SELECT id FROM user_behavior 
            WHERE user_id = %s AND service_id = %s AND behavior_type = %s
            AND created_at > DATE_SUB(NOW(), INTERVAL 1 MINUTE)
            """
            exists = execute_query(query, (user_id, service_id, behavior_type))
            
            # 如果1分钟内没有相同记录，则插入
            if not exists:
                extra_data = behavior.get('extraData')
                query = """
                INSERT INTO user_behavior (user_id, service_id, behavior_type, duration, extra_data, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                """
                execute_query(query, (user_id, service_id, behavior_type, duration, extra_data), fetch=False)
                logger.info(f"已保存用户行为：用户{user_id}，服务{service_id}，行为{behavior_type}")
        except Exception as e:
            logger.error(f"保存行为数据失败: {e}")
        
        # 判断是否需要更新缓存
        if cache.last_update:
            # 添加行为到内存中
            behavior_record = {
                'user_id': user_id,
                'service_id': service_id,
                'behavior_type': behavior_type,
                'duration': duration
            }
            
            if user_id not in cache.user_behaviors:
                cache.user_behaviors[user_id] = []
            
            cache.user_behaviors[user_id].append(behavior_record)
            
            # 更新用户-服务交互矩阵
            from recommend.config import BEHAVIOR_WEIGHTS
            
            if user_id not in cache.user_service_matrix:
                cache.user_service_matrix[user_id] = {}
            
            if service_id not in cache.user_service_matrix[user_id]:
                cache.user_service_matrix[user_id][service_id] = 0
            
            weight = BEHAVIOR_WEIGHTS.get(behavior_type, 1.0)
            if behavior_type == 'view' and duration > 0:
                time_factor = min(duration / 60, 5)  # 最多5分钟
                cache.user_service_matrix[user_id][service_id] += weight * (1 + time_factor)
            elif behavior_type == 'unfavorite':
                # 对于取消收藏，确保不会将分数降到负值
                cache.user_service_matrix[user_id][service_id] = max(0, cache.user_service_matrix[user_id][service_id] + weight)
                logger.info(f"用户{user_id}取消收藏服务{service_id}，更新交互分数")
            else:
                cache.user_service_matrix[user_id][service_id] += weight
            
            logger.info(f"已更新用户{user_id}的交互矩阵")
        
        return {"success": True}
    except Exception as e:
        logger.error(f"处理用户行为失败: {str(e)}")
        return {"success": False, "error": str(e)} 