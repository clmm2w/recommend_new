from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Dict, Any
from datetime import datetime

from recommend.config import logger
from recommend.models.data_cache import DataCache
from recommend.services.recommendation import record_recommendation_log, generate_recommendation_explanations
from recommend.services.ab_testing import get_ab_test_recommendations

router = APIRouter(
    prefix="",
    tags=["recommendations"]
)

# 依赖函数：获取数据缓存对象
async def get_data_cache(force_update: bool = False):
    if data_cache.needs_update(force_update):
        data_cache.update()
    return data_cache

@router.get("/recommend")
async def recommend_for_user(
    user_id: int, 
    limit: int = Query(10, ge=1, le=50),
    diversify: bool = Query(False),
    explain: bool = Query(False),
    ab_test: bool = Query(True),
    cache: DataCache = Depends(get_data_cache)
):
    """为指定用户推荐服务"""
    logger.info(f"为用户 {user_id} 推荐服务，限制 {limit} 条，多样性: {diversify}, 解释: {explain}, A/B测试: {ab_test}")
    
    # 如果是新用户或没有行为记录，返回热门服务
    if user_id not in cache.user_service_matrix:
        logger.info(f"用户 {user_id} 没有行为记录，返回热门服务")
        recommended_ids = [str(sid) for sid in cache.popular_services[:limit]]
        
        # 记录推荐日志（热门推荐）
        record_recommendation_log(user_id, recommended_ids, source_type="popular", 
                                 algorithm="popularity", reason="新用户或无行为记录")
        
        response = {"user_id": user_id, "items": recommended_ids}
    
        # 如果需要解释，添加简单解释
        if explain:
            explanations = {sid: "这是一个热门服务" for sid in recommended_ids}
            response["explanations"] = explanations
        
        return response
    
    # 根据是否使用A/B测试选择推荐方法
    if ab_test:
        recommendations = get_ab_test_recommendations(user_id, cache, limit, diversify)
        algorithm = "ab_test"
        reason = "A/B测试推荐"
    else:
        # 获取当前时间
        current_time = datetime.now().time()
        
        # 获取时间感知推荐
        from recommend.services.recommendation import get_time_aware_recommendations
        recommendations = get_time_aware_recommendations(user_id, cache, current_time, limit * 2)
        
        # 如果需要多样性，应用多样性增强
        if diversify:
            from recommend.services.diversity import get_diversified_recommendations
            recommendations = get_diversified_recommendations(recommendations, cache.services, limit)
            algorithm = "time_aware_diversified"
            reason = "基于时间的多样化推荐"
        else:
            recommendations = recommendations[:limit]
            algorithm = "time_aware"
            reason = "基于时间的推荐"
    
    # 提取推荐的服务ID和分数
    recommended_ids = [str(r[0]) for r in recommendations]
    recommendation_scores = {r[0]: r[1] for r in recommendations}
    
    # 记录推荐日志
    record_recommendation_log(user_id, recommended_ids, scores=recommendation_scores, 
                             source_type="personalized", algorithm=algorithm, reason=reason)
    
    # 构建响应
    response = {"user_id": user_id, "items": recommended_ids}
    
    # 如果需要解释，生成推荐解释
    if explain:
        explanations = generate_recommendation_explanations(user_id, recommendations, cache)
        response["explanations"] = explanations
    
    return response

@router.get("/recommend/similar")
async def similar_services(
    service_id: int,
    limit: int = Query(10, ge=1, le=50),
    cache: DataCache = Depends(get_data_cache)
):
    """获取相似服务"""
    logger.info(f"获取服务 {service_id} 的相似服务，限制 {limit} 条")
    
    # 检查服务是否存在
    if service_id not in cache.services:
        raise HTTPException(status_code=404, detail="服务不存在")
    
    # 获取相似服务
    if service_id in cache.service_similarity_matrix:
        similar_services = cache.service_similarity_matrix[service_id]
        sorted_similar = sorted(similar_services.items(), key=lambda x: x[1], reverse=True)
        
        # 排除自身
        similar_ids = [str(s[0]) for s in sorted_similar if s[0] != service_id][:limit]
        
        return {"service_id": service_id, "items": similar_ids}
    else:
        # 如果没有相似度数据，返回同类别的服务
        category_id = cache.services[service_id]['category_id']
        same_category = [str(s_id) for s_id, s in cache.services.items() 
                        if s['category_id'] == category_id and s_id != service_id]
        
        return {"service_id": service_id, "items": same_category[:limit]}

@router.get("/recommend/trending")
async def trending_services(
    limit: int = Query(10, ge=1, le=50),
    cache: DataCache = Depends(get_data_cache)
):
    """获取热门服务"""
    logger.info(f"获取热门服务，限制 {limit} 条")
    return {"items": [str(sid) for sid in cache.popular_services[:limit]]} 