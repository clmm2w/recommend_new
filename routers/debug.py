from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
from recommend.config import logger
from recommend.models.data_cache import DataCache
from recommend.utils.db import execute_query
from recommend.services.recommendation import generate_recommendation_explanations

router = APIRouter(
    prefix="/debug",
    tags=["debug"]
)

# 依赖函数：获取数据缓存对象
async def get_data_cache(force_update: bool = False):
    if data_cache.needs_update(force_update):
        data_cache.update()
    return data_cache

@router.get("/model-status")
async def model_status(cache: DataCache = Depends(get_data_cache)):
    """查看模型状态"""
    svd_status = "未构建"
    if cache.svd_model:
        svd_status = {
            "潜在因子数": len(cache.svd_model['sigma']),
            "用户数": len(cache.svd_model['user_idx']),
            "服务数": len(cache.svd_model['service_idx'])
        }
    
    from recommend.services.neural_cf import TF_AVAILABLE
    ncf_status = "未构建"
    if TF_AVAILABLE and cache.ncf_model:
        ncf_status = {
            "模型类型": str(type(cache.ncf_model)),
            "已训练": True
        }
    elif not TF_AVAILABLE:
        ncf_status = "TensorFlow未安装"
    
    return {
        "last_update": str(cache.last_update),
        "services_count": len(cache.services),
        "users_count": len(cache.user_behaviors),
        "user_service_matrix_size": len(cache.user_service_matrix),
        "service_similarity_matrix_size": len(cache.service_similarity_matrix),
        "popular_services_count": len(cache.popular_services),
        "popular_services_top10": cache.popular_services[:10],
        "svd_model": svd_status,
        "ncf_model": ncf_status,
        "user_features_count": len(cache.user_features),
        "service_features_count": len(cache.service_features)
    }

@router.get("/recommendation-logs")
async def recommendation_logs(
    user_id: Optional[int] = None,
    service_id: Optional[int] = None,
    limit: int = 20
):
    """查看推荐日志记录"""
    try:
        # 构建查询条件
        conditions = []
        params = []
        
        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)
        
        if service_id:
            conditions.append("service_id = %s")
            params.append(service_id)
        
        # 构建SQL查询
        query = "SELECT * FROM recommendation_log"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        # 执行查询
        logs = execute_query(query, params)
        
        # 处理datetime对象以便JSON序列化
        for log in logs:
            if 'created_at' in log and log['created_at']:
                log['created_at'] = log['created_at'].isoformat()
        
        return {
            "count": len(logs),
            "logs": logs
        }
    except Exception as e:
        logger.error(f"查询推荐日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}/behaviors")
async def get_user_behaviors(
    user_id: int,
    cache: DataCache = Depends(get_data_cache)
):
    """获取用户行为数据详情"""
    logger.info(f"获取用户 {user_id} 的行为数据")
    
    result = {
        "user_id": user_id,
        "behaviors": [],
        "interaction_matrix": {},
        "recent_recommendations": [],
        "recommendation_feedbacks": [],
        "recommendation_source": "unknown"
    }
    
    try:
        # 获取用户行为数据
        query = """
        SELECT id, user_id, service_id, behavior_type, duration, extra_data, created_at
        FROM user_behavior
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 50
        """
        behaviors = execute_query(query, (user_id,))
        
        # 处理日期格式以便JSON序列化
        for behavior in behaviors:
            if 'created_at' in behavior and behavior['created_at']:
                behavior['created_at'] = behavior['created_at'].isoformat()
            # 添加服务名称
            service_id = behavior['service_id']
            if service_id in cache.services:
                behavior['service_name'] = cache.services[service_id]['name']
        
        result["behaviors"] = behaviors
        
        # 获取交互矩阵数据
        if user_id in cache.user_service_matrix:
            interaction_data = []
            for service_id, score in cache.user_service_matrix[user_id].items():
                service_info = {
                    "service_id": service_id,
                    "interaction_score": score
                }
                if service_id in cache.services:
                    service_info["service_name"] = cache.services[service_id]['name']
                    service_info["category"] = cache.services[service_id]['category']
                interaction_data.append(service_info)
            
            # 按交互分数排序
            interaction_data.sort(key=lambda x: x["interaction_score"], reverse=True)
            result["interaction_matrix"] = interaction_data
        
        # 获取最近的推荐记录
        query = """
        SELECT id, service_id, score, is_clicked, algorithm, reason, created_at
        FROM recommendation_log
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 20
        """
        recommendations = execute_query(query, (user_id,))
        
        # 处理日期格式
        for rec in recommendations:
            if 'created_at' in rec and rec['created_at']:
                rec['created_at'] = rec['created_at'].isoformat()
            # 添加服务名称
            service_id = rec['service_id']
            if service_id in cache.services:
                rec['service_name'] = cache.services[service_id]['name']
        
        result["recent_recommendations"] = recommendations
        
        # 确定推荐来源
        if user_id in cache.user_service_matrix and len(cache.user_service_matrix[user_id]) > 0:
            result["recommendation_source"] = "personalized"
        else:
            result["recommendation_source"] = "popular"
        
        # 获取用户推荐反馈
        query = """
        SELECT r.id, r.user_id, r.service_id, r.algorithm, r.reason, r.created_at
        FROM recommendation_log r
        WHERE r.user_id = %s AND r.reason LIKE '%用户反馈%'
        ORDER BY r.created_at DESC
        LIMIT 20
        """
        feedbacks = execute_query(query, (user_id,))
        
        # 处理日期格式
        for feedback in feedbacks:
            if 'created_at' in feedback and feedback['created_at']:
                feedback['created_at'] = feedback['created_at'].isoformat()
            # 添加服务名称
            service_id = feedback['service_id']
            if service_id in cache.services:
                feedback['service_name'] = cache.services[service_id]['name']
        
        result["recommendation_feedbacks"] = feedbacks
        
        return result
    except Exception as e:
        logger.error(f"获取用户行为数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/service/{service_id}/similarity")
async def get_service_similarity(
    service_id: int,
    limit: int = 20,
    cache: DataCache = Depends(get_data_cache)
):
    """获取服务相似度详情"""
    logger.info(f"获取服务 {service_id} 的相似度详情")
    
    # 检查服务是否存在
    if service_id not in cache.services:
        raise HTTPException(status_code=404, detail="服务不存在")
    
    result = {
        "service_id": service_id,
        "service_name": cache.services[service_id]['name'],
        "service_details": cache.services[service_id],
        "similar_services": []
    }
    
    # 获取相似服务
    if service_id in cache.service_similarity_matrix:
        similar_services = cache.service_similarity_matrix[service_id]
        sorted_similar = sorted(similar_services.items(), key=lambda x: x[1], reverse=True)
        
        # 排除自身并限制数量
        similar_data = []
        for similar_id, similarity in sorted_similar:
            if similar_id != service_id:
                service_info = {
                    "service_id": similar_id,
                    "similarity_score": similarity
                }
                if similar_id in cache.services:
                    service_info["service_name"] = cache.services[similar_id]['name']
                    service_info["category"] = cache.services[similar_id]['category']
                similar_data.append(service_info)
                
                if len(similar_data) >= limit:
                    break
        
        result["similar_services"] = similar_data
    
    return result

@router.get("/ab-test-status")
async def ab_test_status(cache: DataCache = Depends(get_data_cache)):
    """查看A/B测试状态"""
    # 统计各组用户数量
    stats = {}
    for test_name, groups in cache.ab_test_groups.items():
        stats[test_name] = {}
        for group_name in set(groups.values()):
            stats[test_name][group_name] = sum(1 for g in groups.values() if g == group_name)
    
    return {
        "active_tests": list(cache.ab_test_groups.keys()),
        "user_counts": stats,
        "total_users": len(cache.user_features)
    }

@router.get("/explain-recommendation")
async def explain_recommendation(
    user_id: int,
    service_id: int,
    cache: DataCache = Depends(get_data_cache)
):
    """获取单个推荐的详细解释"""
    if service_id not in cache.services:
        raise HTTPException(status_code=404, detail="服务不存在")
    
    service = cache.services[service_id]
    
    # 基础服务信息
    result = {
        "service_id": service_id,
        "service_name": service.get('name', ''),
        "category": service.get('category', ''),
        "rating": float(service.get('rating', 0) or 0),
        "review_count": int(service.get('review_count', 0) or 0),
    }
    
    # 如果用户不存在，只返回基础信息
    if user_id not in cache.user_features:
        result["explanation"] = "这是一个推荐服务"
        return result
    
    # 获取用户与该服务的交互
    interaction_score = 0
    if user_id in cache.user_service_matrix and service_id in cache.user_service_matrix[user_id]:
        interaction_score = cache.user_service_matrix[user_id][service_id]
    
    result["user_interaction_score"] = interaction_score
    
    # 获取用户特征
    user_features = cache.user_features[user_id]
    
    # 获取用户的类别偏好
    category_prefs = {}
    for key, value in user_features.items():
        if key.startswith('category_pref_'):
            category_id = int(key.replace('category_pref_', ''))
            category_prefs[category_id] = value
    
    # 计算类别匹配度
    category_match = 0
    category_id = service.get('category_id')
    if category_id and category_id in category_prefs:
        category_match = category_prefs[category_id]
    
    result["category_match"] = category_match
    
    # 查找相似服务
    similar_services = []
    for interacted_id, interaction_score in cache.user_service_matrix.get(user_id, {}).items():
        if interacted_id in cache.service_similarity_matrix and service_id in cache.service_similarity_matrix[interacted_id]:
            similarity = cache.service_similarity_matrix[interacted_id][service_id]
            if similarity > 0.5 and interaction_score > 2:
                if interacted_id in cache.services:
                    similar_services.append({
                        "service_id": interacted_id,
                        "service_name": cache.services[interacted_id]['name'],
                        "similarity": similarity,
                        "user_interaction": interaction_score
                    })
    
    # 按相似度排序
    similar_services.sort(key=lambda x: x["similarity"] * x["user_interaction"], reverse=True)
    result["similar_services"] = similar_services[:5]
    
    # 生成文本解释
    explanations = generate_recommendation_explanations(user_id, [(service_id, 1.0)], cache)
    result["explanation"] = explanations.get(str(service_id), "这是一个推荐服务")
    
    return result

@router.post("/train")
async def train_model(cache: DataCache = Depends(get_data_cache)):
    """触发模型训练"""
    logger.info("触发模型训练")
    try:
        # 强制更新缓存数据
        cache.update()
        return {"success": True, "message": "模型训练成功"}
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}") 
    
