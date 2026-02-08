import json
from collections import defaultdict
from datetime import datetime

from recommend.config import logger
from recommend.utils.db import execute_query
from recommend.services.content_based import get_content_based_recommendations
from recommend.services.diversity import apply_diversity


# ========================================================
# 1. 记录推荐日志函数
# ========================================================
def record_recommendation_log(user_id, service_ids, scores=None, source_type=None, algorithm=None, reason=None):
    """记录推荐日志到数据库"""
    if not service_ids:
        return

    try:
        for i, service_id_str in enumerate(service_ids):
            try:
                service_id = int(service_id_str)
                score = 0.0
                if scores and service_id in scores:
                    score = float(scores[service_id])

                query = """
                INSERT INTO recommendation_log 
                (user_id, service_id, score, is_clicked, algorithm, reason, created_at)
                VALUES (%s, %s, %s, 0, %s, %s, NOW())
                """
                execute_query(query, (user_id, service_id, score, algorithm, reason), fetch=False)
            except Exception as e:
                logger.error(f"记录推荐日志失败 (服务ID: {service_id_str}): {e}")

        logger.info(f"已记录 {len(service_ids)} 条推荐日志，用户ID: {user_id}")
    except Exception as e:
        logger.error(f"记录推荐日志数据库错误: {e}")


# ========================================================
# 2. 核心函数：混合推荐结果（集成 Learning to Rank 权重）
# ========================================================
def get_hybrid_recommendations(user_id, data_cache, limit=10):
    """获取混合推荐结果（集成自动化权重优化策略）"""
    try:
        # 如果用户没有行为记录，返回热门推荐（完全冷启动）
        if user_id not in data_cache.user_service_matrix:
            return [(sid, 1.0) for sid in data_cache.popular_services[:limit]]

        # 获取用户已交互的服务
        interacted_services = set(data_cache.user_service_matrix[user_id].keys())
        num_interactions = len(interacted_services)

        recommendations = defaultdict(float)

        # --- 方案3：集成 Learning to Rank 自动化权重策略 ---
        if num_interactions < 5:
            # 冷启动/新手期：协同过滤数据不足，依赖内容推荐 (CB)
            w_content = 0.8
            w_svd = 0.1
            w_ncf = 0.1
            algo_tag = "Cold-Start (CB Emphasis)"
            logger.info(f"用户 {user_id} 处于新手期({num_interactions})，采用冷启动避让策略")
        else:
            # 成熟期：使用 train_meta_model.py 训练出的最优权重
            # 权重来源：逻辑回归元模型对 5932 个样本的拟合结果
            w_content = 0.0000  # 实验证明活跃用户不需要CB补充
            w_svd = 0.4097  # SVD 贡献度
            w_ncf = 0.5903  # NCF 贡献度 (主导)
            algo_tag = "Learning-to-Rank (Stacking)"
            logger.info(f"用户 {user_id} 为活跃用户，采用自动化最佳权重组合: SVD:{w_svd}, NCF:{w_ncf}")

        # 1. 获取 Content-Based 推荐得分
        if w_content > 0:
            cf_recommendations = get_content_based_recommendations(
                user_id,
                data_cache.user_service_matrix,
                data_cache.service_similarity_matrix,
                limit * 2
            )
            for service_id, score in cf_recommendations:
                recommendations[service_id] += score * w_content

        # 2. 获取 SVD 推荐得分
        if w_svd > 0:
            svd_recommendations = data_cache.get_svd_recommendations(user_id, limit * 2)
            for service_id, score in svd_recommendations:
                # 归一化：SVD预测分(0-5)除以5转换到0-1
                normalized_score = min(max(score / 5.0, 0), 1.0)
                recommendations[service_id] += normalized_score * w_svd

        # 3. 获取 NCF 推荐得分
        if w_ncf > 0:
            from recommend.services.neural_cf import TF_AVAILABLE
            if TF_AVAILABLE and data_cache.ncf_model is not None:
                ncf_recommendations = data_cache.get_ncf_recommendations(user_id, limit * 2)
                for service_id, score in ncf_recommendations:
                    # NCF输出已是0-1
                    recommendations[service_id] += score * w_ncf

        # 排序并过滤已交互
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        final_recommendations = []
        for service_id, score in sorted_recommendations:
            if service_id not in interacted_services:
                final_recommendations.append((service_id, score))
            if len(final_recommendations) >= limit:
                break

        # 兜底补充
        if len(final_recommendations) < limit:
            for service_id in data_cache.popular_services:
                if service_id not in interacted_services and service_id not in [r[0] for r in final_recommendations]:
                    final_recommendations.append((service_id, 0.05))
                    if len(final_recommendations) >= limit:
                        break

        # 自动记录推荐日志
        rec_ids = [r[0] for r in final_recommendations]
        rec_scores = {r[0]: r[1] for r in final_recommendations}
        record_recommendation_log(
            user_id, rec_ids, scores=rec_scores,
            algorithm=algo_tag, reason=f"基于{algo_tag}策略的个性化推荐"
        )

        return final_recommendations[:limit]
    except Exception as e:
        logger.error(f"获取混合推荐失败: {str(e)}")
        return [(sid, 1.0) for sid in data_cache.popular_services[:limit]]


# ========================================================
# 3. 推荐解释生成
# ========================================================
def generate_recommendation_explanations(user_id, recommendations, data_cache):
    """为推荐结果生成可解释性的文案"""
    explanations = {}
    if user_id not in data_cache.user_features:
        return {str(r[0]): "这是一个热门服务" for r in recommendations}

    user_features = data_cache.user_features[user_id]
    category_prefs = {int(k.replace('category_pref_', '')): v for k, v in user_features.items() if
                      k.startswith('category_pref_')}

    for service_id, score in recommendations:
        if service_id not in data_cache.services:
            explanations[str(service_id)] = "推荐服务"
            continue

        service = data_cache.services[service_id]
        reasons = []

        # 理由A: 类别偏好
        cat_id = service.get('category_id')
        if cat_id in category_prefs and category_prefs[cat_id] > 0.2:
            reasons.append(f"符合您对{service.get('category', '同类')}服务的偏好")

        # 理由B: 高分推荐
        rating = float(service.get('rating', 0) or 0)
        if rating >= 4.5: reasons.append("高分优质推荐")

        # 理由C: 相似历史
        similar_count = 0
        for interacted_id in data_cache.user_service_matrix.get(user_id, {}):
            if interacted_id in data_cache.service_similarity_matrix and service_id in \
                    data_cache.service_similarity_matrix[interacted_id]:
                if data_cache.service_similarity_matrix[interacted_id][service_id] > 0.7:
                    similar_count += 1
        if similar_count > 0: reasons.append("与您感兴趣的服务相似")

        explanations[str(service_id)] = "，".join(reasons[:2]) if reasons else "为您精选的个性化服务"

    return explanations


# ========================================================
# 4. 时间感知推荐
# ========================================================
def get_time_aware_recommendations(user_id, data_cache, current_time=None, limit=10):
    """根据当前时间段调整推荐排序"""
    try:
        base_recommendations = get_hybrid_recommendations(user_id, data_cache, limit * 2)
        if not current_time or user_id not in data_cache.time_patterns:
            return base_recommendations[:limit]

        current_hour = current_time.hour
        user_patterns = data_cache.time_patterns[user_id]

        scored_recommendations = []
        for service_id, base_score in base_recommendations:
            time_score = base_score
            hourly_activity = user_patterns['hourly'][current_hour]
            total_hourly = sum(user_patterns['hourly'])

            if total_hourly > 0 and hourly_activity / total_hourly > 0.1:
                time_score *= (1 + hourly_activity / total_hourly)
            scored_recommendations.append((service_id, time_score))

        scored_recommendations.sort(key=lambda x: x[1], reverse=True)
        return scored_recommendations[:limit]
    except Exception as e:
        logger.error(f"时间感知推荐失败: {e}")
        return base_recommendations[:limit]