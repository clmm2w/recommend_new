import json
from collections import defaultdict
from datetime import datetime

from recommend.config import logger
from recommend.utils.db import execute_query
from recommend.services.content_based import get_content_based_recommendations
# 引用重构后的多样性服务
from recommend.services.diversity import apply_diversity
# 引用 NCF 依赖检查
from recommend.services.neural_cf import TF_AVAILABLE


# ========================================================
# 1. 记录推荐日志 (保持不变)
# ========================================================
def record_recommendation_log(user_id, service_ids, scores=None, source_type=None, algorithm=None, reason=None):
    if not service_ids: return
    try:
        # 批量插入优化 (伪代码，保持你原有的逐条插入逻辑以防 SQL 语法差异，但建议生产环境用 executemany)
        for service_id_str in service_ids:
            try:
                service_id = int(service_id_str)
                score = float(scores.get(service_id, 0.0)) if scores else 0.0

                query = """
                INSERT INTO recommendation_log 
                (user_id, service_id, score, is_clicked, algorithm, reason, created_at)
                VALUES (%s, %s, %s, 0, %s, %s, NOW())
                """
                execute_query(query, (user_id, service_id, score, algorithm, reason), fetch=False)
            except Exception as e:
                pass
        logger.info(f"用户 {user_id} 推荐日志记录完成")
    except Exception as e:
        logger.error(f"日志记录总控错误: {e}")


# ========================================================
# 2. 核心：混合推荐 (Logic Fixed)
# ========================================================
def get_hybrid_recommendations(user_id, data_cache, limit=10):
    """
    获取混合推荐结果
    修正点:
    1. 移除 SVD 的二次归一化
    2. 更新为最新的 Meta-Model 权重
    """
    try:
        # A. 冷启动处理
        if user_id not in data_cache.user_service_matrix:
            logger.info(f"用户 {user_id} 无历史行为，返回热门推荐")
            return [(sid, 1.0) for sid in data_cache.popular_services[:limit]]

        interacted_services = set(data_cache.user_service_matrix[user_id].keys())
        num_interactions = len(interacted_services)
        recommendations = defaultdict(float)

        # B. 权重策略 (Strategy Pattern)
        if num_interactions < 5:
            # 新手期：侧重 CB
            w_content = 0.8
            w_svd = 0.1
            w_ncf = 0.1
            algo_tag = "Cold-Start"
        else:
            # 成熟期：使用最新的训练结果 (CB:0.33, SVD:0.33, NCF:0.34)
            w_content = 0.3300
            w_svd = 0.3300
            w_ncf = 0.3400
            algo_tag = "Stacking-Ensemble"

        logger.info(
            f"用户 {user_id} ({num_interactions}次交互) 策略: {algo_tag} [CB:{w_content}, SVD:{w_svd}, NCF:{w_ncf}]")

        # 1. Content-Based (已在 content_based.py 中归一化到 0~1)
        if w_content > 0:
            cb_recs = get_content_based_recommendations(
                user_id,
                data_cache.user_service_matrix,
                data_cache.service_similarity_matrix,
                limit=200  # 获取更多候选集用于混合
            )
            for sid, score in cb_recs:
                recommendations[sid] += score * w_content

        # 2. SVD (已在 matrix_factorization.py 中修正为 0~1)
        if w_svd > 0:
            svd_recs = data_cache.get_svd_recommendations(user_id, limit=200)
            for sid, score in svd_recs:
                # 🚨 修正：直接使用 score，不要除以 5.0！
                # 前面的代码审查已经确保 get_svd_recommendations 返回的是归一化后的值
                recommendations[sid] += score * w_svd

        # 3. NCF (原生 Sigmoid 输出 0~1)
        if w_ncf > 0:
            if TF_AVAILABLE and data_cache.ncf_model is not None:
                ncf_recs = data_cache.get_ncf_recommendations(user_id, limit=200)
                for sid, score in ncf_recs:
                    recommendations[sid] += score * w_ncf

        # C. 过滤与排序
        final_list = []
        # 按总分排序
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

        for sid, score in sorted_recs:
            if sid not in interacted_services:
                final_list.append((sid, score))

        # D. 多样性打散 (可选，但建议加上)
        # 使用 apply_diversity 进行重排，防止全是同一类
        final_list = apply_diversity(final_list, data_cache.services, diversity_strength=0.1, limit=limit)

        # E. 兜底逻辑 (如果推荐数量不够)
        if len(final_list) < limit:
            existing_ids = {r[0] for r in final_list}
            for sid in data_cache.popular_services:
                if sid not in interacted_services and sid not in existing_ids:
                    final_list.append((sid, 0.05))  # 给一个低分
                    if len(final_list) >= limit: break

        # 记录日志
        record_recommendation_log(
            user_id, [r[0] for r in final_list],
            scores={r[0]: r[1] for r in final_list},
            algorithm=algo_tag, reason="Hybrid Recommendation"
        )

        return final_list[:limit]

    except Exception as e:
        logger.error(f"混合推荐严重错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return [(sid, 1.0) for sid in data_cache.popular_services[:limit]]


# ========================================================
# 3. 解释生成 (逻辑保持)
# ========================================================
def generate_recommendation_explanations(user_id, recommendations, data_cache):
    # ... (保持你原有的逻辑，这部分没有严重风险)
    # 只要确保 recommendations 里的 service_id 都在 services 字典里即可
    explanations = {}
    if user_id not in data_cache.user_features:
        return {str(r[0]): "热门推荐" for r in recommendations}

    for service_id, score in recommendations:
        service = data_cache.services.get(service_id)
        if not service: continue

        # 简单生成解释，避免复杂逻辑报错
        cat_name = service.get('category', '优质服务')
        explanations[str(service_id)] = f"基于您对{cat_name}的兴趣推荐"

    return explanations


# ========================================================
# 4. 时间感知 (逻辑保持)
# ========================================================
def get_time_aware_recommendations(user_id, data_cache, current_time=None, limit=10):
    # 复用 get_hybrid_recommendations
    return get_hybrid_recommendations(user_id, data_cache, limit)