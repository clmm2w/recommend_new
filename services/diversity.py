from collections import defaultdict
from recommend.config import logger


def apply_diversity(recommendations, services, diversity_strength=0.1, limit=10):
    """
    应用多样性重排序 (MMR 简化版)

    Args:
        recommendations: [(service_id, score), ...] 列表，score 范围 0~1
        services: 服务详情字典
        diversity_strength: 多样性惩罚系数。建议 0.1~0.2。
                            如果 score 是 0~1，0.5 的惩罚太重了，会直接把高分同类打入冷宫。
        limit: 返回数量限制
    """
    if not recommendations:
        return []

    category_counts = defaultdict(int)
    diversified = []
    # 复制列表，避免修改原数据
    remaining = list(recommendations)

    while remaining and len(diversified) < limit:
        scores = []
        for item in remaining:
            service_id, original_score = item

            # 获取类别
            category_id = None
            if services and service_id in services:
                category_id = services[service_id].get('category_id')

            # 计算惩罚: 该类别已出现次数 * 强度
            count = category_counts.get(category_id, 0) if category_id else 0
            penalty = count * diversity_strength

            # 边际递减得分 (Marginal Relevance)
            final_score = original_score - penalty
            scores.append((item, final_score))

        # 选出当前得分最高的 (贪心算法)
        if not scores: break

        best_item_tuple, _ = max(scores, key=lambda x: x[1])
        best_service_id = best_item_tuple[0]

        diversified.append(best_item_tuple)
        remaining.remove(best_item_tuple)

        # 更新该类别计数
        if services and best_service_id in services:
            cat_id = services[best_service_id].get('category_id')
            if cat_id:
                category_counts[cat_id] += 1

    return diversified


# 兼容旧代码调用的别名
get_diversified_recommendations = apply_diversity