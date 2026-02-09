from collections import defaultdict
from recommend.config import logger
import jieba
import numpy as np


def calculate_text_similarity(words1, words2):
    """计算两组预分词列表的相似度 (优化版)"""
    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0.0


def build_service_similarity_matrix(services):
    """构建服务相似度矩阵 (架构级优化)"""
    # 1. 预处理：提前分词，避免在循环中重复计算
    tokenized_descriptions = {}
    for sid, s in services.items():
        desc = s.get('description', '')
        if desc and any('\u4e00' <= char <= '\u9fff' for char in desc):
            tokenized_descriptions[sid] = set(jieba.cut(desc.lower()))
        elif desc:
            tokenized_descriptions[sid] = set(desc.lower().split())
        else:
            tokenized_descriptions[sid] = set()

    service_features = {}
    for service_id, service in services.items():
        # 类别和标签作为基础特征
        features = [f"cat_{service['category_id']}"]
        if service.get('tags'):
            features.extend([f"tag_{t.strip()}" for t in service['tags'].split(',')])
        service_features[service_id] = set(features)

    similarity_matrix = {}
    all_service_ids = list(services.keys())

    for service_id1 in all_service_ids:
        similarity_matrix[service_id1] = {}
        for service_id2 in all_service_ids:
            if service_id1 == service_id2:
                similarity_matrix[service_id1][service_id2] = 1.0
                continue

            # A. 基础属性 Jaccard (类别+标签)
            feat1, feat2 = service_features[service_id1], service_features[service_id2]
            base_sim = len(feat1.intersection(feat2)) / len(feat1.union(feat2)) if len(feat1.union(feat2)) > 0 else 0.0

            # B. 文本描述相似度
            text_sim = calculate_text_similarity(tokenized_descriptions[service_id1],
                                                 tokenized_descriptions[service_id2])

            # C. 综合加权
            # 提高文本权重，降低类别权重，增加区分度
            similarity = (base_sim * 0.5) + (text_sim * 0.5)

            # D. 同类别强制加分 (逻辑保留但减弱)
            if services[service_id1]['category_id'] == services[service_id2]['category_id']:
                similarity = min(1.0, similarity + 0.1)

            similarity_matrix[service_id1][service_id2] = float(similarity)

    logger.info(f"构建了 {len(similarity_matrix)} 个服务的相似度矩阵")
    return similarity_matrix


def get_content_based_recommendations(user_id, user_service_matrix, service_similarity_matrix, limit=10):
    """基于内容的推荐 (量纲归一化优化版)"""
    if user_id not in user_service_matrix:
        return []

    interacted_services = user_service_matrix[user_id]  # {sid: strength}
    recommendations = defaultdict(float)

    for sid, strength in interacted_services.items():
        if sid not in service_similarity_matrix:
            continue

        similar_items = service_similarity_matrix[sid]
        for sim_id, similarity in similar_items.items():
            if sim_id not in interacted_services:
                # 核心修正：取相似度与强度的加权分
                recommendations[sim_id] += similarity * strength

    if not recommendations:
        return []

    # --- 关键修正：将 CB 总分映射到 0~1 区间 ---
    # 使用 Softmax 的简化思路或简单的 Max-Scaling
    # 这里采用 Min-Max 归一化，确保它能和 NCF 的 0.9 概率公平竞争
    items = list(recommendations.items())
    scores = [x[1] for x in items]
    max_score = max(scores)
    min_score = min(scores)

    range_val = max_score - min_score
    if range_val == 0:
        normalized_recs = [(item_id, 0.5) for item_id, _ in items]
    else:
        normalized_recs = [(item_id, (s - min_score) / range_val) for item_id, s in items]

    sorted_recs = sorted(normalized_recs, key=lambda x: x[1], reverse=True)
    return sorted_recs[:limit]