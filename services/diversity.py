from collections import defaultdict
from recommend.config import logger

def get_diversified_recommendations(recommendations, services, limit=10):
    """增加推荐结果的多样性"""
    if not recommendations:
        return []
        
    # 初始化类别计数
    category_counts = defaultdict(int)
    diversified = []
    remaining = list(recommendations)
    
    while remaining and len(diversified) < limit:
        # 计算每个项目的多样性得分
        scores = []
        for item in remaining:
            service_id, original_score = item
            
            # 获取服务类别
            category_id = None
            if service_id in services:
                category_id = services[service_id].get('category_id')
            
            # 类别多样性惩罚
            category_penalty = category_counts.get(category_id, 0) if category_id else 0
            
            # 多样性得分 = 原始分数 - 0.2 * 类别惩罚
            diversity_score = original_score - 0.2 * category_penalty
            scores.append((item, diversity_score))
        
        # 选择多样性得分最高的项目
        if not scores:
            break
            
        best_item, _ = max(scores, key=lambda x: x[1])
        diversified.append(best_item)
        remaining.remove(best_item)
        
        # 更新类别计数
        service_id = best_item[0]
        if service_id in services:
            category_id = services[service_id].get('category_id')
            if category_id:
                category_counts[category_id] += 1
    
    return diversified

def apply_diversity(recommendations, services, diversity_strength=0.5, limit=10):
    """应用多样性增强，根据多样性强度参数"""
    if not recommendations:
        return []
        
    # 初始化类别计数
    category_counts = defaultdict(int)
    diversified = []
    remaining = list(recommendations)
    
    while remaining and len(diversified) < limit:
        # 计算每个项目的多样性得分
        scores = []
        for item in remaining:
            service_id, original_score = item
            
            # 获取服务类别
            category_id = None
            if service_id in services:
                category_id = services[service_id].get('category_id')
            
            # 类别多样性惩罚
            category_penalty = category_counts.get(category_id, 0) if category_id else 0
            
            # 多样性得分 = 原始分数 - 多样性强度 * 类别惩罚
            diversity_score = original_score - diversity_strength * category_penalty
            scores.append((item, diversity_score))
        
        # 选择多样性得分最高的项目
        if not scores:
            break
            
        best_item, _ = max(scores, key=lambda x: x[1])
        diversified.append(best_item)
        remaining.remove(best_item)
        
        # 更新类别计数
        service_id = best_item[0]
        if service_id in services:
            category_id = services[service_id].get('category_id')
            if category_id:
                category_counts[category_id] += 1
    
    return diversified 