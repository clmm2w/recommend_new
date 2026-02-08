from collections import defaultdict
from recommend.config import logger
import jieba

def calculate_text_similarity(text1, text2):
    """计算两段文本的相似度"""
    if not text1 or not text2:
        return 0.0
        
    # 改进的词袋模型，支持中文分词
    def get_words(text):
        if isinstance(text, str):
            # 使用jieba进行中文分词
            if any('\u4e00' <= char <= '\u9fff' for char in text):  # 检测是否包含中文字符
                return set(jieba.cut(text.lower()))  # 中文分词
            else:
                return set(text.lower().split())  # 英文分词保持原样
        return set()
        
    words1 = get_words(text1)
    words2 = get_words(text2)
    
    # 计算Jaccard相似度
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union > 0:
        return intersection / union
    return 0.0

def build_service_similarity_matrix(services):
    """构建服务相似度矩阵"""
    # 基于内容的相似度计算
    service_features = {}
    for service_id, service in services.items():
        # 提取特征: 类别ID、标签
        features = [
            f"cat_{service['category_id']}",  # 类别特征
        ]
        
        # 添加标签特征
        if service['tags']:
            tags = service['tags'].split(',')
            for tag in tags:
                features.append(f"tag_{tag.strip()}")
        
        service_features[service_id] = features
    
    # 计算相似度
    similarity_matrix = {}
    all_service_ids = list(services.keys())
    
    for i, service_id1 in enumerate(all_service_ids):
        similarity_matrix[service_id1] = {}
        
        for j, service_id2 in enumerate(all_service_ids):
            if service_id1 == service_id2:
                similarity_matrix[service_id1][service_id2] = 1.0
                continue
            
            # 计算Jaccard相似度
            features1 = set(service_features[service_id1])
            features2 = set(service_features[service_id2])
            
            intersection = len(features1.intersection(features2))
            union = len(features1.union(features2))
            
            if union > 0:
                similarity = intersection / union
            else:
                similarity = 0.0
            
            # 同类别加权
            if services[service_id1]['category_id'] == services[service_id2]['category_id']:
                similarity += 0.2
            
            # 添加描述文本相似度
            text_sim = calculate_text_similarity(
                services[service_id1].get('description', ''),
                services[service_id2].get('description', '')
            )
            similarity += text_sim * 0.3  # 文本相似度权重为0.3
            
            # 归一化
            similarity = min(similarity, 1.0)
            
            similarity_matrix[service_id1][service_id2] = similarity
    
    logger.info(f"构建了 {len(similarity_matrix)} 个服务的相似度矩阵")
    return similarity_matrix

def get_content_based_recommendations(user_id, user_service_matrix, service_similarity_matrix, limit=10):
    """基于内容的协同过滤推荐"""
    recommendations = defaultdict(float)
    
    # 获取用户交互过的服务
    if user_id not in user_service_matrix:
        return []
        
    interacted_services = user_service_matrix[user_id]
    
    # 基于用户交互的服务，找出相似服务
    for service_id, interaction_strength in interacted_services.items():
        if service_id not in service_similarity_matrix:
            continue
            
        similar_services = service_similarity_matrix[service_id]
        
        for similar_id, similarity in similar_services.items():
            if similar_id not in interacted_services:
                recommendations[similar_id] += similarity * interaction_strength
    
    # 排序并返回
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return sorted_recommendations[:limit] 