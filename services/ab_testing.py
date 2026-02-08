import random
import zlib  # 新增引用
from datetime import datetime
from recommend.config import logger
from recommend.services.recommendation import get_hybrid_recommendations, get_time_aware_recommendations
from recommend.services.diversity import apply_diversity


def build_ab_test_groups(user_features):
    """构建A/B测试分组"""
    try:
        logger.info("开始构建A/B测试分组...")

        # 获取所有用户ID
        user_ids = list(user_features.keys())

        # 创建测试组
        ab_test_groups = {
            'recommendation_algorithm': {},  # 推荐算法测试
            'diversity_level': {},  # 多样性水平测试
        }

        # 为每个用户分配测试组
        for user_id in user_ids:
            # 使用 CRC32 保证哈希值的稳定性
            # 1. 推荐算法测试：将用户分为三组
            # A组：混合推荐，B组：SVD推荐，C组：NCF推荐
            algo_hash = zlib.crc32(f"algo_{user_id}".encode('utf-8'))
            algo_group = algo_hash % 3

            if algo_group == 0:
                ab_test_groups['recommendation_algorithm'][user_id] = 'hybrid'
            elif algo_group == 1:
                ab_test_groups['recommendation_algorithm'][user_id] = 'svd'
            else:
                ab_test_groups['recommendation_algorithm'][user_id] = 'ncf'

            # 2. 多样性水平测试：将用户分为三组
            # A组：低多样性，B组：中多样性，C组：高多样性
            div_hash = zlib.crc32(f"div_{user_id}".encode('utf-8'))
            div_group = div_hash % 3

            if div_group == 0:
                ab_test_groups['diversity_level'][user_id] = 'low'
            elif div_group == 1:
                ab_test_groups['diversity_level'][user_id] = 'medium'
            else:
                ab_test_groups['diversity_level'][user_id] = 'high'

        logger.info(f"A/B测试分组完成，共 {len(user_ids)} 个用户")
        return ab_test_groups
    except Exception as e:
        logger.error(f"构建A/B测试分组失败: {str(e)}")
        # 出错时返回空字典，系统会自动降级到默认推荐
        return {'recommendation_algorithm': {}, 'diversity_level': {}}


def get_ab_test_recommendations(user_id, data_cache, limit=10, diversify=False):
    """根据A/B测试分组获取推荐结果"""
    try:
        # 如果用户不在A/B测试分组中，使用默认推荐
        if user_id not in data_cache.ab_test_groups.get('recommendation_algorithm', {}):
            return get_time_aware_recommendations(user_id, data_cache, datetime.now().time(), limit)

        # 获取用户的测试组
        algo_group = data_cache.ab_test_groups['recommendation_algorithm'].get(user_id, 'hybrid')
        diversity_group = data_cache.ab_test_groups['diversity_level'].get(user_id, 'medium')

        # 根据算法组选择推荐算法
        if algo_group == 'svd':
            base_recommendations = data_cache.get_svd_recommendations(user_id, limit * 2)
            if not base_recommendations:  # 如果SVD推荐失败，回退到混合推荐
                base_recommendations = get_hybrid_recommendations(user_id, data_cache, limit * 2)
        elif algo_group == 'ncf' and data_cache.ncf_model:
            base_recommendations = data_cache.get_ncf_recommendations(user_id, limit * 2)
            if not base_recommendations:  # 如果NCF推荐失败，回退到混合推荐
                base_recommendations = get_hybrid_recommendations(user_id, data_cache, limit * 2)
        else:  # 默认使用混合推荐
            base_recommendations = get_hybrid_recommendations(user_id, data_cache, limit * 2)

        # 应用时间感知（如果是当前时间）
        current_time = datetime.now().time()
        time_recommendations = get_time_aware_recommendations(user_id, data_cache, current_time, limit * 2)

        # 如果时间感知推荐有结果，与基础推荐合并
        if time_recommendations:
            # 合并两种推荐结果，权重各50%
            merged_recommendations = {}

            # 添加基础推荐，权重0.5
            for service_id, score in base_recommendations:
                merged_recommendations[service_id] = score * 0.5

            # 添加时间感知推荐，权重0.5
            for service_id, score in time_recommendations:
                if service_id in merged_recommendations:
                    merged_recommendations[service_id] += score * 0.5
                else:
                    merged_recommendations[service_id] = score * 0.5

            # 转换回列表并排序
            recommendations = [(service_id, score) for service_id, score in merged_recommendations.items()]
            recommendations.sort(key=lambda x: x[1], reverse=True)
        else:
            recommendations = base_recommendations

        # 应用多样性
        if diversify:
            # 根据多样性组确定多样性参数
            diversity_strength = 0.5  # 默认中等多样性
            if diversity_group == 'low':
                diversity_strength = 0.2
            elif diversity_group == 'high':
                diversity_strength = 0.8

            recommendations = apply_diversity(recommendations, data_cache.services, diversity_strength, limit)
        else:
            recommendations = recommendations[:limit]

        return recommendations
    except Exception as e:
        logger.error(f"A/B测试推荐失败: {str(e)}")
        # 失败时回退到混合推荐
        return get_hybrid_recommendations(user_id, data_cache, limit)