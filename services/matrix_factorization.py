import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from recommend.config import logger

def build_svd_model(user_service_matrix, users, services):
    """构建矩阵分解模型"""
    try:
        logger.info("开始构建矩阵分解模型...")
        
        # 检查输入参数
        if not user_service_matrix or not users or not services:
            logger.warning("构建SVD模型的输入参数不完整")
            return None
            
        # 构建用户-服务矩阵
        user_idx = {u: i for i, u in enumerate(users)}
        service_idx = {s: i for i, s in enumerate(services)}
        
        if len(users) == 0 or len(services) == 0:
            logger.warning("用户或服务数据为空，无法构建矩阵分解模型")
            return None
            
        # 创建稀疏矩阵
        rows, cols, data = [], [], []
        for user_id, interactions in user_service_matrix.items():
            for service_id, score in interactions.items():
                if user_id in user_idx and service_id in service_idx:
                    rows.append(user_idx[user_id])
                    cols.append(service_idx[service_id])
                    data.append(score)
        
        # 检查是否有足够的数据
        if len(data) < 10:
            logger.warning("交互数据太少，无法构建有效的矩阵分解模型")
            return None
            
        matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(services)))
        
        # SVD分解
        k = min(30, min(matrix.shape) - 1)  # 潜在因子数
        logger.info(f"执行SVD分解，潜在因子数: {k}")
        
        try:
            U, sigma, Vt = svds(matrix, k=k)
            
            # 保存模型
            model = {
                'U': U,
                'sigma': sigma,
                'Vt': Vt,
                'user_idx': user_idx,
                'service_idx': service_idx,
                'reverse_user_idx': {v: k for k, v in user_idx.items()},
                'reverse_service_idx': {v: k for k, v in service_idx.items()}
            }
            logger.info("矩阵分解模型构建成功")
            return model
        except Exception as e:
            logger.error(f"SVD分解失败: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"构建矩阵分解模型失败: {str(e)}")
        return None

def get_svd_recommendations(svd_model, user_id, user_service_matrix, limit=10):
    """使用SVD模型获取推荐"""
    if not svd_model or user_id not in svd_model['user_idx']:
        logger.info(f"用户 {user_id} 没有SVD模型数据，无法使用SVD推荐")
        return []
    
    try:
        # 获取用户向量
        user_idx = svd_model['user_idx'][user_id]
        user_vec = svd_model['U'][user_idx] @ np.diag(svd_model['sigma'])
        
        # 计算所有服务的预测评分
        predictions = user_vec @ svd_model['Vt']
        
        # 排除已交互的服务
        interacted = set(user_service_matrix.get(user_id, {}).keys())
        candidates = []
        
        for i, score in enumerate(predictions):
            if i in svd_model['reverse_service_idx']:
                service_id = svd_model['reverse_service_idx'][i]
                if service_id not in interacted:
                    candidates.append((service_id, float(score)))
        
        # 排序并返回
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:limit]
    except Exception as e:
        logger.error(f"获取SVD推荐失败: {str(e)}")
        return [] 