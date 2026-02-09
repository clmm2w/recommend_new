import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from recommend.config import logger


def build_svd_model(user_service_matrix, users, services):
    """构建矩阵分解模型 (增强版)"""
    try:
        logger.info("开始构建矩阵分解模型...")

        if not user_service_matrix or not users or not services:
            logger.warning("构建SVD模型的输入参数不完整")
            return None

        user_idx = {u: i for i, u in enumerate(users)}
        service_idx = {s: i for i, s in enumerate(services)}

        if len(users) == 0 or len(services) == 0:
            return None

        rows, cols, data = [], [], []
        for user_id, interactions in user_service_matrix.items():
            for service_id, score in interactions.items():
                if user_id in user_idx and service_id in service_idx:
                    rows.append(user_idx[user_id])
                    cols.append(service_idx[service_id])
                    # --- 优化点1: 对数平滑 ---
                    # 原始分数可能是 1.0~50.0，差异过大。
                    # log1p(x) = log(1+x)，将 50 压缩到 3.9，将 1 保持在 0.69
                    # 这样 SVD 更容易捕捉模式，而不是被大数值主导
                    data.append(np.log1p(score))

        if len(data) < 10:
            logger.warning("交互数据太少，无法构建 SVD")
            return None

        matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(services)))

        # 动态调整因子数 k，但不超过 50 (过大会过拟合且慢)
        k = min(50, min(matrix.shape) - 1)
        logger.info(f"执行SVD分解，潜在因子数: {k}")

        U, sigma, Vt = svds(matrix, k=k)

        # 预计算 sigma 矩阵，加速推理
        sigma_diag = np.diag(sigma)

        model = {
            'U': U,
            'sigma_diag': sigma_diag,  # 存对角矩阵
            'Vt': Vt,
            'user_idx': user_idx,
            'service_idx': service_idx,
            'reverse_service_idx': {v: k for k, v in service_idx.items()}
        }
        logger.info("矩阵分解模型构建成功")
        return model

    except Exception as e:
        logger.error(f"构建 SVD 失败: {str(e)}")
        return None


def get_svd_recommendations(svd_model, user_id, user_service_matrix, limit=10):
    """使用SVD模型获取推荐 (归一化版)"""
    if not svd_model or user_id not in svd_model['user_idx']:
        return []

    try:
        user_idx = svd_model['user_idx'][user_id]

        # 1. 计算原始预测分数
        # U[u] * Sigma * Vt = 用户的预测向量 (所有物品的得分)
        user_vec = svd_model['U'][user_idx] @ svd_model['sigma_diag']
        predictions = user_vec @ svd_model['Vt']

        # 2. 筛选候选集
        interacted = set(user_service_matrix.get(user_id, {}).keys())
        candidates = []
        scores = []

        for i, score in enumerate(predictions):
            if i in svd_model['reverse_service_idx']:
                sid = svd_model['reverse_service_idx'][i]
                if sid not in interacted:
                    candidates.append(sid)
                    scores.append(float(score))

        if not candidates:
            return []

        # --- 优化点2: 输出归一化 (Min-Max Scaling) ---
        # 这一步至关重要！确保 SVD 的输出在 0~1 之间，能和 NCF/CB 公平竞争
        scores = np.array(scores)
        min_s = scores.min()
        max_s = scores.max()

        if max_s - min_s > 0:
            norm_scores = (scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.zeros_like(scores)  # 如果全一样，就给0

        # 组合回 (sid, score)
        final_results = list(zip(candidates, norm_scores))

        # 排序
        final_results.sort(key=lambda x: x[1], reverse=True)

        return final_results[:limit]

    except Exception as e:
        logger.error(f"SVD 推荐失败: {str(e)}")
        return []