import os
import sys

# ================= 路径修复 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# ===========================================

import mysql.connector
import numpy as np
import pickle
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# 导入你的模块
from recommend.config import DB_CONFIG
from recommend.services.matrix_factorization import build_svd_model, get_svd_recommendations
from recommend.services.neural_cf import build_ncf_model, get_ncf_recommendations, TF_AVAILABLE
from recommend.services.content_based import build_service_similarity_matrix, get_content_based_recommendations


def load_data():
    """加载数据并切分为: 训练集(Base), 验证集(Meta), 测试集(Test)"""
    print("正在加载数据库数据...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    # 加载行为数据
    cursor.execute("SELECT user_id, service_id, behavior_type, created_at FROM user_behavior ORDER BY created_at ASC")
    behaviors = cursor.fetchall()

    # 加载服务数据
    cursor.execute("SELECT id, category_id, tags, description FROM service")
    services = {row['id']: row for row in cursor.fetchall()}
    conn.close()

    # 按时间切分: 60% 训练基模型, 20% 训练权重, 20% 最终测试
    total = len(behaviors)
    split_1 = int(total * 0.6)
    split_2 = int(total * 0.8)

    train_base = behaviors[:split_1]  # 用于训练 SVD/NCF
    train_meta = behaviors[split_1:split_2]  # 用于训练逻辑回归权重
    test_final = behaviors[split_2:]  # 用于最终验证

    print(f"数据切分完成: 基模型训练集 {len(train_base)}, 权重训练集 {len(train_meta)}, 测试集 {len(test_final)}")
    return train_base, train_meta, test_final, services


def build_interaction_matrix(behaviors):
    """构建用户-物品交互矩阵"""
    matrix = {}
    for item in behaviors:
        uid, sid = item['user_id'], item['service_id']
        weight = 1
        if item['behavior_type'] == 'click':
            weight = 2
        elif item['behavior_type'] == 'order':
            weight = 5

        if uid not in matrix: matrix[uid] = {}
        if sid not in matrix[uid]: matrix[uid][sid] = 0
        matrix[uid][sid] += weight
    return matrix


def generate_meta_features(train_base, train_meta, services):
    """
    核心步骤：生成用于逻辑回归的特征
    特征 = [CB得分, SVD得分, NCF得分]
    标签 = 1 (因为train_meta里的都是真实发生的行为)
    注意：为了训练二分类器，我们需要负采样（造一些标签为0的假数据）
    """
    print("正在训练基模型 (SVD & NCF)...")

    # 1. 训练基模型
    base_matrix = build_interaction_matrix(train_base)
    users = list(base_matrix.keys())
    service_ids = list(services.keys())

    # 训练 SVD
    svd_model = build_svd_model(base_matrix, users, service_ids)

    # 训练 NCF
    ncf_model = None
    if TF_AVAILABLE:
        # Mock features for training
        mock_user_features = {u: {} for u in users}
        ncf_model = build_ncf_model(base_matrix, mock_user_features, services)

    # 准备 Content-Based 相似度矩阵
    sim_matrix = build_service_similarity_matrix(services)

    print("基模型训练完成，正在生成元数据特征...")

    X = []  # 特征 [cb_score, svd_score, ncf_score]
    y = []  # 标签 1 or 0

    # 辅助函数：获取单点分数
    def get_score(algo, uid, sid):
        if algo == 'svd':
            # 简化版 SVD 预测，直接用内积
            if svd_model and uid in svd_model['user_idx'] and sid in svd_model['service_idx']:
                u_idx = svd_model['user_idx'][uid]
                s_idx = svd_model['service_idx'][sid]
                # U * sigma * Vt (简化计算)
                # 这里为了速度，我们假设 svd_model 结构里很难直接取单个值，
                # 实际上一行代码里我们通常批量预测。
                # 这里为了代码简单，我们用 get_svd_recommendations 的逻辑反推，或者简单给 0
                # *正确做法*：应该在 build_svd_model 里暴露 predict 方法。
                # 这里我们用一个简化的 trick：如果用户在训练集里没见过，给0。
                pass
            return 0  # 占位，下面会用批量预测填充

    # 为了效率，我们直接对 meta 集的用户进行批量预测
    meta_matrix = build_interaction_matrix(train_meta)

    # 遍历 meta 集中每个用户的真实行为 (正样本)
    for user_id, interactions in meta_matrix.items():
        # 获取该用户在验证集里交互过的所有物品
        true_sids = list(interactions.keys())

        # --- 1. 获取 CB 分数 ---
        # CB 推荐通常返回 [(sid, score)...]
        # 我们取 Top 50，如果 sid 在里面就有分，不在就 0
        cb_recs = get_content_based_recommendations(user_id, base_matrix, sim_matrix, limit=100)
        cb_scores = {sid: score for sid, score in cb_recs}

        # --- 2. 获取 SVD 分数 ---
        svd_scores = {}
        if svd_model:
            svd_recs = get_svd_recommendations(svd_model, user_id, base_matrix, limit=100)
            svd_scores = {sid: score for sid, score in svd_recs}

        # --- 3. 获取 NCF 分数 ---
        ncf_scores = {}
        if ncf_model:
            ncf_recs = get_ncf_recommendations(ncf_model, user_id, base_matrix, services, limit=100)
            ncf_scores = {sid: score for sid, score in ncf_recs}

        # --- 构建正样本 (User 真的看了 Service) ---
        for sid in true_sids:
            s1 = cb_scores.get(sid, 0.0)
            s2 = svd_scores.get(sid, 0.0)
            s3 = ncf_scores.get(sid, 0.0)
            X.append([s1, s2, s3])
            y.append(1)

        # --- 构建负样本 (User 没看 Service) ---
        # 随机采样同等数量的负样本
        all_sids = list(services.keys())
        neg_sids = [s for s in all_sids if s not in true_sids]
        if len(neg_sids) > len(true_sids):
            neg_sids = np.random.choice(neg_sids, len(true_sids), replace=False)

        for sid in neg_sids:
            s1 = cb_scores.get(sid, 0.0)
            s2 = svd_scores.get(sid, 0.0)
            s3 = ncf_scores.get(sid, 0.0)
            X.append([s1, s2, s3])
            y.append(0)

    return np.array(X), np.array(y)


def train_logistic_regression(X, y):
    """训练逻辑回归模型，获取权重 (修复版本)"""
    print("\n正在训练逻辑回归模型 (Learning to Rank)...")

    # 1. 归一化特征
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 初始化逻辑回归 (移除了 positive=True 以保证兼容性)
    # fit_intercept=False 是为了让权重直接对应三个模型的贡献度
    clf = LogisticRegression(fit_intercept=False, solver='lbfgs', C=0.1)
    clf.fit(X_scaled, y)

    # 3. 获取系数
    raw_weights = clf.coef_[0]

    # 4. 强制处理负数（如果某个模型系数为负，说明它在当前数据下起反作用，设为0）
    processed_weights = np.maximum(raw_weights, 0)

    # 如果所有权重都成了0（极端情况），回退到平均分配
    if np.sum(processed_weights) == 0:
        print("警告：模型未学习到有效权重，采用平均权重分配。")
        processed_weights = np.array([1 / 3, 1 / 3, 1 / 3])
    else:
        # 归一化使得和为 1
        processed_weights = processed_weights / np.sum(processed_weights)

    print("-" * 30)
    print("权重优化完成！")
    print(f"推荐算法最佳权重组合 (根据真实数据计算得出):")
    print(f"  Content-Based (CB) : {processed_weights[0]:.4f}")
    print(f"  SVD (Matrix Factor): {processed_weights[1]:.4f}")
    print(f"  NCF (Deep Learning): {processed_weights[2]:.4f}")
    print("-" * 30)

    return processed_weights


if __name__ == "__main__":
    # 1. 加载数据
    train_base, train_meta, test_final, services = load_data()

    # 2. 生成特征
    X, y = generate_meta_features(train_base, train_meta, services)

    if len(X) == 0:
        print("错误：生成的特征集为空，请检查是否有足够的训练数据。")
    else:
        # 3. 训练并输出权重
        weights = train_logistic_regression(X, y)