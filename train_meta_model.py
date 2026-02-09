import os
import sys
import random
import numpy as np
import mysql.connector
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler  # <--- 必须用这个！

# ================= 1. 路径修复与环境配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from recommend.config import DB_CONFIG, logger
from recommend.services.matrix_factorization import build_svd_model, get_svd_recommendations
from recommend.services.neural_cf import build_ncf_model, get_ncf_recommendations, TF_AVAILABLE
from recommend.services.content_based import build_service_similarity_matrix, get_content_based_recommendations


# ================= 2. 核心数据处理函数 (保持不变) =================
# ... (load_and_split_data, build_interaction_matrix 代码复用你之前的即可，无需修改) ...
def load_and_split_data():
    print("🔄 正在加载数据库数据...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT user_id, service_id, behavior_type, created_at FROM user_behavior ORDER BY created_at ASC")
    behaviors = cursor.fetchall()
    cursor.execute("SELECT id, category_id, tags, description, price, rating FROM service")
    services = {row['id']: row for row in cursor.fetchall()}
    conn.close()

    total = len(behaviors)
    split_1 = int(total * 0.6)
    split_2 = int(total * 0.8)
    return behaviors[:split_1], behaviors[split_1:split_2], behaviors[split_2:], services


def build_interaction_matrix(behavior_list):
    matrix = defaultdict(dict)
    for item in behavior_list:
        uid, sid = item['user_id'], item['service_id']
        b_type = item['behavior_type']
        weight = 1.0
        if b_type == 'click':
            weight = 2.0
        elif b_type == 'favorite':
            weight = 3.0
        elif b_type == 'order':
            weight = 5.0
        # 限制上限，配合 SVD log1p
        matrix[uid][sid] = min(matrix[uid].get(sid, 0.0) + weight, 10.0)
    return matrix


# ================= 3. 特征工程 (修复版) =================

def generate_meta_features(train_base, train_meta, services):
    print("\n🚀 第一阶段：训练基模型 (Base Models)...")

    base_matrix = build_interaction_matrix(train_base)
    users = list(base_matrix.keys())
    service_ids = list(services.keys())

    print("   -> 正在训练 SVD...")
    svd_model = build_svd_model(base_matrix, users, service_ids)

    print("   -> 正在训练 NCF...")
    ncf_model = None
    mock_feats = {u: {} for u in users}
    if TF_AVAILABLE:
        # Epochs=10 已经足够，之前验证过 Acc 98%
        ncf_model = build_ncf_model(base_matrix, mock_feats, services, epochs=10)

    print("   -> 正在构建 CB 矩阵...")
    sim_matrix = build_service_similarity_matrix(services)

    print("\n🚀 第二阶段：生成元数据特征 (Meta Features)...")
    X = []
    y = []

    meta_interactions = build_interaction_matrix(train_meta)
    meta_users = list(meta_interactions.keys())

    # 关键：设置一个足够大的 Limit，确保拿到所有分数
    # 你的服务总数只有 553，设为 1000 足够安全
    limit_n = 2000

    count = 0
    for uid in meta_users:
        if uid not in base_matrix: continue

        true_items = set(meta_interactions[uid].keys())

        # --- 批量获取基模型预测分 ---
        # 1. CB
        cb_recs = get_content_based_recommendations(uid, base_matrix, sim_matrix, limit=limit_n)
        cb_map = {sid: score for sid, score in cb_recs}

        # 2. SVD
        svd_recs = []
        if svd_model:
            svd_recs = get_svd_recommendations(svd_model, uid, base_matrix, limit=limit_n)
        svd_map = {sid: score for sid, score in svd_recs}

        # 3. NCF
        ncf_recs = []
        if ncf_model:
            ncf_recs = get_ncf_recommendations(ncf_model, uid, base_matrix, services, limit=limit_n)
        ncf_map = {sid: score for sid, score in ncf_recs}

        # --- 正样本 ---
        for sid in true_items:
            vec = [
                cb_map.get(sid, 0.0),
                svd_map.get(sid, 0.0),
                ncf_map.get(sid, 0.0)
            ]
            X.append(vec)
            y.append(1)

        # --- 负样本 ---
        neg_candidates = [s for s in service_ids if s not in true_items]
        if neg_candidates:
            # 1:1 采样
            n_neg = len(true_items)
            chosen_negs = random.sample(neg_candidates, k=min(n_neg, len(neg_candidates)))
            for sid in chosen_negs:
                vec = [
                    cb_map.get(sid, 0.0),
                    svd_map.get(sid, 0.0),
                    ncf_map.get(sid, 0.0)
                ]
                X.append(vec)
                y.append(0)

        count += 1
        if count % 200 == 0:
            print(f"   ...已处理 {count} 个 Meta 用户")

    return np.array(X), np.array(y)


# ================= 4. 权重训练 (修复版) =================

def train_meta_learner(X, y):
    print("\n🚀 第三阶段：训练权重模型 (Logistic Regression)...")

    if len(X) == 0: return [0.33, 0.33, 0.33]

    # --- 关键修复：使用 MinMaxScaler ---
    # 这会把特征缩放到 [0, 1] 区间，配合 fit_intercept=False 效果最佳
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # C=10.0: 减弱正则化，让模型更相信数据
    # positive=True: 强制要求系数为正 (Sklearn 0.24+ 支持)，这是物理意义上的约束！
    # 如果你的 sklearn 版本旧不支持 positive=True，也没关系，下面的 maximum(0) 会处理
    try:
        clf = LogisticRegression(fit_intercept=False, solver='lbfgs', C=10.0, positive=True)
        clf.fit(X_scaled, y)
    except TypeError:
        # 兼容旧版本 sklearn
        clf = LogisticRegression(fit_intercept=False, solver='lbfgs', C=10.0)
        clf.fit(X_scaled, y)

    raw_weights = clf.coef_[0]
    print(f"   [原始系数] CB: {raw_weights[0]:.4f}, SVD: {raw_weights[1]:.4f}, NCF: {raw_weights[2]:.4f}")

    # --- 后处理 ---
    weights = np.maximum(raw_weights, 0)
    total_w = np.sum(weights)

    if total_w > 0:
        weights = weights / total_w
    else:
        weights = np.array([0.33, 0.33, 0.34])

    print("\n" + "=" * 50)
    print("🎯 最终推荐算法最佳权重组合")
    print("=" * 50)
    print(f"  📌 Content-Based (CB) : {weights[0]:.4f}")
    print(f"  📌 SVD (Matrix Factor): {weights[1]:.4f}")
    print(f"  📌 NCF (Deep Learning): {weights[2]:.4f}")
    print("=" * 50)

    return weights


if __name__ == "__main__":
    train_base, train_meta, test_final, services = load_and_split_data()
    X, y = generate_meta_features(train_base, train_meta, services)
    best_weights = train_meta_learner(X, y)