import os
import sys

# ================= 1. 路径修复逻辑 (必须在所有 recommend 导入之前) =================
# 获取当前文件 (evaluate_model.py) 的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录 (D:\Study\WebFinalWork\高级\recommend)
current_dir = os.path.dirname(current_file_path)
# 获取项目的根目录 (D:\Study\WebFinalWork\高级)
# 只有把这个目录加入 sys.path，Python 才能识别到 'recommend' 文件夹是一个包
root_dir = os.path.dirname(current_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# ================= 2. 标准库导入 =================
import mysql.connector
import numpy as np
from collections import defaultdict

# ================= 3. 项目模块导入 (现在不会报错了) =================
from recommend.config import DB_CONFIG, logger
from recommend.services.matrix_factorization import build_svd_model, get_svd_recommendations
from recommend.services.neural_cf import build_ncf_model, get_ncf_recommendations, TF_AVAILABLE
from recommend.services.content_based import build_service_similarity_matrix, get_content_based_recommendations

def load_all_data():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT user_id, service_id, behavior_type, created_at 
        FROM user_behavior 
        ORDER BY created_at ASC
    """)
    behaviors = cursor.fetchall()

    # 加载完整的服务数据，用于 Content-Based 计算
    cursor.execute("SELECT id, category_id, tags, description FROM service")
    services = {row['id']: row for row in cursor.fetchall()}

    conn.close()
    return behaviors, services


def split_train_test(behaviors):
    user_behaviors = defaultdict(list)
    for b in behaviors:
        user_behaviors[b['user_id']].append(b)

    train_matrix = {}
    test_set = defaultdict(set)

    for user_id, user_data in user_behaviors.items():
        if len(user_data) < 5:
            train_items = user_data
            test_items = []
        else:
            split_idx = int(len(user_data) * 0.8)
            train_items = user_data[:split_idx]
            test_items = user_data[split_idx:]

        train_matrix[user_id] = {}
        for item in train_items:
            sid = item['service_id']
            weight = 1
            if item['behavior_type'] == 'click':
                weight = 2
            elif item['behavior_type'] == 'order':
                weight = 5

            if sid not in train_matrix[user_id]:
                train_matrix[user_id][sid] = 0
            train_matrix[user_id][sid] += weight

        for item in test_items:
            test_set[user_id].add(item['service_id'])

    return train_matrix, test_set


def evaluate_metrics(recommendations, test_set):
    precision_sum = 0
    recall_sum = 0
    users_count = 0

    for user_id, true_items in test_set.items():
        if user_id not in recommendations or not true_items:
            continue
        pred_items = set([x[0] for x in recommendations[user_id]])
        hits = len(true_items.intersection(pred_items))
        if len(pred_items) > 0:
            precision_sum += hits / len(pred_items)
        if len(true_items) > 0:
            recall_sum += hits / len(true_items)
        users_count += 1

    if users_count == 0:
        return 0.0, 0.0
    return precision_sum / users_count, recall_sum / users_count


def run_evaluation():
    print("正在加载数据...")
    behaviors, services_data = load_all_data()
    print(f"加载了 {len(behaviors)} 条行为数据")

    print("切分训练集与测试集...")
    train_matrix, test_set = split_train_test(behaviors)
    users = list(train_matrix.keys())
    service_ids = list(services_data.keys())

    # 提前计算相似度矩阵 (用于混合算法和CB算法)
    print("正在计算服务相似度矩阵...")
    sim_matrix = build_service_similarity_matrix(services_data)

    results = {}

    # --- 1. 评估 SVD ---
    print("\n[1/3] 正在训练与评估 SVD 模型...")
    svd_model = build_svd_model(train_matrix, users, service_ids)
    svd_recs = {}
    if svd_model:
        for user_id in test_set.keys():
            svd_recs[user_id] = get_svd_recommendations(svd_model, user_id, train_matrix, limit=10)
        p, r = evaluate_metrics(svd_recs, test_set)
        results['SVD'] = {'precision': p, 'recall': r}

    # --- 2. 评估 NCF ---
    ncf_recs = {}
    if TF_AVAILABLE:
        print("\n[2/3] 正在训练与评估 NCF 模型...")
        mock_user_features = {u: {} for u in users}
        ncf_model = build_ncf_model(train_matrix, mock_user_features, services_data)
        if ncf_model:
            eval_users = list(test_set.keys())[:100]
            for i, user_id in enumerate(eval_users):
                ncf_recs[user_id] = get_ncf_recommendations(ncf_model, user_id, train_matrix, services_data, limit=10)
            p, r = evaluate_metrics(ncf_recs, test_set)
            results['NCF'] = {'precision': p, 'recall': r}

        # --- 3. 评估 Hybrid (同步最新的 LTR 动态权重策略) ---
        print("\n[3/3] 正在评估 Hybrid 混合推荐算法...")
        hybrid_recs = {}

        # 填入最新的训练结果
        BEST_W_SVD = 0.33
        BEST_W_NCF = 0.67

        for user_id in test_set.keys():
            scores = defaultdict(float)

            # 获取基础推荐列表
            cb_list = get_content_based_recommendations(user_id, train_matrix, sim_matrix, limit=20)
            s_list = svd_recs.get(user_id, [])
            n_list = ncf_recs.get(user_id, [])

            # 获取该用户在训练集中的交互次数，模拟真实系统的动态判断
            num_interactions = len(train_matrix.get(user_id, {}))

            # --- 这里的权重逻辑必须和 recommendation.py 保持完全一致 ---
            if num_interactions < 5:
                # 冷启动策略
                w_cb, w_svd, w_ncf = 0.8, 0.1, 0.1
            else:
                # 活跃期策略 (使用 Learning to Rank 算出的权重)
                # 因为之前算出的 CB 是 0.0，所以这里直接用 0
                w_cb, w_svd, w_ncf = 0.0, BEST_W_SVD, BEST_W_NCF

            # 融合分值
            for sid, score in cb_list: scores[sid] += score * w_cb
            for sid, score in s_list:  scores[sid] += (score / 5.0) * w_svd  # SVD 归一化
            for sid, score in n_list:  scores[sid] += score * w_ncf

            # 排序并取 Top 10
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            hybrid_recs[user_id] = sorted_items[:10]

    p, r = evaluate_metrics(hybrid_recs, test_set)
    results['Hybrid'] = {'precision': p, 'recall': r}

    print("\n" + "=" * 30)
    print("       最终评估报告")
    print("=" * 30)
    print(f"{'算法':<10} | {'准确率':<10} | {'召回率':<10}")
    print("-" * 35)
    for name in ['SVD', 'NCF', 'Hybrid']:
        if name in results:
            m = results[name]
            print(f"{name:<10} | {m['precision']:<10.2%} | {m['recall']:<10.2%}")
    print("=" * 30)


if __name__ == "__main__":
    run_evaluation()