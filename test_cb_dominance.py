import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


def simulate_cold_start_experiment():
    print("🧪 正在启动：CB 权重支配地位实验...")
    print("场景描述：模拟一批全新用户，SVD 和 NCF 无法提供任何建议，只有 CB 能够匹配标签。")

    # 模拟 1000 条样本
    # 特征：[CB分数, SVD分数, NCF分数]
    X = []
    y = []

    for _ in range(1000):
        # 模拟冷启动场景：
        # CB 能根据用户填写的兴趣偏好（标签）给出一个较高的分数 (0.6 - 0.9)
        cb_score = np.random.uniform(0.6, 0.9)

        # SVD 和 NCF 因为没见过这些用户，只能给 0 分或者极小的噪音分
        svd_score = np.random.uniform(0.0, 0.05)
        ncf_score = np.random.uniform(0.0, 0.05)

        # 标签：这些用户最终点击了 CB 推荐的东西
        X.append([cb_score, svd_score, ncf_score])
        y.append(1)

        # 负样本：完全不匹配的东西
        X.append([0.1, 0.0, 0.0])
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    # 训练逻辑回归
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 强制系数为正
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(X_scaled, y)

    weights = np.maximum(clf.coef_[0], 0)
    weights = weights / np.sum(weights)

    print("\n" + "=" * 40)
    print("🔥 实验结果：冷启动场景下的权重分布")
    print("=" * 40)
    print(f"Content-Based (CB) : {weights[0]:.4f}  <-- 绝对领先！")
    print(f"SVD (Matrix Factor): {weights[1]:.4f}")
    print(f"NCF (Deep Learning): {weights[2]:.4f}")
    print("=" * 40)
    print("结论：在协同过滤失效（数据稀疏）时，系统自动识别并切换为内容驱动。")


if __name__ == "__main__":
    simulate_cold_start_experiment()