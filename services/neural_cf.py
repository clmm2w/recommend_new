import numpy as np
import os
from recommend.config import logger

# 尝试导入 TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Multiply, Dropout, \
        BatchNormalization, Activation, Lambda
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K

    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not found. NCF functionality will be disabled.")
    TF_AVAILABLE = False


def _prepare_ncf_data(user_service_matrix, all_service_ids, negative_ratio=4):
    """
    准备 NCF 训练数据 (含负采样)
    """
    user_input = []
    item_input = []
    labels = []

    # 获取所有有交互的用户
    users = list(user_service_matrix.keys())

    for u in users:
        # 获取该用户交互过的物品集合
        interacted_items = set(user_service_matrix[u].keys())

        # 1. 添加正样本
        for i in interacted_items:
            user_input.append(u)
            item_input.append(i)
            labels.append(1)  # 正样本标签为 1

        # 2. 添加负样本 (Negative Sampling)
        # 随机选择用户没交互过的物品
        num_neg = len(interacted_items) * negative_ratio
        for _ in range(num_neg):
            j = np.random.choice(all_service_ids)
            while j in interacted_items:
                j = np.random.choice(all_service_ids)

            user_input.append(u)
            item_input.append(j)
            labels.append(0)  # 负样本标签为 0

    return [np.array(user_input), np.array(item_input)], np.array(labels)


def build_ncf_model(user_service_matrix, user_features, services):
    """
    构建带有【注意力机制】的神经协同过滤模型 (Attention-NCF)
    """
    if not TF_AVAILABLE:
        logger.warning("TensorFlow未安装，跳过NCF模型构建")
        return None

    try:
        logger.info("开始构建 Attention-Enhanced NCF 模型...")

        # 获取所有服务ID列表
        all_service_ids = list(services.keys())

        # 准备训练数据
        X_train, y_train = _prepare_ncf_data(user_service_matrix, all_service_ids)

        if len(X_train[0]) == 0 or len(y_train) == 0:
            return None

        # 确定 Embedding 维度
        # 取数据中出现的ID和列表中的ID的最大值，防止索引越界
        max_user_id = max(int(np.max(X_train[0])), max(user_features.keys()) if user_features else 0) + 1
        max_service_id = max(int(np.max(X_train[1])), max(services.keys()) if services else 0) + 1

        embedding_dim = 64  # 增加维度以承载更多信息

        logger.info(f"Attention-NCF 参数: Users={max_user_id}, Services={max_service_id}, Dim={embedding_dim}")

        # ================= 模型架构定义 =================

        # 1. 输入层
        user_input = Input(shape=(1,), name='user_input')
        service_input = Input(shape=(1,), name='service_input')

        # 2. Embedding 层 (学习隐向量)
        # 使用 He Normal 初始化，适合 ReLU 网络
        user_embedding = Embedding(
            input_dim=max_user_id,
            output_dim=embedding_dim,
            embeddings_initializer='he_normal',
            name='user_embedding'
        )(user_input)

        service_embedding = Embedding(
            input_dim=max_service_id,
            output_dim=embedding_dim,
            embeddings_initializer='he_normal',
            name='service_embedding'
        )(service_input)

        # 展平
        u_vec = Flatten()(user_embedding)
        i_vec = Flatten()(service_embedding)

        # ================= 核心：注意力机制模块 (Attention Module) =================

        # A. 特征交互 (Element-wise Product)
        # 这一步捕捉用户和物品在同一维度上的共鸣
        interaction = Multiply(name='interaction_layer')([u_vec, i_vec])

        # B. 注意力网络 (Attention Net)
        # 这是一个小型的 MLP，用来计算每个交互维度的重要性分数 (Attention Scores)
        attention_dense = Dense(embedding_dim, activation='relu', name='attention_dense')(interaction)
        attention_probs = Dense(embedding_dim, activation='sigmoid', name='attention_probs')(attention_dense)

        # C. 施加注意力 (Apply Attention)
        # 将原始交互向量与注意力分数相乘，放大重要特征，抑制噪声特征
        attended_interaction = Multiply(name='attended_interaction')([interaction, attention_probs])

        # ================= 深度学习部分 (MLP) =================

        # 将 "原始特征" 和 "经过注意力加权的交互特征" 拼接
        # 这种类似 ResNet 的结构能保留原始信息，防止过拟合
        concat = Concatenate()([u_vec, i_vec, attended_interaction])

        # 多层感知机 (MLP)
        layer1 = Dense(128, activation='relu')(concat)
        layer1 = BatchNormalization()(layer1)  # 批归一化，加速收敛
        layer1 = Dropout(0.3)(layer1)  # Dropout，防止过拟合

        layer2 = Dense(64, activation='relu')(layer1)
        layer2 = BatchNormalization()(layer2)
        layer2 = Dropout(0.2)(layer2)

        layer3 = Dense(32, activation='relu')(layer2)

        # 输出层 (预测点击概率 0-1)
        output = Dense(1, activation='sigmoid', name='prediction')(layer3)

        # ================= 编译与训练 =================

        model = Model(inputs=[user_input, service_input], outputs=output)

        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mae', 'accuracy']
        )

        # 训练模型
        model.fit(
            X_train, y_train,
            batch_size=256,  # 加大Batch size加快训练
            epochs=8,  # 增加 Epochs 因为模型变复杂了
            verbose=0,
            shuffle=True
        )

        logger.info("Attention-NCF 模型训练完成")
        return model

    except Exception as e:
        logger.error(f"构建 Attention-NCF 模型失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def get_ncf_recommendations(model, user_id, user_service_matrix, services, limit=10):
    """
    使用 NCF 模型进行预测
    """
    if not TF_AVAILABLE or model is None:
        return []

    try:
        # 1. 找出用户没交互过的候选服务
        interacted_items = set(user_service_matrix.get(user_id, {}).keys())
        all_services = list(services.keys())

        candidate_services = [s for s in all_services if s not in interacted_items]

        if not candidate_services:
            return []

        # 2. 构造预测输入
        user_input = np.array([user_id] * len(candidate_services))
        item_input = np.array(candidate_services)

        # 3. 批量预测
        predictions = model.predict([user_input, item_input], verbose=0)

        # 4. 整理结果
        results = []
        for i, score in enumerate(predictions):
            results.append((candidate_services[i], float(score[0])))

        # 5. 排序并返回 Top-K
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    except Exception as e:
        logger.error(f"NCF 预测失败: {str(e)}")
        return []