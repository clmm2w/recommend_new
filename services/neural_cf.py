import numpy as np
import random
from recommend.config import logger

# 尝试导入 TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Multiply, Dropout, \
        BatchNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not found. NCF functionality will be disabled.")
    TF_AVAILABLE = False


def _prepare_ncf_data(user_service_matrix, all_service_ids, negative_ratio=4):
    """
    准备 NCF 训练数据 (优化版)
    """
    user_input = []
    item_input = []
    labels = []

    users = list(user_service_matrix.keys())
    # 预先转换为 set 提高查找速度
    all_service_set = set(all_service_ids)

    for u in users:
        # 获取正样本
        interacted_items = set(user_service_matrix[u].keys())

        # 如果该用户交互了所有物品，无法进行负采样，跳过
        if len(interacted_items) >= len(all_service_set):
            continue

        # --- 1. 正样本入列 ---
        for i in interacted_items:
            user_input.append(u)
            item_input.append(i)
            labels.append(1.0)  # float

        # --- 2. 负样本采样 (优化性能) ---
        # 直接计算需要的负样本数量
        num_neg = min(len(interacted_items) * negative_ratio, len(all_service_set) - len(interacted_items))

        if num_neg <= 0: continue

        # 快速采样：不断随机直到选够不重复的负样本
        negatives = []
        while len(negatives) < num_neg:
            t = random.choice(all_service_ids)
            if t not in interacted_items and t not in negatives:
                negatives.append(t)

        user_input.extend([u] * num_neg)
        item_input.extend(negatives)
        labels.extend([0.0] * num_neg)

    return [np.array(user_input), np.array(item_input)], np.array(labels)


def build_ncf_model(user_service_matrix, user_features, services, epochs=10):
    """
    构建 Attention-NCF 模型 (增强版)
    """
    if not TF_AVAILABLE: return None

    try:
        all_service_ids = list(services.keys())
        X_train, y_train = _prepare_ncf_data(user_service_matrix, all_service_ids)

        if len(X_train[0]) == 0:
            logger.warning("NCF 训练数据为空")
            return None

        # 动态计算 Embedding 维度
        # 关键修正：为了防止新ID越界，我们预留一定的 Buffer 空间 (比如 +1000)
        # 这样即使有新用户注册，模型在短期内也不会崩，只会对新ID输出随机Embedding(这也是合理的冷启动表现)
        max_user_id = int(np.max(X_train[0])) + 1000
        max_service_id = int(np.max(X_train[1])) + 200

        # 确保覆盖现有的 services ID
        if services:
            max_service_id = max(max_service_id, max(services.keys()) + 50)

        embedding_dim = 64

        logger.info(f"NCF 训练规模: Users={max_user_id}, Items={max_service_id}, Samples={len(y_train)}")

        # --- 模型架构 ---
        user_input = Input(shape=(1,), name='user_input')
        service_input = Input(shape=(1,), name='service_input')

        # He Normal 初始化 + L2 正则化(可选，这里暂不加以免太复杂)
        user_embedding = Embedding(max_user_id, embedding_dim, name='user_embedding')(user_input)
        service_embedding = Embedding(max_service_id, embedding_dim, name='service_embedding')(service_input)

        u_vec = Flatten()(user_embedding)
        i_vec = Flatten()(service_embedding)

        # Attention 模块
        interaction = Multiply()([u_vec, i_vec])
        attention_probs = Dense(embedding_dim, activation='sigmoid', name='attention_probs')(
            Dense(embedding_dim, activation='relu')(interaction)
        )
        attended_vec = Multiply()([interaction, attention_probs])

        # 拼接 + MLP
        concat = Concatenate()([u_vec, i_vec, attended_vec])

        x = Dense(128, activation='relu')(concat)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(32, activation='relu')(x)

        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[user_input, service_input], outputs=output)

        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']  # 移除 mae，二分类看 acc 足够
        )

        # 加入早停机制：如果验证集 Loss 2轮不下降，就停止
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        model.fit(
            [X_train[0], X_train[1]], y_train,
            batch_size=512,
            epochs=epochs,
            validation_split=0.1,  # 关键：拿出 10% 数据做验证，看真实的准确率！
            callbacks=[early_stop],
            verbose=1,
            shuffle=True
        )

        # 将维度信息保存到模型对象中，方便预测时检查
        model.user_dim_limit = max_user_id
        model.item_dim_limit = max_service_id

        logger.info("Attention-NCF 模型构建完成")
        return model

    except Exception as e:
        logger.error(f"NCF 构建失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def get_ncf_recommendations(model, user_id, user_service_matrix, services, limit=10):
    """
    使用 NCF 模型预测
    """
    if not TF_AVAILABLE or model is None: return []

    try:
        # 1. 边界检查 (防止崩盘)
        # 如果 user_id 超过了模型训练时的最大 Embedding 索引，直接返回空
        # (或者你可以 fallback 到热门推荐，但这由外部逻辑控制)
        if hasattr(model, 'user_dim_limit') and user_id >= model.user_dim_limit:
            # logger.debug(f"用户ID {user_id} 超出模型范围，跳过NCF")
            return []

        interacted = set(user_service_matrix.get(user_id, {}).keys())
        all_sids = list(services.keys())

        # 筛选候选集
        candidates = []
        for sid in all_sids:
            if sid not in interacted:
                # 同样检查物品 ID 边界
                if hasattr(model, 'item_dim_limit') and sid >= model.item_dim_limit:
                    continue
                candidates.append(sid)

        if not candidates: return []

        # 2. 构造 Batch 预测
        u_input = np.array([user_id] * len(candidates))
        i_input = np.array(candidates)

        preds = model.predict([u_input, i_input], batch_size=256, verbose=0)

        # 3. 结果整理
        # preds 是 [[0.1], [0.9], ...]
        results = []
        for idx, score in enumerate(preds):
            results.append((candidates[idx], float(score[0])))  # score 已经是 0-1 之间

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    except Exception as e:
        logger.error(f"NCF 预测出错: {e}")
        return []