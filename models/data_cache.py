import mysql.connector
from mysql.connector import Error
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
import pickle
import os
import time

# 导入配置和依赖服务
from recommend.config import DB_CONFIG
# 确保引用路径与你的项目结构一致
from recommend.services.matrix_factorization import build_svd_model
from recommend.services.neural_cf import build_ncf_model, TF_AVAILABLE
from recommend.services.content_based import build_service_similarity_matrix
from recommend.services.ab_testing import build_ab_test_groups

if TF_AVAILABLE:
    import tensorflow as tf

logger = logging.getLogger("recommend-service")


class DataCache:
    def __init__(self):
        self.services = {}  # 服务数据
        self.user_behaviors = defaultdict(list)  # 用户行为数据
        self.user_service_matrix = {}  # 用户-服务交互矩阵
        self.service_similarity_matrix = {}  # 服务相似度矩阵
        self.popular_services = []  # 热门服务
        self.last_update = None  # 最后更新时间
        self.svd_model = None  # SVD模型
        self.ncf_model = None  # 神经协同过滤模型
        self.user_features = {}  # 用户特征
        self.service_features = {}  # 服务特征
        self.service_locations = {}  # 服务地理位置
        self.time_patterns = {}  # 时间模式数据
        self.ab_test_groups = {}  # A/B测试分组

        # 模型保存路径配置
        self.model_dir = "saved_models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # 启动时尝试加载已有模型，避免冷启动等待
        self._load_models()

    def needs_update(self, force=False):
        """检查是否需要更新缓存"""
        # 如果从未更新过且没有加载到历史模型，则必须更新
        if not self.last_update:
            return True
        if force:
            return True
        # 默认每小时更新一次
        now = datetime.now()
        diff = (now - self.last_update).total_seconds()
        return diff > 3600

    def update(self):
        """执行全量数据更新和模型训练"""
        try:
            logger.info("开始更新数据缓存...")
            start_time = time.time()

            # 1. 加载基础数据
            self._load_services()
            self._load_user_behaviors()
            self._build_user_service_matrix()
            self._extract_features()

            # 2. 计算热门服务
            self._calculate_popular_services()

            # 3. 构建相似度矩阵 (基于内容)
            # 只有在服务数据更新后才需要重算
            self.service_similarity_matrix = build_service_similarity_matrix(self.services)

            # 4. 训练 SVD 模型
            users = list(self.user_service_matrix.keys())
            service_ids = list(self.services.keys())
            self.svd_model = build_svd_model(self.user_service_matrix, users, service_ids)

            # 5. 训练 NCF 模型 (如果支持)
            if TF_AVAILABLE:
                # 传入 user_features 和 services 是为了确定Embedding层的维度
                self.ncf_model = build_ncf_model(self.user_service_matrix, self.user_features, self.services)

            # 6. 更新 A/B 测试分组
            self.ab_test_groups = build_ab_test_groups(self.user_features)

            self.last_update = datetime.now()

            # 7. 更新完成后，保存模型到硬盘
            self._save_models()

            elapsed = time.time() - start_time
            logger.info(f"数据缓存更新成功，耗时 {elapsed:.2f} 秒，时间: {self.last_update}")
        except Exception as e:
            logger.error(f"数据缓存更新失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def _save_models(self):
        """将训练好的模型保存到硬盘"""
        try:
            # 保存 SVD 模型 (Pickle)
            if self.svd_model:
                svd_path = os.path.join(self.model_dir, "svd_model.pkl")
                with open(svd_path, 'wb') as f:
                    pickle.dump(self.svd_model, f)
                logger.info(f"SVD模型已保存至 {svd_path}")

            # 保存 NCF 模型 (TensorFlow SavedModel)
            if self.ncf_model and TF_AVAILABLE:
                ncf_path = os.path.join(self.model_dir, "ncf_model")
                self.ncf_model.save(ncf_path)
                logger.info(f"NCF模型已保存至 {ncf_path}")

            # 保存元数据 (最后更新时间)
            meta_path = os.path.join(self.model_dir, "meta.pkl")
            with open(meta_path, 'wb') as f:
                pickle.dump({'last_update': self.last_update}, f)

        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")

    def _load_models(self):
        """从硬盘加载历史模型"""
        try:
            loaded = False
            # 加载 SVD
            svd_path = os.path.join(self.model_dir, "svd_model.pkl")
            if os.path.exists(svd_path):
                with open(svd_path, 'rb') as f:
                    self.svd_model = pickle.load(f)
                loaded = True

            # 加载 NCF
            ncf_path = os.path.join(self.model_dir, "ncf_model")
            if os.path.exists(ncf_path) and TF_AVAILABLE:
                self.ncf_model = tf.keras.models.load_model(ncf_path)
                loaded = True

            # 加载元数据
            meta_path = os.path.join(self.model_dir, "meta.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    self.last_update = meta.get('last_update')

            # 如果成功加载了模型，我们需要重新加载基础数据(Services/Users)来配合模型使用
            # 但不需要重新"训练"(fit)
            if loaded and self.last_update:
                logger.info(f"发现历史模型缓存 (上次更新: {self.last_update})，正在加载基础数据...")
                self._load_services()
                self._load_user_behaviors()
                self._build_user_service_matrix()
                self._extract_features()
                self.service_similarity_matrix = build_service_similarity_matrix(self.services)
                self.ab_test_groups = build_ab_test_groups(self.user_features)
                self._calculate_popular_services()
                logger.info("基础数据加载完成，系统已就绪 (无需重新训练)")
            else:
                logger.info("未发现历史模型缓存，将在首次请求或启动时进行训练")

        except Exception as e:
            logger.error(f"加载历史模型失败: {str(e)}")
            # 如果加载出错，重置状态，等待重新训练
            self.svd_model = None
            self.ncf_model = None
            self.last_update = None

    # --- 以下为基础数据加载逻辑 (保持原逻辑不变) ---

    def _load_services(self):
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, name, category_id, category, description, tags, 
                       rating, review_count, recommended, create_time, price, duration,
                       latitude, longitude, address, open_time, close_time
                FROM service WHERE status = 'active'
            """)
            services = cursor.fetchall()
            self.services = {service['id']: service for service in services}

            # 处理位置和时间
            for service_id, service in self.services.items():
                if service.get('latitude') and service.get('longitude'):
                    self.service_locations[service_id] = {
                        'latitude': float(service['latitude'] or 0),
                        'longitude': float(service['longitude'] or 0),
                        'address': service.get('address', '')
                    }
                # 处理时间对象 (简化处理，假设已是time对象或字符串)
                if service.get('open_time'):
                    self.services[service_id]['open_time_obj'] = service['open_time']
                if service.get('close_time'):
                    self.services[service_id]['close_time_obj'] = service['close_time']

            logger.info(f"加载了 {len(services)} 个服务")
        except Error as e:
            logger.error(f"数据库连接错误 (_load_services): {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()

    def _load_user_behaviors(self):
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT user_id, service_id, behavior_type, duration, created_at FROM user_behavior")
            behaviors = cursor.fetchall()

            cursor.execute("SELECT user_id, service_id, order_time FROM orders WHERE status IN ('completed', 'paid')")
            orders = cursor.fetchall()
            for order in orders:
                order['behavior_type'] = 'order'
                order['duration'] = 0
                order['created_at'] = order['order_time']
                behaviors.append(order)

            self.user_behaviors = defaultdict(list)
            for behavior in behaviors:
                self.user_behaviors[behavior['user_id']].append(behavior)

            logger.info(f"加载了 {len(behaviors)} 条用户行为数据")
        except Error as e:
            logger.error(f"数据库连接错误 (_load_user_behaviors): {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()

    def _build_user_service_matrix(self):
        user_service_matrix = {}
        behavior_weights = {'view': 1.0, 'click': 2.0, 'favorite': 3.0, 'unfavorite': -3.0, 'order': 5.0}
        now = datetime.now()

        for user_id, behaviors in self.user_behaviors.items():
            user_service_matrix[user_id] = {}
            for behavior in behaviors:
                service_id = behavior['service_id']
                behavior_type = behavior['behavior_type']
                duration = behavior.get('duration', 0)

                if service_id not in user_service_matrix[user_id]:
                    user_service_matrix[user_id][service_id] = 0

                weight = behavior_weights.get(behavior_type, 1.0)

                # 时间衰减
                if 'created_at' in behavior and behavior['created_at']:
                    try:
                        created_at = behavior['created_at']
                        if isinstance(created_at, str):  # 防止字符串类型报错
                            # 如果需要解析字符串时间，可在此添加 datetime.strptime
                            days_diff = 0
                        elif isinstance(created_at, datetime):
                            days_diff = (now - created_at).days
                        else:
                            days_diff = 0

                        time_decay = 1.0 / (1.0 + days_diff / 14.0)
                        weight *= time_decay
                    except Exception:
                        pass  # 忽略时间计算错误

                if behavior_type == 'view' and duration > 0:
                    time_factor = min(duration / 60, 5)
                    user_service_matrix[user_id][service_id] += weight * (1 + time_factor)
                elif behavior_type == 'unfavorite':
                    user_service_matrix[user_id][service_id] = max(0, user_service_matrix[user_id][service_id] + weight)
                else:
                    user_service_matrix[user_id][service_id] += weight

        self.user_service_matrix = user_service_matrix

    def _extract_features(self):
        # 提取服务特征
        for service_id, service in self.services.items():
            features = {
                'category_id': service.get('category_id', 0),
                'rating': float(service.get('rating', 0) or 0),
                'review_count': int(service.get('review_count', 0) or 0),
                'is_recommended': 1 if service.get('recommended') else 0
            }
            if service.get('tags'):
                for tag in service.get('tags', '').split(','):
                    features[f'tag_{tag.strip()}'] = 1
            if service_id in self.service_locations:
                features['has_location'] = 1
            else:
                features['has_location'] = 0
            self.service_features[service_id] = features

        self._extract_user_features()
        self._extract_time_patterns()

    def _extract_user_features(self):
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, gender, register_date, last_login_date FROM user WHERE role = 'user'")
            users = cursor.fetchall()

            for user in users:
                user_id = user['id']
                features = {'gender': user.get('gender', 'unknown'), 'age': 0}
                # 简单计算活跃度
                features['activity_level'] = 0.5  # 默认值

                # 类别偏好计算
                if user_id in self.user_service_matrix:
                    category_counts = defaultdict(float)
                    total = 0
                    for sid, score in self.user_service_matrix[user_id].items():
                        if sid in self.services:
                            cid = self.services[sid].get('category_id')
                            if cid:
                                category_counts[cid] += score
                                total += score
                    if total > 0:
                        for cid, count in category_counts.items():
                            features[f'category_pref_{cid}'] = count / total

                self.user_features[user_id] = features
        except Error as e:
            logger.error(f"数据库错误 (_extract_user_features): {e}")
        finally:
            if conn: conn.close()

    def _extract_time_patterns(self):
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT user_id, HOUR(created_at) as hour, COUNT(*) as count, DAYOFWEEK(created_at) as day_of_week
                FROM user_behavior WHERE created_at IS NOT NULL
                GROUP BY user_id, HOUR(created_at), DAYOFWEEK(created_at)
            """)
            patterns = cursor.fetchall()
            for p in patterns:
                uid = p['user_id']
                if uid not in self.time_patterns:
                    self.time_patterns[uid] = {'hourly': [0] * 24, 'daily': [0] * 7}
                self.time_patterns[uid]['hourly'][p['hour']] += p['count']
                self.time_patterns[uid]['daily'][(p['day_of_week'] - 1) % 7] += p['count']
        except Error as e:
            logger.error(f"数据库错误 (_extract_time_patterns): {e}")
        finally:
            if conn: conn.close()

    def _calculate_popular_services(self):
        popular_services = []
        for service_id, service in self.services.items():
            rating = float(service['rating'] or 0)
            review_count = int(service['review_count'] or 0)
            recommended = 1 if service['recommended'] else 0
            score = (rating * 0.6) + (min(review_count, 100) / 100 * 0.3) + (recommended * 0.1)
            popular_services.append({'service_id': service_id, 'score': score})
        popular_services.sort(key=lambda x: x['score'], reverse=True)
        self.popular_services = [item['service_id'] for item in popular_services]

    # --- 代理方法，供外部调用推荐逻辑 ---

    def get_svd_recommendations(self, user_id, limit=10):
        if not self.svd_model or user_id not in self.svd_model['user_idx']:
            return []
        try:
            user_idx = self.svd_model['user_idx'][user_id]
            user_vec = self.svd_model['U'][user_idx] @ np.diag(self.svd_model['sigma'])
            predictions = user_vec @ self.svd_model['Vt']

            interacted = set(self.user_service_matrix.get(user_id, {}).keys())
            candidates = []
            for i, score in enumerate(predictions):
                if i in self.svd_model['reverse_service_idx']:
                    sid = self.svd_model['reverse_service_idx'][i]
                    if sid not in interacted:
                        candidates.append((sid, float(score)))
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:limit]
        except Exception as e:
            logger.error(f"SVD推荐出错: {e}")
            return []

    def get_ncf_recommendations(self, user_id, limit=10):
        if not TF_AVAILABLE or self.ncf_model is None:
            return []
        try:
            from recommend.services.neural_cf import get_ncf_recommendations
            return get_ncf_recommendations(self.ncf_model, user_id, self.user_service_matrix, self.services, limit)
        except Exception as e:
            logger.error(f"NCF推荐出错: {e}")
            return []