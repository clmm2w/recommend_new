from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import uvicorn
import json
import os
import logging
from datetime import datetime, time
import mysql.connector
from mysql.connector import Error
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import random
import math

# 尝试导入geopy，如果不可用则提供替代方案
try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    # 简单的替代函数，使用欧几里得距离
    def calculate_distance(coord1, coord2):
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        # 简化的距离计算（公里）
        return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111.32

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow未安装，神经网络推荐模型将不可用")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("recommend.log")
    ]
)
logger = logging.getLogger("recommend-service")

# 创建FastAPI应用
app = FastAPI(title="ezServe推荐系统", description="为ezServe平台提供个性化推荐服务")

# 跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库配置
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "123456"),
    "database": os.environ.get("DB_NAME", "ezserve_db"),
    "port": int(os.environ.get("DB_PORT", "3306"))
}

# 数据缓存
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
    
    def needs_update(self, force=False):
        """检查是否需要更新缓存"""
        if force or not self.last_update:
            return True
        # 每小时更新一次
        now = datetime.now()
        diff = (now - self.last_update).total_seconds()
        return diff > 3600
    
    def update(self):
        """更新缓存数据"""
        try:
            self._load_services()
            self._load_user_behaviors()
            self._build_user_service_matrix()
            self._extract_features()  # 提取特征
            self._build_service_similarity_matrix()
            self._calculate_popular_services()
            self._build_matrix_factorization_model()  # 添加矩阵分解模型构建
            
            # 如果TensorFlow可用，构建NCF模型
            if TF_AVAILABLE:
                self._build_ncf_model()
            
            # 构建A/B测试分组
            self._build_ab_test_groups()
                
            self.last_update = datetime.now()
            logger.info(f"数据缓存更新成功，时间: {self.last_update}")
        except Exception as e:
            logger.error(f"数据缓存更新失败: {str(e)}")
    
    def _load_services(self):
        """加载服务数据"""
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, name, category_id, category, description, tags, 
                       rating, review_count, recommended, create_time, price, duration
                FROM service WHERE status = 'active'
            """)
            services = cursor.fetchall()
            self.services = {service['id']: service for service in services}
            
            # 提取服务位置信息
            for service_id, service in self.services.items():
                if service.get('latitude') and service.get('longitude'):
                    self.service_locations[service_id] = {
                        'latitude': float(service['latitude'] or 0),
                        'longitude': float(service['longitude'] or 0),
                        'address': service.get('address', '')
                    }
                
                # 提取营业时间信息
                if service.get('open_time') and service.get('close_time'):
                    try:
                        open_time = service['open_time']
                        close_time = service['close_time']
                        self.services[service_id]['open_time_obj'] = open_time
                        self.services[service_id]['close_time_obj'] = close_time
                    except Exception as e:
                        logger.error(f"解析服务 {service_id} 的营业时间失败: {e}")
            
            logger.info(f"加载了 {len(services)} 个服务")
        except Error as e:
            logger.error(f"数据库连接错误: {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()
    
    def _load_user_behaviors(self):
        """加载用户行为数据"""
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            # 获取用户行为数据，添加创建时间字段
            cursor.execute("SELECT user_id, service_id, behavior_type, duration, created_at FROM user_behavior")
            behaviors = cursor.fetchall()
            
            # 获取订单数据（作为强烈的正向信号）
            # 使用order_time替代created_at
            cursor.execute("SELECT user_id, service_id, order_time FROM orders WHERE status IN ('completed', 'paid')")
            orders = cursor.fetchall()
            for order in orders:
                order['behavior_type'] = 'order'
                order['duration'] = 0
                order['created_at'] = order['order_time']  # 映射字段
                behaviors.append(order)
            
            # 按用户ID映射分组
            # 这种分组方式的目的是便于后续处理每个用户的行为数据
            self.user_behaviors = defaultdict(list)
            for behavior in behaviors:
                self.user_behaviors[behavior['user_id']].append(behavior)
            
            logger.info(f"加载了 {len(behaviors)} 条用户行为数据，涉及 {len(self.user_behaviors)} 个用户")
        except Error as e:
            logger.error(f"数据库连接错误: {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()
    
    def _build_user_service_matrix(self):
        """构建用户-服务交互矩阵"""
        user_service_matrix = {}
        
        # 行为权重
        behavior_weights = {
            'view': 1.0,
            'click': 2.0,
            'favorite': 3.0,
            'unfavorite': -3.0,  # 取消收藏，抵消收藏行为
            'order': 5.0
        }
        
        # 获取当前时间，用于计算时间衰减
        now = datetime.now()
        
        # 构建矩阵
        for user_id, behaviors in self.user_behaviors.items():
            user_service_matrix[user_id] = {}
            
            for behavior in behaviors:
                service_id = behavior['service_id']
                behavior_type = behavior['behavior_type']
                duration = behavior.get('duration', 0)
                
                # 初始化
                if service_id not in user_service_matrix[user_id]:
                    user_service_matrix[user_id][service_id] = 0
                
                # 计算基础权重
                weight = behavior_weights.get(behavior_type, 1.0)
                
                # 应用时间衰减因子
                if 'created_at' in behavior and behavior['created_at']:
                    # 计算天数差
                    days_diff = (now - behavior['created_at']).days
                    # 时间衰减因子，14天衰减一半
                    time_decay = 1.0 / (1.0 + days_diff / 14.0)
                    weight *= time_decay
                
                # 累加交互强度
                if behavior_type == 'view' and duration > 0:
                    # 浏览时间越长，权重越高，但有上限
                    time_factor = min(duration / 60, 5)  # 最多5分钟
                    user_service_matrix[user_id][service_id] += weight * (1 + time_factor)
                elif behavior_type == 'unfavorite':
                    # 对于取消收藏，确保不会将分数降到负值
                    user_service_matrix[user_id][service_id] = max(0, user_service_matrix[user_id][service_id] + weight)
                    logger.info(f"用户{user_id}取消收藏服务{service_id}，更新交互分数")
                else:
                    user_service_matrix[user_id][service_id] += weight
        
        self.user_service_matrix = user_service_matrix
        logger.info(f"构建了 {len(user_service_matrix)} 个用户的服务交互矩阵")
    
    def _calculate_text_similarity(self, text1, text2):
        """计算两段文本的相似度"""
        if not text1 or not text2:
            return 0.0
            
        # 简单的词袋模型
        def get_words(text):
            # 将文本转换为小写并分词
            if isinstance(text, str):
                return set(text.lower().split())
            return set()
            
        words1 = get_words(text1)
        words2 = get_words(text2)
        
        # 计算Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union > 0:
            return intersection / union
        return 0.0
    
    def _build_service_similarity_matrix(self):
        """构建服务相似度矩阵"""
        # 基于内容的相似度计算
        service_features = {}
        for service_id, service in self.services.items():
            # 提取特征: 类别ID、标签
            features = [
                f"cat_{service['category_id']}",  # 类别特征
            ]
            
            # 添加标签特征
            if service['tags']:
                tags = service['tags'].split(',')
                for tag in tags:
                    features.append(f"tag_{tag.strip()}")
            
            service_features[service_id] = features
        
        # 计算相似度
        similarity_matrix = {}
        all_service_ids = list(self.services.keys())
        
        for i, service_id1 in enumerate(all_service_ids):
            similarity_matrix[service_id1] = {}
            
            for j, service_id2 in enumerate(all_service_ids):
                if service_id1 == service_id2:
                    similarity_matrix[service_id1][service_id2] = 1.0
                    continue
                
                # 计算Jaccard相似度
                features1 = set(service_features[service_id1])
                features2 = set(service_features[service_id2])
                
                intersection = len(features1.intersection(features2))
                union = len(features1.union(features2))
                
                if union > 0:
                    similarity = intersection / union
                else:
                    similarity = 0.0
                
                # 同类别加权
                if self.services[service_id1]['category_id'] == self.services[service_id2]['category_id']:
                    similarity += 0.2
                
                # 添加描述文本相似度
                text_sim = self._calculate_text_similarity(
                    self.services[service_id1].get('description', ''),
                    self.services[service_id2].get('description', '')
                )
                similarity += text_sim * 0.3  # 文本相似度权重为0.3
                
                # 归一化
                similarity = min(similarity, 1.0)
                
                similarity_matrix[service_id1][service_id2] = similarity
        
        self.service_similarity_matrix = similarity_matrix
        logger.info(f"构建了 {len(similarity_matrix)} 个服务的相似度矩阵")
    
    def _calculate_popular_services(self):
        """计算热门服务"""
        # 基于评分和评论数量计算热门度
        popular_services = []
        
        for service_id, service in self.services.items():
            # 将decimal类型转换为float
            rating = float(service['rating'] or 0)
            review_count = int(service['review_count'] or 0)
            recommended = 1 if service['recommended'] else 0
            
            # 计算热门度分数
            popularity_score = (rating * 0.6) + (min(review_count, 100) / 100 * 0.3) + (recommended * 0.1)
            
            popular_services.append({
                'service_id': service_id,
                'score': popularity_score
            })
        
        # 按热门度排序
        popular_services.sort(key=lambda x: x['score'], reverse=True)
        self.popular_services = [item['service_id'] for item in popular_services]
        
        logger.info(f"计算了 {len(self.popular_services)} 个热门服务")

    def _build_matrix_factorization_model(self):
        """构建矩阵分解模型"""
        try:
            logger.info("开始构建矩阵分解模型...")
            
            # 构建用户-服务矩阵
            users = list(self.user_service_matrix.keys())
            services = list(self.services.keys())
            
            if len(users) == 0 or len(services) == 0:
                logger.warning("用户或服务数据为空，无法构建矩阵分解模型")
                return
                
            user_idx = {u: i for i, u in enumerate(users)}
            service_idx = {s: i for i, s in enumerate(services)}
            
            # 创建稀疏矩阵
            rows, cols, data = [], [], []
            for user_id, interactions in self.user_service_matrix.items():
                for service_id, score in interactions.items():
                    if user_id in user_idx and service_id in service_idx:
                        rows.append(user_idx[user_id])
                        cols.append(service_idx[service_id])
                        data.append(score)
            
            # 检查是否有足够的数据
            if len(data) < 10:
                logger.warning("交互数据太少，无法构建有效的矩阵分解模型")
                return
                
            matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(services)))
            
            # SVD分解
            k = min(30, min(matrix.shape) - 1)  # 潜在因子数
            logger.info(f"执行SVD分解，潜在因子数: {k}")
            
            try:
                U, sigma, Vt = svds(matrix, k=k)
                
                # 保存模型
                self.svd_model = {
                    'U': U,
                    'sigma': sigma,
                    'Vt': Vt,
                    'user_idx': user_idx,
                    'service_idx': service_idx,
                    'reverse_user_idx': {v: k for k, v in user_idx.items()},
                    'reverse_service_idx': {v: k for k, v in service_idx.items()}
                }
                logger.info("矩阵分解模型构建成功")
            except Exception as e:
                logger.error(f"SVD分解失败: {str(e)}")
        except Exception as e:
            logger.error(f"构建矩阵分解模型失败: {str(e)}")
    
    def get_svd_recommendations(self, user_id, limit=10):
        """使用SVD模型获取推荐"""
        if not self.svd_model or user_id not in self.svd_model['user_idx']:
            logger.info(f"用户 {user_id} 没有SVD模型数据，无法使用SVD推荐")
            return []
        
        try:
            # 获取用户向量
            user_idx = self.svd_model['user_idx'][user_id]
            user_vec = self.svd_model['U'][user_idx] @ np.diag(self.svd_model['sigma'])
            
            # 计算所有服务的预测评分
            predictions = user_vec @ self.svd_model['Vt']
            
            # 排除已交互的服务
            interacted = set(self.user_service_matrix.get(user_id, {}).keys())
            candidates = []
            
            for i, score in enumerate(predictions):
                if i in self.svd_model['reverse_service_idx']:
                    service_id = self.svd_model['reverse_service_idx'][i]
                    if service_id not in interacted:
                        candidates.append((service_id, float(score)))
            
            # 排序并返回
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:limit]
        except Exception as e:
            logger.error(f"获取SVD推荐失败: {str(e)}")
            return []

    def _extract_features(self):
        """提取用户和服务特征，用于高级推荐模型"""
        try:
            logger.info("开始提取用户和服务特征...")
            
            # 提取服务特征
            for service_id, service in self.services.items():
                features = {
                    'category_id': service.get('category_id', 0),
                    'rating': float(service.get('rating', 0) or 0),
                    'review_count': int(service.get('review_count', 0) or 0),
                    'is_recommended': 1 if service.get('recommended') else 0
                }
                
                # 处理标签
                if service.get('tags'):
                    tags = service.get('tags', '').split(',')
                    for tag in tags:
                        tag = tag.strip()
                        if tag:
                            features[f'tag_{tag}'] = 1
                
                # 添加位置特征
                if service_id in self.service_locations:
                    features['has_location'] = 1
                else:
                    features['has_location'] = 0
                
                # 添加营业时间特征
                if service.get('open_time_obj') and service.get('close_time_obj'):
                    # 计算营业时间长度（小时）
                    open_time = service['open_time_obj']
                    close_time = service['close_time_obj']
                    
                    # 如果跨天营业
                    if close_time < open_time:
                        hours_open = (24 - open_time.hour) + close_time.hour
                    else:
                        hours_open = close_time.hour - open_time.hour
                    
                    features['hours_open'] = hours_open
                    features['opens_early'] = 1 if open_time.hour < 9 else 0
                    features['opens_late'] = 1 if close_time.hour >= 20 else 0
                
                self.service_features[service_id] = features
            
            # 提取用户特征和时间模式
            self._extract_user_features()
            self._extract_time_patterns()
            
            logger.info(f"提取了 {len(self.service_features)} 个服务特征和 {len(self.user_features)} 个用户特征")
        except Exception as e:
            logger.error(f"提取特征失败: {str(e)}")
    
    def _extract_user_features(self):
        """提取用户特征"""
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            # 获取用户数据，只选择实际存在的字段
            cursor.execute("""
                SELECT id, gender, register_date, last_login_date
                FROM user WHERE role = 'user' AND status = 'active'
            """)
            users = cursor.fetchall()
            
            for user in users:
                user_id = user['id']
                features = {
                    'gender': user.get('gender', 'unknown'),
                    'age': 0  # 使用默认值代替实际年龄
                }
                
                # 计算用户活跃度
                if user.get('register_date') and user.get('last_login_date'):
                    days_since_register = (datetime.now() - user['register_date']).days
                    days_since_login = (datetime.now() - user['last_login_date']).days
                    
                    if days_since_register > 0:
                        features['activity_level'] = max(0, 100 - days_since_login) / days_since_register
                    else:
                        features['activity_level'] = 0
                else:
                    features['activity_level'] = 0
                
                # 计算用户类别偏好
                if user_id in self.user_service_matrix:
                    category_counts = defaultdict(float)
                    total_interactions = 0
                    
                    for service_id, score in self.user_service_matrix[user_id].items():
                        if service_id in self.services:
                            category_id = self.services[service_id].get('category_id')
                            if category_id:
                                category_counts[category_id] += score
                                total_interactions += score
                    
                    # 归一化类别偏好
                    if total_interactions > 0:
                        for category_id, count in category_counts.items():
                            features[f'category_pref_{category_id}'] = count / total_interactions
                
                self.user_features[user_id] = features
            
            logger.info(f"提取了 {len(self.user_features)} 个用户特征")
        except Error as e:
            logger.error(f"提取用户特征时数据库错误: {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()
    
    def _extract_time_patterns(self):
        """提取用户的时间模式"""
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            # 获取用户行为的时间模式
            cursor.execute("""
                SELECT user_id, HOUR(created_at) as hour, 
                       COUNT(*) as count, 
                       DAYOFWEEK(created_at) as day_of_week
                FROM user_behavior
                WHERE created_at IS NOT NULL
                GROUP BY user_id, HOUR(created_at), DAYOFWEEK(created_at)
            """)
            time_patterns = cursor.fetchall()
            
            # 按用户ID分组
            for pattern in time_patterns:
                user_id = pattern['user_id']
                hour = pattern['hour']
                count = pattern['count']
                day_of_week = pattern['day_of_week']
                
                if user_id not in self.time_patterns:
                    self.time_patterns[user_id] = {
                        'hourly': [0] * 24,
                        'daily': [0] * 7,
                        'hourly_daily': [[0 for _ in range(24)] for _ in range(7)]
                    }
                
                # 更新小时统计
                self.time_patterns[user_id]['hourly'][hour] += count
                
                # 更新星期统计 (MySQL的DAYOFWEEK从1开始，1=周日)
                day_idx = (day_of_week - 1) % 7  # 转换为0-6，0=周日
                self.time_patterns[user_id]['daily'][day_idx] += count
                
                # 更新小时+星期组合统计
                self.time_patterns[user_id]['hourly_daily'][day_idx][hour] += count
            
            logger.info(f"提取了 {len(self.time_patterns)} 个用户的时间模式")
        except Error as e:
            logger.error(f"提取时间模式时数据库错误: {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()
    
    def get_context_aware_recommendations(self, user_id, context, limit=10):
        """基于上下文的推荐"""
        try:
            # 获取基础推荐列表
            base_recommendations = self.get_hybrid_recommendations(user_id, limit * 2)
            
            # 如果没有上下文信息或用户没有足够的数据，直接返回基础推荐
            if not context or user_id not in self.user_features:
                return base_recommendations[:limit]
            
            # 提取上下文信息
            current_time = context.get('time', datetime.now().time())
            current_hour = current_time.hour
            current_day = datetime.now().weekday()  # 0-6, 0=周一
            user_location = context.get('location', None)
            
            # 重新排序推荐结果
            scored_recommendations = []
            
            for service_id, base_score in base_recommendations:
                context_score = base_score
                
                # 1. 时间相关性评分
                if service_id in self.services:
                    service = self.services[service_id]
                    
                    # 检查当前是否在营业时间内
                    is_open = False
                    if service.get('open_time_obj') and service.get('close_time_obj'):
                        open_time = service['open_time_obj']
                        close_time = service['close_time_obj']
                        
                        # 处理跨天的情况
                        if close_time < open_time:  # 例如22:00-06:00
                            is_open = current_time >= open_time or current_time <= close_time
                        else:
                            is_open = open_time <= current_time <= close_time
                        
                        # 如果在营业时间内，提高分数
                        if is_open:
                            context_score *= 1.2
                        else:
                            # 如果即将开业（1小时内），稍微提高分数
                            if open_time > current_time:
                                time_diff = (open_time.hour - current_time.hour) * 60 + (open_time.minute - current_time.minute)
                                if 0 < time_diff <= 60:
                                    context_score *= 1.1
                
                # 2. 用户时间模式匹配
                if user_id in self.time_patterns:
                    user_patterns = self.time_patterns[user_id]
                    
                    # 获取用户在当前小时的活跃度
                    hourly_activity = user_patterns['hourly'][current_hour]
                    total_hourly = sum(user_patterns['hourly'])
                    
                    # 获取用户在当前星期几的活跃度
                    daily_activity = user_patterns['daily'][current_day]
                    total_daily = sum(user_patterns['daily'])
                    
                    # 如果用户在当前时间段特别活跃，提高分数
                    if total_hourly > 0 and hourly_activity / total_hourly > 0.1:
                        context_score *= 1 + (hourly_activity / total_hourly)
                    
                    if total_daily > 0 and daily_activity / total_daily > 0.2:
                        context_score *= 1 + (daily_activity / total_daily * 0.5)
                
                # 3. 位置相关性评分
                if user_location and service_id in self.service_locations:
                    service_location = self.service_locations[service_id]
                    
                    # 计算距离（公里）
                    user_coords = (user_location['latitude'], user_location['longitude'])
                    service_coords = (service_location['latitude'], service_location['longitude'])
                    
                    try:
                        # 根据是否安装geopy选择距离计算方法
                        if GEOPY_AVAILABLE:
                            distance = geodesic(user_coords, service_coords).kilometers
                        else:
                            distance = calculate_distance(user_coords, service_coords)
                        
                        # 距离评分：距离越近，分数越高
                        # 使用指数衰减函数，5公里内几乎不衰减，之后迅速衰减
                        distance_factor = math.exp(-max(0, distance - 5) / 10)
                        context_score *= distance_factor
                    except Exception as e:
                        logger.error(f"计算距离失败: {e}")
                
                scored_recommendations.append((service_id, context_score))
            
            # 重新排序并限制结果数量
            scored_recommendations.sort(key=lambda x: x[1], reverse=True)
            return scored_recommendations[:limit]
        except Exception as e:
            logger.error(f"基于上下文的推荐失败: {str(e)}")
            return base_recommendations[:limit]
    
    def _build_ncf_model(self):
        """构建神经协同过滤模型"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow未安装，跳过NCF模型构建")
            return
            
        try:
            logger.info("开始构建神经协同过滤模型...")
            
            # 准备训练数据
            X_train, y_train = self._prepare_ncf_data()
            
            if len(X_train[0]) == 0 or len(y_train) == 0:
                logger.warning("没有足够的训练数据，跳过NCF模型构建")
                return
                
            # 获取用户和服务的最大ID，确保嵌入层大小足够
            max_user_id = max(self.user_features.keys()) if self.user_features else 0
            max_service_id = max(self.services.keys()) if self.services else 0
            
            # 嵌入层大小需要比最大ID大1，因为ID从0开始计数
            num_users = max_user_id + 1
            num_services = max_service_id + 1
            
            # 输出调试信息
            logger.info(f"NCF模型参数：用户数量={num_users}，服务数量={num_services}")
            
            # 定义模型
            user_input = tf.keras.layers.Input(shape=(1,), name='user_input')
            service_input = tf.keras.layers.Input(shape=(1,), name='service_input')
            
            # 嵌入层
            user_embedding = tf.keras.layers.Embedding(
                input_dim=num_users, 
                output_dim=32,
                name='user_embedding'
            )(user_input)
            
            service_embedding = tf.keras.layers.Embedding(
                input_dim=num_services, 
                output_dim=32,
                name='service_embedding'
            )(service_input)
            
            # 展平
            user_vector = tf.keras.layers.Flatten()(user_embedding)
            service_vector = tf.keras.layers.Flatten()(service_embedding)
            
            # 连接
            concat = tf.keras.layers.Concatenate()([user_vector, service_vector])
            
            # 全连接层
            dense1 = tf.keras.layers.Dense(64, activation='relu')(concat)
            dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
            dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout1)
            dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
            output = tf.keras.layers.Dense(1)(dropout2)
            
            # 编译模型
            model = tf.keras.Model([user_input, service_input], output)
            model.compile(
                loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
            )
            
            # 训练模型
            logger.info(f"开始训练NCF模型，训练样本数: {len(y_train)}")
            model.fit(
                X_train, y_train,
                batch_size=64,
                epochs=5,
                verbose=0
            )
            
            self.ncf_model = model
            logger.info("NCF模型训练完成")
            
        except Exception as e:
            logger.error(f"构建NCF模型失败: {str(e)}")
    
    def _prepare_ncf_data(self):
        """准备NCF模型的训练数据"""
        user_ids = []
        service_ids = []
        ratings = []
        
        # 收集所有交互数据
        for user_id, interactions in self.user_service_matrix.items():
            for service_id, score in interactions.items():
                # 确保用户ID和服务ID在有效范围内
                if user_id in self.user_features and service_id in self.services:
                    user_ids.append(user_id)
                    service_ids.append(service_id)
                    ratings.append(score)
        
        # 如果数据太少，返回空
        if len(ratings) < 10:
            return [np.array([]), np.array([])], np.array([])
            
        # 转换为numpy数组
        user_ids = np.array(user_ids)
        service_ids = np.array(service_ids)
        ratings = np.array(ratings)
        
        # 归一化评分
        max_rating = np.max(ratings)
        if max_rating > 0:
            ratings = ratings / max_rating
        
        return [user_ids, service_ids], ratings
    
    def get_ncf_recommendations(self, user_id, limit=10):
        """使用NCF模型获取推荐"""
        if not TF_AVAILABLE or self.ncf_model is None:
            logger.info("NCF模型不可用，无法使用NCF推荐")
            return []
            
        try:
            # 获取所有服务ID
            all_services = list(self.services.keys())
            
            # 排除用户已交互的服务
            interacted = set(self.user_service_matrix.get(user_id, {}).keys())
            candidate_services = [s_id for s_id in all_services if s_id not in interacted]
            
            # 如果候选服务太多，随机选择一部分
            if len(candidate_services) > 100:
                candidate_services = random.sample(candidate_services, 100)
            
            if not candidate_services:
                return []
                
            # 准备预测数据
            user_ids = np.array([user_id] * len(candidate_services))
            service_ids = np.array(candidate_services)
            
            # 预测评分
            predictions = self.ncf_model.predict([user_ids, service_ids], verbose=0)
            
            # 创建服务ID和预测分数的对应关系
            service_scores = [(candidate_services[i], float(predictions[i][0])) 
                             for i in range(len(candidate_services))]
            
            # 排序并返回
            service_scores.sort(key=lambda x: x[1], reverse=True)
            return service_scores[:limit]
        except Exception as e:
            logger.error(f"获取NCF推荐失败: {str(e)}")
            return []
    
    def get_hybrid_recommendations(self, user_id, limit=10):
        """获取混合推荐结果"""
        try:
            # 如果用户没有行为记录，返回热门推荐
            if user_id not in self.user_service_matrix:
                return [(sid, 1.0) for sid in self.popular_services[:limit]]
            
            # 获取用户已交互的服务
            interacted_services = set(self.user_service_matrix[user_id].keys())
            
            # 混合推荐结果
            recommendations = defaultdict(float)
            
            # 1. 基于内容的协同过滤推荐 (权重: 0.4)
            cf_recommendations = self._get_content_based_recommendations(user_id, limit * 2)
            for service_id, score in cf_recommendations:
                recommendations[service_id] += score * 0.4
            
            # 2. 基于SVD的推荐 (权重: 0.3)
            svd_recommendations = self.get_svd_recommendations(user_id, limit * 2)
            for service_id, score in svd_recommendations:
                # 归一化SVD分数
                normalized_score = score / 5.0  # 假设最大分数为5
                recommendations[service_id] += normalized_score * 0.3
            
            # 3. 基于NCF的推荐 (权重: 0.3)
            if TF_AVAILABLE and self.ncf_model is not None:
                ncf_recommendations = self.get_ncf_recommendations(user_id, limit * 2)
                for service_id, score in ncf_recommendations:
                    recommendations[service_id] += score * 0.3
            
            # 排序并限制数量
            sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            
            # 如果推荐结果不足，补充热门服务
            if len(sorted_recommendations) < limit:
                for service_id in self.popular_services:
                    if service_id not in interacted_services and service_id not in [r[0] for r in sorted_recommendations]:
                        sorted_recommendations.append((service_id, 0.1))  # 低分，表示是热门补充
                        if len(sorted_recommendations) >= limit:
                            break
            
            return sorted_recommendations[:limit]
        except Exception as e:
            logger.error(f"获取混合推荐失败: {str(e)}")
            return [(sid, 1.0) for sid in self.popular_services[:limit]]
    
    def _get_content_based_recommendations(self, user_id, limit=10):
        """基于内容的协同过滤推荐"""
        recommendations = defaultdict(float)
        
        # 获取用户交互过的服务
        if user_id not in self.user_service_matrix:
            return []
            
        interacted_services = self.user_service_matrix[user_id]
        
        # 基于用户交互的服务，找出相似服务
        for service_id, interaction_strength in interacted_services.items():
            if service_id not in self.service_similarity_matrix:
                continue
                
            similar_services = self.service_similarity_matrix[service_id]
            
            for similar_id, similarity in similar_services.items():
                if similar_id not in interacted_services:
                    recommendations[similar_id] += similarity * interaction_strength
        
        # 排序并返回
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:limit]
    
    def get_diversified_recommendations(self, recommendations, limit=10):
        """增加推荐结果的多样性"""
        if not recommendations:
            return []
            
        # 初始化类别计数
        category_counts = defaultdict(int)
        diversified = []
        remaining = list(recommendations)
        
        while remaining and len(diversified) < limit:
            # 计算每个项目的多样性得分
            scores = []
            for item in remaining:
                service_id, original_score = item
                
                # 获取服务类别
                category_id = None
                if service_id in self.services:
                    category_id = self.services[service_id].get('category_id')
                
                # 类别多样性惩罚
                category_penalty = category_counts.get(category_id, 0) if category_id else 0
                
                # 多样性得分 = 原始分数 - 0.2 * 类别惩罚
                diversity_score = original_score - 0.2 * category_penalty
                scores.append((item, diversity_score))
            
            # 选择多样性得分最高的项目
            if not scores:
                break
                
            best_item, _ = max(scores, key=lambda x: x[1])
            diversified.append(best_item)
            remaining.remove(best_item)
            
            # 更新类别计数
            service_id = best_item[0]
            if service_id in self.services:
                category_id = self.services[service_id].get('category_id')
                if category_id:
                    category_counts[category_id] += 1
        
        return diversified
    
    def _build_ab_test_groups(self):
        """构建A/B测试分组"""
        try:
            logger.info("开始构建A/B测试分组...")
            
            # 获取所有用户ID
            user_ids = list(self.user_features.keys())
            
            # 创建测试组
            self.ab_test_groups = {
                'recommendation_algorithm': {},  # 推荐算法测试
                'diversity_level': {},          # 多样性水平测试
            }
            
            # 为每个用户分配测试组
            for user_id in user_ids:
                # 1. 推荐算法测试：将用户分为三组
                # A组：混合推荐，B组：SVD推荐，C组：NCF推荐
                algo_group = hash(f"algo_{user_id}") % 3
                if algo_group == 0:
                    self.ab_test_groups['recommendation_algorithm'][user_id] = 'hybrid'
                elif algo_group == 1:
                    self.ab_test_groups['recommendation_algorithm'][user_id] = 'svd'
                else:
                    self.ab_test_groups['recommendation_algorithm'][user_id] = 'ncf'
                
                # 2. 多样性水平测试：将用户分为三组
                # A组：低多样性，B组：中多样性，C组：高多样性
                div_group = hash(f"div_{user_id}") % 3
                if div_group == 0:
                    self.ab_test_groups['diversity_level'][user_id] = 'low'
                elif div_group == 1:
                    self.ab_test_groups['diversity_level'][user_id] = 'medium'
                else:
                    self.ab_test_groups['diversity_level'][user_id] = 'high'
            
            logger.info(f"A/B测试分组完成，共 {len(user_ids)} 个用户")
        except Exception as e:
            logger.error(f"构建A/B测试分组失败: {str(e)}")
    
    def get_ab_test_recommendations(self, user_id, limit=10, diversify=False):
        """根据A/B测试分组获取推荐结果"""
        try:
            # 如果用户不在A/B测试分组中，使用默认推荐
            if user_id not in self.ab_test_groups.get('recommendation_algorithm', {}):
                return self.get_context_aware_recommendations(user_id, limit)
            
            # 获取用户的测试组
            algo_group = self.ab_test_groups['recommendation_algorithm'].get(user_id, 'hybrid')
            diversity_group = self.ab_test_groups['diversity_level'].get(user_id, 'medium')
            
            # 根据算法组选择推荐算法
            if algo_group == 'svd':
                base_recommendations = self.get_svd_recommendations(user_id, limit * 2)
                if not base_recommendations:  # 如果SVD推荐失败，回退到混合推荐
                    base_recommendations = self.get_hybrid_recommendations(user_id, limit * 2)
            elif algo_group == 'ncf' and TF_AVAILABLE and self.ncf_model:
                base_recommendations = self.get_ncf_recommendations(user_id, limit * 2)
                if not base_recommendations:  # 如果NCF推荐失败，回退到混合推荐
                    base_recommendations = self.get_hybrid_recommendations(user_id, limit * 2)
            else:  # 默认使用混合推荐
                base_recommendations = self.get_hybrid_recommendations(user_id, limit * 2)
            
            # 应用时间感知（如果是当前时间）
            current_time = datetime.now().time()
            time_recommendations = self.get_time_aware_recommendations(user_id, current_time, limit * 2)
            
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
                
                recommendations = self._apply_diversity(recommendations, diversity_strength, limit)
            else:
                recommendations = recommendations[:limit]
            
            return recommendations
        except Exception as e:
            logger.error(f"A/B测试推荐失败: {str(e)}")
            # 失败时回退到混合推荐
            return self.get_hybrid_recommendations(user_id, limit)
    
    def _apply_diversity(self, recommendations, diversity_strength, limit):
        """应用多样性增强，根据多样性强度参数"""
        if not recommendations:
            return []
            
        # 初始化类别计数
        category_counts = defaultdict(int)
        diversified = []
        remaining = list(recommendations)
        
        while remaining and len(diversified) < limit:
            # 计算每个项目的多样性得分
            scores = []
            for item in remaining:
                service_id, original_score = item
                
                # 获取服务类别
                category_id = None
                if service_id in self.services:
                    category_id = self.services[service_id].get('category_id')
                
                # 类别多样性惩罚
                category_penalty = category_counts.get(category_id, 0) if category_id else 0
                
                # 多样性得分 = 原始分数 - 多样性强度 * 类别惩罚
                diversity_score = original_score - diversity_strength * category_penalty
                scores.append((item, diversity_score))
            
            # 选择多样性得分最高的项目
            if not scores:
                break
                
            best_item, _ = max(scores, key=lambda x: x[1])
            diversified.append(best_item)
            remaining.remove(best_item)
            
            # 更新类别计数
            service_id = best_item[0]
            if service_id in self.services:
                category_id = self.services[service_id].get('category_id')
                if category_id:
                    category_counts[category_id] += 1
        
        return diversified
    
    def generate_recommendation_explanations(self, user_id, recommendations):
        """为推荐结果生成解释"""
        explanations = {}
        
        # 如果用户没有特征数据，无法生成个性化解释
        if user_id not in self.user_features:
            return {str(r[0]): "这是一个热门服务" for r in recommendations}
        
        # 获取用户特征
        user_features = self.user_features[user_id]
        
        # 获取用户的类别偏好
        category_prefs = {}
        for key, value in user_features.items():
            if key.startswith('category_pref_'):
                category_id = int(key.replace('category_pref_', ''))
                category_prefs[category_id] = value
        
        # 为每个推荐生成解释
        for service_id, score in recommendations:
            if service_id not in self.services:
                explanations[str(service_id)] = "这是一个推荐服务"
                continue
                
            service = self.services[service_id]
            reasons = []
            
            # 1. 基于类别偏好
            category_id = service.get('category_id')
            if category_id and category_id in category_prefs and category_prefs[category_id] > 0.2:
                category_name = service.get('category', f"类别{category_id}")
                reasons.append(f"您喜欢{category_name}类服务")
            
            # 2. 基于评分
            rating = float(service.get('rating', 0) or 0)
            if rating >= 4.5:
                reasons.append("这是一个高评分服务")
            elif rating >= 4.0:
                reasons.append("这是一个好评服务")
            
            # 3. 基于相似服务
            similar_services = []
            for interacted_id, interaction_score in self.user_service_matrix.get(user_id, {}).items():
                if interacted_id in self.service_similarity_matrix and service_id in self.service_similarity_matrix[interacted_id]:
                    similarity = self.service_similarity_matrix[interacted_id][service_id]
                    if similarity > 0.7 and interaction_score > 3:
                        if interacted_id in self.services:
                            similar_services.append(self.services[interacted_id]['name'])
            
            if similar_services:
                if len(similar_services) == 1:
                    reasons.append(f"与您喜欢的服务\"{similar_services[0]}\"相似")
                else:
                    reasons.append(f"与您喜欢的多个服务相似")
            
            # 4. 基于热门度
            if service.get('recommended'):
                reasons.append("这是平台推荐的服务")
            elif service.get('review_count', 0) > 50:
                reasons.append("这是一个热门服务")
            
            # 如果没有找到原因，添加一个通用原因
            if not reasons:
                reasons.append("这可能符合您的兴趣")
            
            # 组合解释（最多使用2个原因）
            explanation = "，".join(reasons[:2])
            explanations[str(service_id)] = explanation
        
        return explanations

    def get_time_aware_recommendations(self, user_id, current_time=None, limit=10):
        """基于时间的推荐"""
        try:
            # 获取基础推荐列表
            base_recommendations = self.get_hybrid_recommendations(user_id, limit * 2)
            
            # 如果没有时间信息或用户没有足够的数据，直接返回基础推荐
            if not current_time or user_id not in self.time_patterns:
                return base_recommendations[:limit]
            
            # 提取时间信息
            current_hour = current_time.hour
            current_day = datetime.now().weekday()  # 0-6, 0=周一
            
            # 重新排序推荐结果
            scored_recommendations = []
            
            for service_id, base_score in base_recommendations:
                time_score = base_score
                
                # 用户时间模式匹配
                if user_id in self.time_patterns:
                    user_patterns = self.time_patterns[user_id]
                    
                    # 获取用户在当前小时的活跃度
                    hourly_activity = user_patterns['hourly'][current_hour]
                    total_hourly = sum(user_patterns['hourly'])
                    
                    # 获取用户在当前星期几的活跃度
                    daily_activity = user_patterns['daily'][current_day]
                    total_daily = sum(user_patterns['daily'])
                    
                    # 如果用户在当前时间段特别活跃，提高分数
                    if total_hourly > 0 and hourly_activity / total_hourly > 0.1:
                        time_score *= 1 + (hourly_activity / total_hourly)
                    
                    if total_daily > 0 and daily_activity / total_daily > 0.2:
                        time_score *= 1 + (daily_activity / total_daily * 0.5)
                
                scored_recommendations.append((service_id, time_score))
            
            # 重新排序并限制结果数量
            scored_recommendations.sort(key=lambda x: x[1], reverse=True)
            return scored_recommendations[:limit]
        except Exception as e:
            logger.error(f"基于时间的推荐失败: {str(e)}")
            return base_recommendations[:limit]

# 创建数据缓存实例
data_cache = DataCache()

# 记录推荐日志函数
def record_recommendation_log(user_id, service_ids, scores=None, source_type=None, algorithm=None, reason=None):
    """记录推荐日志到数据库"""
    if not service_ids:
        return
    
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        for i, service_id_str in enumerate(service_ids):
            try:
                service_id = int(service_id_str)
                # 默认分数
                score = 0.0
                
                # 如果提供了分数，使用提供的分数
                if scores and service_id in scores:
                    score = float(scores[service_id])
                
                # 构建额外数据
                extra_data = None
                if source_type:
                    extra_data = json.dumps({"source_type": source_type, "position": i})
                
                # 插入推荐日志
                query = """
                INSERT INTO recommendation_log 
                (user_id, service_id, score, is_clicked, algorithm, reason, created_at)
                VALUES (%s, %s, %s, 0, %s, %s, NOW())
                """
                cursor.execute(query, (user_id, service_id, score, algorithm, reason))
            except Exception as e:
                logger.error(f"记录推荐日志失败 (服务ID: {service_id_str}): {e}")
        
        conn.commit()
        logger.info(f"已记录 {len(service_ids)} 条推荐日志，用户ID: {user_id}")
    except Error as e:
        logger.error(f"记录推荐日志数据库错误: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()

# 依赖项：获取数据缓存
async def get_data_cache(force_update: bool = False):
    if data_cache.needs_update(force_update):
        data_cache.update()
    return data_cache

# API路由
@app.get("/")
async def root():
    return {"message": "ezServe推荐系统API", "status": "running"}

@app.get("/recommend")
async def recommend_for_user(
    user_id: int, 
    limit: int = Query(10, ge=1, le=50),
    diversify: bool = Query(False),
    explain: bool = Query(False),
    ab_test: bool = Query(True),
    cache: DataCache = Depends(get_data_cache)
):
    """为指定用户推荐服务"""
    logger.info(f"为用户 {user_id} 推荐服务，限制 {limit} 条，多样性: {diversify}, 解释: {explain}, A/B测试: {ab_test}")
    
    # 如果是新用户或没有行为记录，返回热门服务
    if user_id not in cache.user_service_matrix:
        logger.info(f"用户 {user_id} 没有行为记录，返回热门服务")
        recommended_ids = [str(sid) for sid in cache.popular_services[:limit]]
        
        # 记录推荐日志（热门推荐）
        record_recommendation_log(user_id, recommended_ids, source_type="popular", 
                                 algorithm="popularity", reason="新用户或无行为记录")
        
        response = {"user_id": user_id, "items": recommended_ids}
    
        # 如果需要解释，添加简单解释
        if explain:
            explanations = {sid: "这是一个热门服务" for sid in recommended_ids}
            response["explanations"] = explanations
        
        return response
    
    # 根据是否使用A/B测试选择推荐方法
    if ab_test:
        recommendations = cache.get_ab_test_recommendations(user_id, limit * 2, diversify)
        algorithm = "ab_test"
        reason = "A/B测试推荐"
    else:
        # 获取当前时间
        current_time = datetime.now().time()
        
        # 获取时间感知推荐
        recommendations = cache.get_time_aware_recommendations(user_id, current_time, limit * 2)
        
        # 如果需要多样性，应用多样性增强
        if diversify:
            recommendations = cache.get_diversified_recommendations(recommendations, limit)
            algorithm = "time_aware_diversified"
            reason = "基于时间的多样化推荐"
        else:
            recommendations = recommendations[:limit]
            algorithm = "time_aware"
            reason = "基于时间的推荐"
    
    # 提取推荐的服务ID和分数
    recommended_ids = [str(r[0]) for r in recommendations]
    recommendation_scores = {r[0]: r[1] for r in recommendations}
    
    # 记录推荐日志
    record_recommendation_log(user_id, recommended_ids, scores=recommendation_scores, 
                             source_type="personalized", algorithm=algorithm, reason=reason)
    
    # 构建响应
    response = {"user_id": user_id, "items": recommended_ids}
    
    # 如果需要解释，生成推荐解释
    if explain:
        explanations = cache.generate_recommendation_explanations(user_id, recommendations)
        response["explanations"] = explanations
    
    return response

@app.get("/debug/model-status")
async def model_status(cache: DataCache = Depends(get_data_cache)):
    """查看模型状态"""
    svd_status = "未构建"
    if cache.svd_model:
        svd_status = {
            "潜在因子数": len(cache.svd_model['sigma']),
            "用户数": len(cache.svd_model['user_idx']),
            "服务数": len(cache.svd_model['service_idx'])
        }
    
    ncf_status = "未构建"
    if TF_AVAILABLE and cache.ncf_model:
        ncf_status = {
            "模型类型": str(type(cache.ncf_model)),
            "已训练": True
        }
    elif not TF_AVAILABLE:
        ncf_status = "TensorFlow未安装"
    
    return {
        "last_update": str(cache.last_update),
        "services_count": len(cache.services),
        "users_count": len(cache.user_behaviors),
        "user_service_matrix_size": len(cache.user_service_matrix),
        "service_similarity_matrix_size": len(cache.service_similarity_matrix),
        "popular_services_count": len(cache.popular_services),
        "popular_services_top10": cache.popular_services[:10],
        "svd_model": svd_status,
        "ncf_model": ncf_status,
        "user_features_count": len(cache.user_features),
        "service_features_count": len(cache.service_features)
    }

@app.get("/debug/recommendation-logs")
async def recommendation_logs(
    user_id: int = None,
    service_id: int = None,
    limit: int = 20
):
    """查看推荐日志记录"""
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # 构建查询条件
        conditions = []
        params = []
        
        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)
        
        if service_id:
            conditions.append("service_id = %s")
            params.append(service_id)
        
        # 构建SQL查询
        query = "SELECT * FROM recommendation_log"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        # 执行查询
        cursor.execute(query, params)
        logs = cursor.fetchall()
        
        # 处理datetime对象以便JSON序列化
        for log in logs:
            if 'created_at' in log and log['created_at']:
                log['created_at'] = log['created_at'].isoformat()
        
        return {
            "count": len(logs),
            "logs": logs
        }
    except Error as e:
        logger.error(f"查询推荐日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn and conn.is_connected():
            conn.close()

@app.get("/recommend/similar")
async def similar_services(
    service_id: int,
    limit: int = Query(10, ge=1, le=50),
    cache: DataCache = Depends(get_data_cache)
):
    """获取相似服务"""
    logger.info(f"获取服务 {service_id} 的相似服务，限制 {limit} 条")
    
    # 检查服务是否存在
    if service_id not in cache.services:
        raise HTTPException(status_code=404, detail="服务不存在")
    
    # 获取相似服务
    if service_id in cache.service_similarity_matrix:
        similar_services = cache.service_similarity_matrix[service_id]
        sorted_similar = sorted(similar_services.items(), key=lambda x: x[1], reverse=True)
        
        # 排除自身
        similar_ids = [str(s[0]) for s in sorted_similar if s[0] != service_id][:limit]
        
        return {"service_id": service_id, "items": similar_ids}
    else:
        # 如果没有相似度数据，返回同类别的服务
        category_id = cache.services[service_id]['category_id']
        same_category = [str(s_id) for s_id, s in cache.services.items() 
                        if s['category_id'] == category_id and s_id != service_id]
        
        return {"service_id": service_id, "items": same_category[:limit]}

@app.get("/recommend/trending")
async def trending_services(
    limit: int = Query(10, ge=1, le=50),
    cache: DataCache = Depends(get_data_cache)
):
    """获取热门服务"""
    logger.info(f"获取热门服务，限制 {limit} 条")
    return {"items": [str(sid) for sid in cache.popular_services[:limit]]}

@app.post("/recommend/train")
async def train_model(cache: DataCache = Depends(get_data_cache)):
    """触发模型训练"""
    logger.info("触发模型训练")
    try:
        # 强制更新缓存数据
        cache.update()
        return {"success": True, "message": "模型训练成功"}
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")

# 初始化数据
@app.on_event("startup")
async def startup_event():
    logger.info("推荐系统启动，初始化数据...")
    data_cache.update()
    logger.info("数据初始化完成")

@app.post("/recommendation/feedback")
async def recommendation_feedback(
    user_id: int,
    service_id: int,
    is_clicked: bool = False,
    feedback_type: str = None
):
    """记录推荐反馈"""
    logger.info(f"收到推荐反馈：用户{user_id}, 服务{service_id}, 点击={is_clicked}, 类型={feedback_type}")
    
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 先检查是否有最近的推荐记录
        check_query = """
        SELECT id FROM recommendation_log 
        WHERE user_id = %s AND service_id = %s
        ORDER BY created_at DESC LIMIT 1
        """
        cursor.execute(check_query, (user_id, service_id))
        existing_record = cursor.fetchone()
        
        if existing_record:
            # 如果有记录，更新它
            record_id = existing_record[0]
            update_query = """
            UPDATE recommendation_log 
            SET is_clicked = %s,
                algorithm = %s,
                reason = %s
            WHERE id = %s
            """
            cursor.execute(update_query, (1 if is_clicked else 0, "feedback", f"用户反馈: {feedback_type}", record_id))
            logger.info(f"已更新推荐记录 ID: {record_id}")
        else:
            # 如果没有记录，创建一个新记录
            insert_query = """
            INSERT INTO recommendation_log 
            (user_id, service_id, score, is_clicked, algorithm, reason, created_at)
            VALUES (%s, %s, 0.0, %s, %s, %s, NOW())
            """
            reason = f"用户直接反馈: {feedback_type}"
            cursor.execute(insert_query, (user_id, service_id, 1 if is_clicked else 0, 
                                        "feedback", reason))
            logger.info(f"已创建新的推荐反馈记录，用户ID: {user_id}, 服务ID: {service_id}")
        
        affected_rows = cursor.rowcount
        conn.commit()
        
        return {"success": True, "updated_rows": affected_rows}
    except Error as e:
        logger.error(f"记录推荐反馈失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn and conn.is_connected():
            conn.close()

@app.post("/update-behavior")
async def update_behavior(behavior: dict):
    """处理用户行为更新"""
    logger.info(f"收到用户行为更新：{behavior}")
    
    try:
        user_id = behavior.get('userId')
        service_id = behavior.get('serviceId')
        behavior_type = behavior.get('behaviorType')
        duration = behavior.get('duration', 0)
        
        if not user_id or not service_id or not behavior_type:
            logger.error("行为数据不完整")
            return {"success": False, "error": "缺少必要参数"}
        
        # 将行为数据保存到数据库（可选，因为Java端已经保存）
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            
            # 检查是否已经存在相同记录（避免重复）
            query = """
            SELECT id FROM user_behavior 
            WHERE user_id = %s AND service_id = %s AND behavior_type = %s
            AND created_at > DATE_SUB(NOW(), INTERVAL 1 MINUTE)
            """
            cursor.execute(query, (user_id, service_id, behavior_type))
            exists = cursor.fetchone()
            
            # 如果1分钟内没有相同记录，则插入
            if not exists:
                extra_data = behavior.get('extraData')
                query = """
                INSERT INTO user_behavior (user_id, service_id, behavior_type, duration, extra_data, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                """
                cursor.execute(query, (user_id, service_id, behavior_type, duration, extra_data))
                conn.commit()
                logger.info(f"已保存用户行为：用户{user_id}，服务{service_id}，行为{behavior_type}")
        except Error as e:
            logger.error(f"保存行为数据失败: {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()
        
        # 判断是否需要更新缓存
        if data_cache.last_update:
            # 添加行为到内存中
            behavior_record = {
                'user_id': user_id,
                'service_id': service_id,
                'behavior_type': behavior_type,
                'duration': duration
            }
            
            if user_id not in data_cache.user_behaviors:
                data_cache.user_behaviors[user_id] = []
            
            data_cache.user_behaviors[user_id].append(behavior_record)
            
            # 更新用户-服务交互矩阵
            behavior_weights = {
                'view': 1.0,
                'click': 2.0,
                'favorite': 3.0,
                'unfavorite': -3.0,  # 取消收藏，抵消收藏行为
                'order': 5.0
            }
            
            if user_id not in data_cache.user_service_matrix:
                data_cache.user_service_matrix[user_id] = {}
            
            if service_id not in data_cache.user_service_matrix[user_id]:
                data_cache.user_service_matrix[user_id][service_id] = 0
            
            weight = behavior_weights.get(behavior_type, 1.0)
            if behavior_type == 'view' and duration > 0:
                time_factor = min(duration / 60, 5)  # 最多5分钟
                data_cache.user_service_matrix[user_id][service_id] += weight * (1 + time_factor)
            elif behavior_type == 'unfavorite':
                # 对于取消收藏，确保不会将分数降到负值
                data_cache.user_service_matrix[user_id][service_id] = max(0, data_cache.user_service_matrix[user_id][service_id] + weight)
                logger.info(f"用户{user_id}取消收藏服务{service_id}，更新交互分数")
            else:
                data_cache.user_service_matrix[user_id][service_id] += weight
            
            logger.info(f"已更新用户{user_id}的交互矩阵")
        
        return {"success": True}
    except Exception as e:
        logger.error(f"处理用户行为失败: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/debug/user/{user_id}/behaviors")
async def get_user_behaviors(
    user_id: int,
    cache: DataCache = Depends(get_data_cache)
):
    """获取用户行为数据详情"""
    logger.info(f"获取用户 {user_id} 的行为数据")
    
    conn = None
    result = {
        "user_id": user_id,
        "behaviors": [],
        "interaction_matrix": {},
        "recent_recommendations": [],
        "recommendation_feedbacks": [],
        "recommendation_source": "unknown"
    }
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # 获取用户行为数据
        query = """
        SELECT id, user_id, service_id, behavior_type, duration, extra_data, created_at
        FROM user_behavior
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 50
        """
        cursor.execute(query, (user_id,))
        behaviors = cursor.fetchall()
        
        # 处理日期格式以便JSON序列化
        for behavior in behaviors:
            if 'created_at' in behavior and behavior['created_at']:
                behavior['created_at'] = behavior['created_at'].isoformat()
            # 添加服务名称
            service_id = behavior['service_id']
            if service_id in cache.services:
                behavior['service_name'] = cache.services[service_id]['name']
        
        result["behaviors"] = behaviors
        
        # 获取交互矩阵数据
        if user_id in cache.user_service_matrix:
            interaction_data = []
            for service_id, score in cache.user_service_matrix[user_id].items():
                service_info = {
                    "service_id": service_id,
                    "interaction_score": score
                }
                if service_id in cache.services:
                    service_info["service_name"] = cache.services[service_id]['name']
                    service_info["category"] = cache.services[service_id]['category']
                interaction_data.append(service_info)
            
            # 按交互分数排序
            interaction_data.sort(key=lambda x: x["interaction_score"], reverse=True)
            result["interaction_matrix"] = interaction_data
        
        # 获取最近的推荐记录
        query = """
        SELECT id, service_id, score, is_clicked, algorithm, reason, created_at
        FROM recommendation_log
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 20
        """
        cursor.execute(query, (user_id,))
        recommendations = cursor.fetchall()
        
        # 处理日期格式
        for rec in recommendations:
            if 'created_at' in rec and rec['created_at']:
                rec['created_at'] = rec['created_at'].isoformat()
            # 添加服务名称
            service_id = rec['service_id']
            if service_id in cache.services:
                rec['service_name'] = cache.services[service_id]['name']
        
        result["recent_recommendations"] = recommendations
        
        # 确定推荐来源
        if user_id in cache.user_service_matrix and len(cache.user_service_matrix[user_id]) > 0:
            result["recommendation_source"] = "personalized"
        else:
            result["recommendation_source"] = "popular"
        
        # 获取用户推荐反馈
        query = """
        SELECT r.id, r.user_id, r.service_id, r.algorithm, r.reason, r.created_at
        FROM recommendation_log r
        WHERE r.user_id = %s AND r.reason LIKE '%用户反馈%'
        ORDER BY r.created_at DESC
        LIMIT 20
        """
        cursor.execute(query, (user_id,))
        feedbacks = cursor.fetchall()
        
        # 处理日期格式
        for feedback in feedbacks:
            if 'created_at' in feedback and feedback['created_at']:
                feedback['created_at'] = feedback['created_at'].isoformat()
            # 添加服务名称
            service_id = feedback['service_id']
            if service_id in cache.services:
                feedback['service_name'] = cache.services[service_id]['name']
        
        result["recommendation_feedbacks"] = feedbacks
        
        return result
    except Error as e:
        logger.error(f"获取用户行为数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn and conn.is_connected():
            conn.close()

@app.get("/debug/service/{service_id}/similarity")
async def get_service_similarity(
    service_id: int,
    limit: int = 20,
    cache: DataCache = Depends(get_data_cache)
):
    """获取服务相似度详情"""
    logger.info(f"获取服务 {service_id} 的相似度详情")
    
    # 检查服务是否存在
    if service_id not in cache.services:
        raise HTTPException(status_code=404, detail="服务不存在")
    
    result = {
        "service_id": service_id,
        "service_name": cache.services[service_id]['name'],
        "service_details": cache.services[service_id],
        "similar_services": []
    }
    
    # 获取相似服务
    if service_id in cache.service_similarity_matrix:
        similar_services = cache.service_similarity_matrix[service_id]
        sorted_similar = sorted(similar_services.items(), key=lambda x: x[1], reverse=True)
        
        # 排除自身并限制数量
        similar_data = []
        for similar_id, similarity in sorted_similar:
            if similar_id != service_id:
                service_info = {
                    "service_id": similar_id,
                    "similarity_score": similarity
                }
                if similar_id in cache.services:
                    service_info["service_name"] = cache.services[similar_id]['name']
                    service_info["category"] = cache.services[similar_id]['category']
                similar_data.append(service_info)
                
                if len(similar_data) >= limit:
                    break
        
        result["similar_services"] = similar_data
    
    return result

@app.get("/debug/ab-test-status")
async def ab_test_status(cache: DataCache = Depends(get_data_cache)):
    """查看A/B测试状态"""
    # 统计各组用户数量
    stats = {}
    for test_name, groups in cache.ab_test_groups.items():
        stats[test_name] = {}
        for group_name in set(groups.values()):
            stats[test_name][group_name] = sum(1 for g in groups.values() if g == group_name)
    
    return {
        "active_tests": list(cache.ab_test_groups.keys()),
        "user_counts": stats,
        "total_users": len(cache.user_features)
    }

@app.get("/debug/explain-recommendation")
async def explain_recommendation(
    user_id: int,
    service_id: int,
    cache: DataCache = Depends(get_data_cache)
):
    """获取单个推荐的详细解释"""
    if service_id not in cache.services:
        raise HTTPException(status_code=404, detail="服务不存在")
    
    service = cache.services[service_id]
    
    # 基础服务信息
    result = {
        "service_id": service_id,
        "service_name": service.get('name', ''),
        "category": service.get('category', ''),
        "rating": float(service.get('rating', 0) or 0),
        "review_count": int(service.get('review_count', 0) or 0),
    }
    
    # 如果用户不存在，只返回基础信息
    if user_id not in cache.user_features:
        result["explanation"] = "这是一个推荐服务"
        return result
    
    # 获取用户与该服务的交互
    interaction_score = 0
    if user_id in cache.user_service_matrix and service_id in cache.user_service_matrix[user_id]:
        interaction_score = cache.user_service_matrix[user_id][service_id]
    
    result["user_interaction_score"] = interaction_score
    
    # 获取用户特征
    user_features = cache.user_features[user_id]
    
    # 获取用户的类别偏好
    category_prefs = {}
    for key, value in user_features.items():
        if key.startswith('category_pref_'):
            category_id = int(key.replace('category_pref_', ''))
            category_prefs[category_id] = value
    
    # 计算类别匹配度
    category_match = 0
    category_id = service.get('category_id')
    if category_id and category_id in category_prefs:
        category_match = category_prefs[category_id]
    
    result["category_match"] = category_match
    
    # 查找相似服务
    similar_services = []
    for interacted_id, interaction_score in cache.user_service_matrix.get(user_id, {}).items():
        if interacted_id in cache.service_similarity_matrix and service_id in cache.service_similarity_matrix[interacted_id]:
            similarity = cache.service_similarity_matrix[interacted_id][service_id]
            if similarity > 0.5 and interaction_score > 2:
                if interacted_id in cache.services:
                    similar_services.append({
                        "service_id": interacted_id,
                        "service_name": cache.services[interacted_id]['name'],
                        "similarity": similarity,
                        "user_interaction": interaction_score
                    })
    
    # 按相似度排序
    similar_services.sort(key=lambda x: x["similarity"] * x["user_interaction"], reverse=True)
    result["similar_services"] = similar_services[:5]
    
    # 生成文本解释
    explanations = cache.generate_recommendation_explanations(user_id, [(service_id, 1.0)])
    result["explanation"] = explanations.get(str(service_id), "这是一个推荐服务")
    
    return result

# 运行服务
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
