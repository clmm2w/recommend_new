import os
import sys
import random
import uuid
import mysql.connector
from datetime import datetime, timedelta
import numpy as np

# ================= 1. 路径修复逻辑 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from recommend.config import DB_CONFIG, logger

# ================= 2. 实验配置 =================
NUM_USERS = 5000  # 每次生成1000名模拟用户
NOISE_RATE = 0.12  # 12% 噪音行为
AVG_ACTIONS = 15  # 平均每人交互次数


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def fetch_services(cursor):
    """获取服务及其属性，用于匹配人设"""
    cursor.execute("SELECT id, name, category_id, price, rating, provider_id FROM service WHERE status = 'active'")
    services = cursor.fetchall()
    if not services: return None, None, None, None

    services_by_cat = {}
    for s in services:
        cat = s['category_id']
        if cat not in services_by_cat: services_by_cat[cat] = []
        services_by_cat[cat].append(s)

    high_q = [s for s in services if (s['rating'] or 0) >= 4.5]
    low_p = [s for s in services if (s['price'] or 1000) < 150]
    return services, services_by_cat, high_q, low_p


def generate_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    all_s, s_by_cat, high_q, low_p = fetch_services(cursor)
    if not all_s:
        print("❌ 错误：service表为空，请先录入服务数据！")
        return

    all_cats = list(s_by_cat.keys())

    # 1. 数据库清理
    choice = input("⚠️ 是否清空所有模拟用户、行为及订单数据 (y/n)? ").strip().lower()
    if choice == 'y':
        cursor.execute("SET FOREIGN_KEY_CHECKS=0")
        cursor.execute("TRUNCATE TABLE user_behavior")
        cursor.execute("TRUNCATE TABLE orders")
        cursor.execute("DELETE FROM user WHERE role = 'user' AND name LIKE 'sim_%'")
        cursor.execute("SET FOREIGN_KEY_CHECKS=1")
        conn.commit()
        print("🧹 数据库已重置")

    # 2. 生成逻辑
    users_batch = []
    behaviors_batch = []
    orders_batch = []

    cursor.execute("SELECT MAX(id) FROM user")
    start_uid = (cursor.fetchone()['MAX(id)'] or 0) + 1

    print(f"📊 正在生成 {NUM_USERS} 个高质量模拟人设并匹配行为...")

    for i in range(NUM_USERS):
        uid = start_uid + i
        reg_time = datetime.now() - timedelta(days=random.randint(5, 90))

        # 适配字段：name, email, password, phone, address, role, register_date
        u_name = f"sim_{uid}"
        users_batch.append((
            u_name, f"{u_name}@example.com", "123456",
            f"138{random.randint(10000000, 99999999)}",
            "模拟测试地址", "user", reg_time
        ))

        # 活跃度服从指数分布（长尾定律）
        num_actions = int(np.random.exponential(scale=AVG_ACTIONS)) + 3
        num_actions = min(num_actions, 60)

        # 分配人设
        persona = random.choices(['loyalist', 'value_hunter', 'quality_pro', 'random'],
                                 weights=[0.5, 0.2, 0.2, 0.1], k=1)[0]

        # 确定池子
        if persona == 'loyalist':
            target_cats = random.sample(all_cats, k=min(2, len(all_cats)))
            pool = [s for c in target_cats for s in s_by_cat[c]]
        elif persona == 'value_hunter':
            pool = low_p
        elif persona == 'quality_pro':
            pool = high_q
        else:
            pool = all_s

        if not pool: pool = all_s

        interacted_sids = set()
        current_time = reg_time

        for _ in range(num_actions):
            current_time += timedelta(minutes=random.randint(10, 300))
            if current_time > datetime.now(): break

            target_svc = random.choice(all_s if random.random() < NOISE_RATE else pool)
            sid = target_svc['id']
            if sid in interacted_sids: continue
            interacted_sids.add(sid)

            # 决定行为类型 (view, click, favorite)
            b_type = random.choices(['view', 'click', 'favorite'], weights=[0.6, 0.3, 0.1], k=1)[0]
            behaviors_batch.append((uid, sid, b_type, current_time))

            # 决定是否下单 (转化漏斗)
            order_prob = 0.35 if b_type == 'click' and persona != 'random' else 0.05
            if random.random() < order_prob:
                order_time = current_time + timedelta(minutes=random.randint(2, 20))
                # 构造订单：order_number, user_id, service_id, provider_id, order_time, service_time, amount, status, address, contact_name, contact_phone
                order_no = f"ORD{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}"
                orders_batch.append((
                    order_no, uid, sid, target_svc['provider_id'] or 1,
                    order_time, order_time + timedelta(days=1),
                    target_svc['price'], 'completed', '模拟服务地址', u_name, '13800000000'
                ))

    # 3. 执行批量写入
    print("💾 正在同步至数据库...")
    cursor.executemany(
        "INSERT INTO user (name, email, password, phone, address, role, register_date) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        users_batch
    )

    for b in range(0, len(behaviors_batch), 2000):
        cursor.executemany(
            "INSERT INTO user_behavior (user_id, service_id, behavior_type, created_at) VALUES (%s, %s, %s, %s)",
            behaviors_batch[b:b + 2000]
        )

    for o in range(0, len(orders_batch), 1000):
        cursor.executemany(
            "INSERT INTO orders (order_number, user_id, service_id, provider_id, order_time, service_time, amount, status, address, contact_name, contact_phone) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            orders_batch[o:o + 1000]
        )

    conn.commit()
    print(f"✅ 完成！生成了 {len(users_batch)} 个用户、{len(behaviors_batch)} 条行为及 {len(orders_batch)} 个订单记录。")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    generate_data()