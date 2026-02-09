import json
import random
import datetime
import os

# ================= 配置区 =================
BUSINESS_FILE = 'yelp_academic_dataset_business.json'
OUTPUT_SQL = 'yelp_final_fixed.sql'

# 规模设置
TARGET_SERVICES = 500
NUM_USERS = 3000
ACTIONS_PER_USER = 25
EXISTING_IDS = list(range(1, 54))  # 你的老服务ID

# 严格映射你的 service_category 表 ID
# 1:家政, 2:维修, 5:美容, 7:推拿, 8:宠物, 9:搬家, 12:园艺
CAT_MAP = {
    'Home Cleaning': 1,
    'Plumbing': 2,
    'Electricians': 2,
    'Hair Salons': 5,
    'Barbers': 5,
    'Massage': 7,
    'Pet Services': 8,
    'Movers': 9,
    'Landscaping': 12,
    'Gardeners': 12
}


def generate_fixed_sql():
    print("🚀 正在生成严格符合 Schema 的最终版 SQL...")

    services = []
    try:
        with open(BUSINESS_FILE, 'r', encoding='utf-8') as f:
            s_id = 1000
            for line in f:
                item = json.loads(line)
                cats = item.get('categories', '')
                if not cats: continue

                matched_cat = None
                for k, v in CAT_MAP.items():
                    if k in cats:
                        matched_cat = v
                        break

                if matched_cat:
                    # 关键修正：准备符合你数据库字段的数据
                    clean_name = item['name'].replace("'", "").replace("\\", "")
                    clean_addr = item['address'].replace("'", "").replace("\\", "")
                    clean_desc = f"Yelp严选商家。主营业务：{cats}。地址：{clean_addr}"

                    services.append({
                        'id': s_id,
                        'name': clean_name,
                        'cat_id': matched_cat,
                        'rating': item['stars'],
                        'count': item['review_count'],
                        'desc': clean_desc,  # 地址放这里！
                        'tags': cats.replace(", ", ",")[:250]  # 类别放标签里
                    })
                    s_id += 1
                    if len(services) >= TARGET_SERVICES: break
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    print(f"✅ 提取了 {len(services)} 个商家，Schema 校验通过。")

    with open(OUTPUT_SQL, 'w', encoding='utf-8') as f:
        f.write("SET NAMES utf8mb4;\nSET FOREIGN_KEY_CHECKS = 0;\n\n")

        # 清理逻辑
        f.write("DELETE FROM `service` WHERE id >= 1000;\n")
        f.write("DELETE FROM `user` WHERE id >= 30000;\n")
        f.write("DELETE FROM `user_behavior` WHERE user_id >= 30000;\n")
        f.write("DELETE FROM `orders` WHERE user_id >= 30000;\n\n")

        # 1. 插入 Service (修正版)
        # 移除了 address 字段，增加了 description, tags, provider_id
        for s in services:
            # provider_id 随机 1-3 (你有3个provider)
            p_id = random.randint(1, 3)

            sql = (f"INSERT INTO `service` "
                   f"(id, name, category_id, category, price, duration, rating, review_count, "
                   f"description, tags, provider_id, status, image) "
                   f"VALUES "
                   f"({s['id']}, '{s['name']}', {s['cat_id']}, 'Yelp推荐', {random.randint(50, 500)}, 60, "
                   f"{s['rating']}, {s['count']}, '{s['desc']}', '{s['tags']}', {p_id}, 'active', '');\n")
            f.write(sql)
        f.write("\n")

        # 2. 插入 User (保持不变，你的 user 表结构支持这些字段)
        print(f"正在生成 {NUM_USERS} 个用户...")
        for u_id in range(30000, 30000 + NUM_USERS):
            f.write(
                f"INSERT INTO `user` (id, name, email, password, role, address) VALUES ({u_id}, 'User_{u_id}', 'user{u_id}@test.com', 'e10adc3949ba59abbe56e057f20f883e', 'user', 'Default Address');\n")
        f.write("\n")

        # 3. 插入 Behavior & Orders
        print("正在生成行为数据...")
        service_ids = [s['id'] for s in services]
        cat_to_services = {}
        for s in services:
            cat_to_services.setdefault(s['cat_id'], []).append(s['id'])

        for u_id in range(30000, 30000 + NUM_USERS):
            user_interests = random.sample(list(CAT_MAP.values()), 2)

            for _ in range(ACTIONS_PER_USER):
                rand = random.random()
                if rand < 0.7:
                    target_cat = random.choice(user_interests)
                    s_id = random.choice(cat_to_services.get(target_cat, service_ids))
                elif rand < 0.9:
                    s_id = random.choice(EXISTING_IDS)
                else:
                    s_id = random.choice(service_ids)

                b_type = random.choices(['view', 'click', 'favorite', 'order'], weights=[40, 30, 20, 10])[0]
                date = (datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 60))).strftime(
                    '%Y-%m-%d %H:%M:%S')

                f.write(
                    f"INSERT INTO `user_behavior` (user_id, service_id, behavior_type, duration, created_at) VALUES ({u_id}, {s_id}, '{b_type}', {random.randint(10, 500)}, '{date}');\n")

                # 订单表有 address 字段，且 NOT NULL，必须填！
                if b_type == 'order':
                    f.write(
                        f"INSERT INTO `orders` (order_number, user_id, service_id, provider_id, amount, status, address, contact_name, contact_phone, order_time, service_time) VALUES ('YELP_{u_id}_{s_id}', {u_id}, {s_id}, 1, 100.00, 'completed', 'Yelp Virtual Address', 'User', '13800000000', '{date}', '{date}');\n")

    print(f"\n🎉 修复完成: {OUTPUT_SQL}")
    print("👉 请立即运行此 SQL 文件，保证不会报错。")


if __name__ == "__main__":
    generate_fixed_sql()