import mysql.connector
from mysql.connector import Error

# ================= 数据库配置 =================
# 请确保这里的配置与你的 clmm2w/recommend_new/config.py 一致
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "ezserve_db_not_sport",
    "port": 3306
}

# ================= 模拟数据定义的范围 =================
# 必须与 augment_data.py 中的设定严格一致
NEW_SERVICE_START_ID = 1000
NEW_USER_START_ID = 30000


def cleanup():
    conn = None
    try:
        # 1. 连接数据库
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            cursor = conn.cursor()
            print(f"成功连接到数据库: {DB_CONFIG['database']}")

            # 关闭外键检查以防止删除报错（虽然你的表没设物理外键，但这是个好习惯）
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")

            # 2. 执行删除操作

            # 删除模拟订单（识别订单号前缀 'SIM_' 或者 ID 范围）
            print("正在清理模拟订单...")
            cursor.execute(f"""
                DELETE FROM `orders` 
                WHERE user_id >= {NEW_USER_START_ID} 
                OR service_id >= {NEW_SERVICE_START_ID}
                OR order_number LIKE 'SIM_%';
            """)
            print(f"删除了 {cursor.rowcount} 条模拟订单。")

            # 删除模拟行为记录
            print("正在清理模拟用户行为...")
            cursor.execute(f"""
                DELETE FROM `user_behavior` 
                WHERE user_id >= {NEW_USER_START_ID} 
                OR service_id >= {NEW_SERVICE_START_ID};
            """)
            print(f"删除了 {cursor.rowcount} 条行为记录。")

            # 删除模拟推荐日志
            print("正在清理模拟推荐日志...")
            cursor.execute(f"""
                DELETE FROM `recommendation_log` 
                WHERE user_id >= {NEW_USER_START_ID} 
                OR service_id >= {NEW_SERVICE_START_ID};
            """)
            print(f"删除了 {cursor.rowcount} 条推荐日志。")

            # 删除模拟用户
            print("正在清理模拟用户...")
            cursor.execute(f"DELETE FROM `user` WHERE id >= {NEW_USER_START_ID};")
            print(f"删除了 {cursor.rowcount} 个模拟用户。")

            # 删除模拟服务（无图服务）
            print("正在清理模拟服务...")
            cursor.execute(f"DELETE FROM `service` WHERE id >= {NEW_SERVICE_START_ID};")
            print(f"删除了 {cursor.rowcount} 个模拟服务。")

            # 恢复外键检查
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")

            # 3. 提交更改
            conn.commit()
            print("\n✅ 数据清理完成！你的原始数据（Service 1-53 等）已恢复到注入前的状态。")

    except Error as e:
        print(f"清理过程中发生错误: {e}")
        if conn:
            conn.rollback()
            print("由于发生错误，已回滚所有操作。")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            print("数据库连接已关闭。")


if __name__ == "__main__":
    confirm = input("⚠️ 该操作将永久删除 ID 1000+ 的服务和 ID 30000+ 的用户及其关联数据。确定继续吗？(y/n): ")
    if confirm.lower() == 'y':
        cleanup()
    else:
        print("操作已取消。")