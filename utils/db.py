import mysql.connector
from mysql.connector import Error
from recommend.config import DB_CONFIG, logger

def get_db_connection():
    """创建数据库连接"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        logger.error(f"数据库连接错误: {e}")
        return None

def execute_query(query, params=None, fetch=True):
    """执行SQL查询"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return None
            
        cursor = conn.cursor(dictionary=True)
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        if fetch:
            result = cursor.fetchall()
            return result
        else:
            conn.commit()
            return cursor.rowcount
    except Error as e:
        logger.error(f"SQL执行错误: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            conn.close() 