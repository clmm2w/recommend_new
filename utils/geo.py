# 尝试导入geopy
try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    
def calculate_distance(coord1, coord2):
    """
    计算两点之间的距离
    
    Args:
        coord1: 第一个坐标点(latitude, longitude)
        coord2: 第二个坐标点(latitude, longitude)
        
    Returns:
        距离(公里)
    """
    if GEOPY_AVAILABLE:
        return geodesic(coord1, coord2).kilometers
    else:
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        # 简化的距离计算（公里）
        return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111.32 