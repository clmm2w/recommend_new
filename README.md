# ezServe推荐系统

这是ezServe平台的推荐系统服务，基于Python FastAPI实现。

## 项目结构

```
recommend/
├── app.py            # 主应用入口
├── config.py         # 配置信息
├── requirements.txt  # 项目依赖
├── start.py          # 启动脚本
├── models/
│   ├── __init__.py
│   └── data_cache.py # 数据缓存类
├── services/
│   ├── __init__.py
│   ├── collaborative_filtering.py # 协同过滤算法
│   ├── content_based.py          # 基于内容的推荐
│   ├── matrix_factorization.py   # 矩阵分解模型
│   ├── neural_cf.py              # 神经网络协同过滤
│   ├── diversity.py              # 多样性处理
│   ├── recommendation.py         # 核心推荐逻辑
│   └── ab_testing.py            # A/B测试服务
├── routers/
│   ├── __init__.py
│   ├── recommend.py  # 推荐相关API
│   ├── feedback.py   # 反馈相关API
│   └── debug.py      # 调试相关API
└── utils/
    ├── __init__.py
    ├── db.py         # 数据库操作
    ├── logging.py    # 日志配置
    └── geo.py        # 地理位置计算
```

## 功能

- 用户个性化推荐：基于用户历史行为推荐服务
- 相似服务推荐：为特定服务找出相似的其他服务
- 热门服务推荐：基于评分、评价数量等指标推荐热门服务
- A/B测试功能：测试不同推荐算法和多样性参数的效果

## 技术实现

- 协同过滤：基于用户行为相似性推荐
- 内容推荐：基于服务特征相似性推荐 
- 矩阵分解：使用SVD算法进行推荐
- 神经网络：使用TensorFlow实现神经协同过滤(可选)
- 多样性处理：避免推荐结果过于单一
- 实时反馈：记录用户对推荐的反馈
- 时间感知：根据用户的时间模式调整推荐

## 安装与运行

### 环境要求

- Python 3.8+
- MySQL数据库

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

可以通过环境变量配置数据库连接信息，或者直接修改`config.py`文件：

```bash
export DB_HOST=localhost
export DB_USER=root
export DB_PASSWORD=123456
export DB_NAME=ezserve_db
export DB_PORT=3306
```

### 运行服务

```bash
python start.py
```

或者使用uvicorn直接运行：

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API接口

### 用户推荐

```
GET /recommend?user_id={user_id}&limit={limit}
```

### 相似服务推荐

```
GET /recommend/similar?service_id={service_id}&limit={limit}
```

### 热门服务推荐

```
GET /recommend/trending?limit={limit}
```

### 触发模型训练

```
POST /debug/train
```

## Swagger文档

启动服务后，访问 http://localhost:8000/docs 可以查看API文档。 

## 调试工具

系统提供了多种调试工具，可以通过`/debug`路径访问：

- `/debug/model-status`: 查看模型状态
- `/debug/recommendation-logs`: 查看推荐日志
- `/debug/user/{user_id}/behaviors`: 查看用户行为数据
- `/debug/service/{service_id}/similarity`: 查看服务相似度
- `/debug/ab-test-status`: 查看A/B测试状态
- `/debug/explain-recommendation`: 获取推荐解释 