import uvicorn

if __name__ == "__main__":
    print("启动ezServe推荐系统...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 