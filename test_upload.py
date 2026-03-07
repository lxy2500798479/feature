#!/usr/bin/env python3
"""测试文档上传接口"""
import requests
import sys

# API 地址
API_URL = "http://localhost:8000/api/v1/documents/upload"

# 测试文件
TEST_FILE = "/Users/luoxingyao/repo/feature/mockData/西安蓝想BP1.pdf"

def main():
    print(f"测试上传文件: {TEST_FILE}")
    
    with open(TEST_FILE, "rb") as f:
        files = {"file": ("西安蓝想BP1.pdf", f, "application/pdf")}
        data = {"async_mode": "true"}
        
        response = requests.post(API_URL, files=files, data=data)
        
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
    
    if response.status_code == 200:
        print("✅ 上传成功!")
    else:
        print("❌ 上传失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
