#!/usr/bin/env python3
"""
测试缓存清理机制的脚本
"""

import requests
import time
import json

def test_cache_cleanup():
    """测试缓存清理功能"""
    base_url = "http://localhost:5000"
    
    print("=== 测试缓存清理机制 ===")
    
    # 1. 获取初始内存统计
    print("\n1. 获取初始内存统计...")
    try:
        response = requests.get(f"{base_url}/memory_stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"初始状态: {json.dumps(stats, indent=2, ensure_ascii=False)}")
        else:
            print(f"获取内存统计失败: {response.status_code}")
            return
    except Exception as e:
        print(f"连接失败: {e}")
        return
    
    # 2. 测试清理缓存
    print("\n2. 测试清理缓存...")
    try:
        response = requests.post(f"{base_url}/cleanup_cache")
        if response.status_code == 200:
            result = response.json()
            print(f"清理结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"清理缓存失败: {response.status_code}")
            print(f"错误信息: {response.text}")
    except Exception as e:
        print(f"清理缓存请求失败: {e}")
    
    # 3. 再次获取内存统计
    print("\n3. 获取清理后的内存统计...")
    try:
        response = requests.get(f"{base_url}/memory_stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"清理后状态: {json.dumps(stats, indent=2, ensure_ascii=False)}")
        else:
            print(f"获取内存统计失败: {response.status_code}")
    except Exception as e:
        print(f"获取内存统计失败: {e}")
    
    print("\n=== 测试完成 ===")

def test_health_check():
    """测试健康检查"""
    base_url = "http://localhost:5000"
    
    print("=== 测试健康检查 ===")
    try:
        response = requests.get(f"{base_url}/healthy")
        if response.status_code == 200:
            print(f"健康检查通过: {response.text}")
        else:
            print(f"健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"健康检查请求失败: {e}")

if __name__ == "__main__":
    # 先测试健康检查
    test_health_check()
    
    # 等待一下
    time.sleep(1)
    
    # 测试缓存清理
    test_cache_cleanup()
