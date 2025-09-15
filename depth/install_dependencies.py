#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度视频转换程序 - 依赖安装脚本
"""

import subprocess
import sys
import os


def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package} 安装失败")
        return False


def check_package(package):
    """检查包是否已安装"""
    try:
        __import__(package)
        print(f"✅ {package} 已安装")
        return True
    except ImportError:
        print(f"❌ {package} 未安装")
        return False


def main():
    """主函数"""
    print("深度视频转换程序 - 依赖安装")
    print("=" * 50)
    
    # 必需的包
    required_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"), 
        ("opencv-python", "cv2"),
        ("transformers", "transformers"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy"),
    ]
    
    print("检查依赖包...")
    missing_packages = []
    
    for package_name, import_name in required_packages:
        if not check_package(import_name):
            missing_packages.append(package_name)
    
    if not missing_packages:
        print("\n🎉 所有依赖包都已安装!")
        return
    
    print(f"\n需要安装的包: {', '.join(missing_packages)}")
    
    # 安装缺失的包
    print("\n开始安装...")
    failed_packages = []
    
    for package in missing_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print("\n" + "=" * 50)
    
    if failed_packages:
        print(f"❌ 以下包安装失败: {', '.join(failed_packages)}")
        print("\n请手动安装:")
        for package in failed_packages:
            print(f"pip install {package}")
    else:
        print("🎉 所有依赖包安装完成!")
        print("\n现在可以运行深度视频转换程序了:")
        print("python depth_video_converter.py input.mp4 output.mp4")


if __name__ == "__main__":
    main()
