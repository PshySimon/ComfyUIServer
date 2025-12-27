#!/usr/bin/env python3
"""
测试模型搜索功能
"""

import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from scripts.model_downloader import ModelDownloader, ModelInfo


def test_search_rife47():
    """测试搜索 rife47.pth 模型"""
    print("=" * 60)
    print("测试: 搜索 rife47.pth")
    print("=" * 60)
    
    downloader = ModelDownloader(
        comfyui_dir=Path("./ComfyUI"),
        workflow_file=None
    )
    
    # 测试 Brave Search 搜索
    print("\n1. 测试 Brave Search 搜索...")
    result = downloader.search_web_for_huggingface("rife47.pth")
    
    if result:
        print(f"   ✓ 找到模型!")
        print(f"   名称: {result.name}")
        print(f"   URL: {result.url}")
        print(f"   目录: {result.directory}")
        
        # 验证 URL 是否有效
        print("\n2. 验证下载链接...")
        import urllib.request
        try:
            req = urllib.request.Request(
                result.url, 
                method='HEAD',
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                size = response.headers.get('Content-Length')
                if size:
                    size_mb = int(size) / (1024 * 1024)
                    print(f"   ✓ URL 有效! 文件大小: {size_mb:.1f} MB")
                else:
                    print(f"   ✓ URL 有效! (大小未知)")
        except Exception as e:
            print(f"   ✗ URL 验证失败: {e}")
            return False
        
        return True
    else:
        print("   ✗ 未找到模型")
        return False


def test_search_common_models():
    """测试搜索几个常见模型"""
    print("\n" + "=" * 60)
    print("测试: 搜索常见模型")
    print("=" * 60)
    
    downloader = ModelDownloader(
        comfyui_dir=Path("./ComfyUI"),
        workflow_file=None
    )
    
    # 测试几个常见模型
    test_models = [
        "rife47.pth",
        # 可以添加更多测试模型
    ]
    
    results = {}
    for model_name in test_models:
        print(f"\n搜索: {model_name}")
        result = downloader.find_model_url(model_name)
        if result:
            print(f"  ✓ 找到: {result.url[:60]}...")
            results[model_name] = True
        else:
            print(f"  ✗ 未找到")
            results[model_name] = False
    
    # 统计
    found = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n结果: {found}/{total} 个模型找到")
    
    return found == total


if __name__ == "__main__":
    print("ComfyUI 模型搜索测试")
    print("=" * 60)
    
    # 运行测试
    success = True
    
    if not test_search_rife47():
        success = False
    
    # 可选：测试更多模型
    # if not test_search_common_models():
    #     success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过!")
        sys.exit(0)
    else:
        print("✗ 部分测试失败")
        sys.exit(1)
