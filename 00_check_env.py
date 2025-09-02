#!/usr/bin/env python3
"""
環境構築チェックスクリプト
Genesis Humanoid Diary プロジェクト用
"""

import sys
import subprocess
import importlib.util
import os

def check_python_version():
    """Pythonバージョンをチェック"""
    print("=== Python バージョンチェック ===")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python バージョン OK")
        return True
    else:
        print("✗ Python 3.8以上が必要です")
        return False

def check_package(package_name, import_name=None):
    """パッケージのインストール状況をチェック"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'バージョン不明')
            print(f"✓ {package_name}: {version}")
            return True
        else:
            print(f"✗ {package_name}: インストールされていません")
            return False
    except Exception as e:
        print(f"✗ {package_name}: エラー - {e}")
        return False

def check_torch_gpu():
    """PyTorchのGPU対応をチェック"""
    print("\n=== PyTorch GPU チェック ===")
    try:
        import torch
        print(f"PyTorch バージョン: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA 利用可能")
            print(f"  GPU数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("✗ CUDA 利用不可")
            return False
    except ImportError:
        print("✗ PyTorch がインストールされていません")
        return False

def check_genesis():
    """Genesisの基本チェック"""
    print("\n=== Genesis チェック ===")
    try:
        import genesis as gs
        print(f"✓ Genesis インポート成功")
        
        # 基本的な初期化テスト
        gs.init()
        print("✓ Genesis 初期化成功")
        return True
    except ImportError:
        print("✗ Genesis がインストールされていません")
        return False
    except Exception as e:
        print(f"✗ Genesis 初期化エラー: {e}")
        return False

def check_h1_urdf():
    """H1 URDFファイルの存在チェック"""
    print("\n=== H1 URDF ファイルチェック ===")
    urdf_path = "unitree_ros/robots/h1_description/urdf/h1.urdf"
    
    if os.path.exists(urdf_path):
        print(f"✓ H1 URDF ファイル発見: {urdf_path}")
        
        # ファイルサイズチェック
        file_size = os.path.getsize(urdf_path)
        print(f"  ファイルサイズ: {file_size} bytes")
        
        if file_size > 0:
            print("✓ URDFファイルは空ではありません")
            return True
        else:
            print("✗ URDFファイルが空です")
            return False
    else:
        print(f"✗ H1 URDF ファイルが見つかりません: {urdf_path}")
        return False

def check_venv():
    """仮想環境のチェック"""
    print("\n=== 仮想環境チェック ===")
    
    # 仮想環境内かどうかチェック
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ 仮想環境内で実行されています")
        print(f"  Python実行パス: {sys.executable}")
        return True
    else:
        print("✗ 仮想環境外で実行されています")
        print("  venvを有効化してから実行してください")
        return False

def main():
    """メイン関数"""
    print("Genesis Humanoid Diary 環境チェック")
    print("=" * 50)
    
    results = []
    
    # 仮想環境チェック
    results.append(check_venv())
    
    # 基本パッケージチェック
    print("\n=== 基本パッケージチェック ===")
    results.append(check_python_version())
    results.append(check_package("numpy"))
    results.append(check_package("torch"))
    results.append(check_package("torchvision"))
    results.append(check_package("torchaudio"))
    
    # PyTorch GPU チェック
    results.append(check_torch_gpu())
    
    # Genesis チェック
    results.append(check_genesis())
    
    # H1 URDF チェック
    results.append(check_h1_urdf())
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("チェック結果サマリー")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"成功: {passed}/{total}")
    
    if passed == total:
        print("✓ 全てのチェックが成功しました！")
        print("環境構築が完了しています。")
        print("\n次のステップ:")
        print("- Genesis を使用したヒューマノイドシミュレーションの開発を開始できます")
        print("- H1ロボットモデルを使用した強化学習環境の構築が可能です")
    else:
        print("✗ いくつかの問題があります。")
        print("上記のエラーメッセージを確認してください。")
        
        if not results[0]:  # 仮想環境チェックが失敗
            print("\n推奨アクション:")
            print("1. 仮想環境を有効化: source venv/bin/activate")
            print("2. 再度このスクリプトを実行")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
