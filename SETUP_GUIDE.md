# Genesis Humanoid Diary 環境構築手順書

## 概要
このドキュメントは、Genesisを使用したUnitree H1ヒューマノイドモデルの歩行強化学習環境を構築するための詳細な手順書です。

## 前提条件
- Ubuntu 22.04 LTS (推奨)
- Python 3.8以上
- NVIDIA GPU (CUDA対応)
- Git

## 必要なシステム要件
- GPU: NVIDIA GeForce RTX 2070 SUPER以上 (CUDA対応)
- メモリ: 16GB以上推奨
- ストレージ: 10GB以上の空き容量

## 環境構築手順

### 1. プロジェクトのクローン
```bash
git clone git@github.com:kakimochi/Genesis-Humanoid-Diary.git
cd Genesis-Humanoid-Diary
```

### 2. Python仮想環境の作成
```bash
# 仮想環境の作成
python3 -m venv venv

# 仮想環境の有効化
source venv/bin/activate
```

### 3. pipのアップグレード
```bash
pip install --upgrade pip
```

### 4. PyTorch GPU版のインストール
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**注意**: CUDA 11.8対応版をインストールしています。お使いのGPUに応じてCUDAバージョンを調整してください。

### 5. Genesisのインストール
```bash
pip install genesis-world
```

**注意**: インストール中に依存関係の警告が表示される場合がありますが、仮想環境内での問題のため、基本的な動作には影響ありません。

### 6. サブモジュールの取得
```bash
git submodule update --init --recursive
```

**注意**: 以下のリポジトリがgit submoduleとして管理されています：
- `unitree_ros`: Unitree H1ロボットモデル
- `genesis-samples`: Genesis公式サンプルコード

初回クローン時は上記コマンドでsubmoduleを初期化してください。

### 7. 環境チェックの実行
```bash
python 00_check_env.py
```

## 環境チェック項目

環境チェックスクリプト(`00_check_env.py`)は以下の項目を確認します：

1. **仮想環境チェック**: 仮想環境内で実行されているか
2. **Pythonバージョン**: Python 3.8以上
3. **基本パッケージ**: numpy, torch, torchvision, torchaudio
4. **PyTorch GPU対応**: CUDA利用可能性とGPU情報
5. **Genesis**: インポートと初期化の成功
6. **H1 URDFファイル**: Unitree H1モデルファイルの存在

## 成功時の出力例

```
Genesis Humanoid Diary 環境チェック
==================================================

=== 仮想環境チェック ===
✓ 仮想環境内で実行されています
  Python実行パス: /path/to/Genesis-Humanoid-Diary/venv/bin/python3

=== 基本パッケージチェック ===
=== Python バージョンチェック ===
Python 3.10.12
✓ Python バージョン OK
✓ numpy: 2.1.2
✓ torch: 2.7.1+cu118
✓ torchvision: 0.22.1+cu118
✓ torchaudio: 2.7.1+cu118

=== PyTorch GPU チェック ===
PyTorch バージョン: 2.7.1+cu118
✓ CUDA 利用可能
  GPU数: 1
  GPU 0: NVIDIA GeForce RTX 2070 SUPER

=== Genesis チェック ===
✓ Genesis インポート成功
✓ Genesis 初期化成功

=== H1 URDF ファイルチェック ===
✓ H1 URDF ファイル発見: unitree_ros/robots/h1_description/urdf/h1.urdf
  ファイルサイズ: 22594 bytes
✓ URDFファイルは空ではありません

==================================================
チェック結果サマリー
==================================================
成功: 9/9
✓ 全てのチェックが成功しました！
環境構築が完了しています。
```

## トラブルシューティング

### 1. CUDA関連のエラー
- **症状**: PyTorchでCUDAが利用できない
- **解決策**: 
  - NVIDIA ドライバーの更新
  - CUDA Toolkitの再インストール
  - 適切なPyTorchバージョンの選択

### 2. 仮想環境の問題
- **症状**: 仮想環境外で実行されている警告
- **解決策**: 
  ```bash
  source venv/bin/activate
  ```

### 3. Genesis初期化エラー
- **症状**: Genesis初期化時のエラー
- **解決策**: 
  - GPU メモリの確認
  - 他のGPUプロセスの終了
  - システムの再起動

### 4. 依存関係の警告
- **症状**: pip install時の依存関係警告
- **解決策**: 
  - 仮想環境内での問題のため、基本的には無視可能
  - 必要に応じて個別パッケージの再インストール

## ファイル構成

環境構築完了後のプロジェクト構成：

```
Genesis-Humanoid-Diary/
├── venv/                           # Python仮想環境
├── unitree_ros/                    # Unitree ロボットモデル
│   └── robots/
│       └── h1_description/
│           └── urdf/
│               └── h1.urdf         # H1ロボットモデル
├── log_task/                       # 作業ログ
│   └── log_task_00.md
├── 00_check_env.py                 # 環境チェックスクリプト
├── SETUP_GUIDE.md                  # この手順書
├── prompt.md                       # プロジェクト指示書
└── README.md                       # プロジェクト概要
```

## 次のステップ

環境構築が完了したら、以下の開発を開始できます：

1. **基本シミュレーション**: Genesisを使用したH1モデルの基本動作確認
2. **強化学習環境**: 歩行学習のための環境設定
3. **学習アルゴリズム**: PPO等の強化学習アルゴリズムの実装
4. **評価・可視化**: 学習結果の評価と可視化

## 参考リンク

- [Genesis Documentation](https://genesis-world.readthedocs.io/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Unitree Robotics](https://www.unitree.com/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

## 更新履歴

- 2025-09-02: 初版作成
  - 基本環境構築手順の記載
  - 環境チェックスクリプトの説明
  - トラブルシューティング情報の追加
