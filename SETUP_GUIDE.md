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

### 7. 強化学習ライブラリのインストール
```bash
pip install rsl-rl-lib==2.2.4 tensorboard
```

### 8. 環境チェックの実行
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

## H1ヒューマノイド強化学習の使用方法

環境構築が完了したら、以下の強化学習機能を使用できます：

### 1. 学習環境のテスト
```bash
# 学習環境の動作確認（2環境、2イテレーション）
python h1_train.py --exp_name test --num_envs 2 --max_iterations 2
```

### 2. 本格的な学習の実行
```bash
# ビューアー表示ありで学習実行
python h1_train.py --exp_name h1-walking-full --num_envs 4096 --max_iterations 1000

# ビューアー表示なしで学習実行（高速）
python h1_train.py --exp_name h1-walking-fast --num_envs 4096 --max_iterations 1000 --no_viewer
```

### 3. 学習結果の評価
```bash
# 訓練済みモデルの評価（ビューアー表示あり）
python h1_eval.py --exp_name h1-walking-full

# ビューアー表示なしで評価
python h1_eval.py --exp_name h1-walking-full --no_viewer
```

### 4. TensorBoardでの学習進捗確認
```bash
# TensorBoardの起動
tensorboard --logdir logs/

# ブラウザで http://localhost:6006 にアクセス
```

## 学習環境の仕様

- **観測空間**: 39次元
  - 角速度 (3次元)
  - 重力ベクトル (3次元)
  - コマンド (3次元)
  - 関節位置 (10次元)
  - 関節速度 (10次元)
  - 前回のアクション (10次元)

- **アクション空間**: 10次元（H1の脚関節）
  - 左脚: hip_yaw, hip_roll, hip_pitch, knee, ankle
  - 右脚: hip_yaw, hip_roll, hip_pitch, knee, ankle

- **報酬関数**:
  - 線形速度追跡
  - 角速度追跡
  - 垂直速度ペナルティ
  - アクション変化率ペナルティ
  - デフォルト姿勢維持
  - 基準高度維持
  - 直立姿勢維持
  - 関節制限ペナルティ

## 次のステップ

環境構築が完了したら、以下の開発を開始できます：

1. **基本シミュレーション**: `python h1_train.py --exp_name test --num_envs 2 --max_iterations 2`
2. **強化学習訓練**: `python h1_train.py --exp_name your-experiment`
3. **学習結果評価**: `python h1_eval.py --exp_name your-experiment`
4. **進捗可視化**: `tensorboard --logdir logs/`

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
