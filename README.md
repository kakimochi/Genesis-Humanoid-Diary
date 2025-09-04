# Genesis Humanoid Diary

Genesisを使用したUnitree H1ヒューマノイドロボットの歩行強化学習プロジェクト

## 🎯 プロジェクト概要

このプロジェクトは、最新の物理シミュレーションエンジン「Genesis」を使用して、Unitree H1ヒューマノイドロボットの歩行動作を強化学習で習得させることを目的としています。

## 🚀 主な特徴

- **最新技術の活用**: Genesis 0.3.3による高精度物理シミュレーション
- **実機対応**: Unitree H1の実際のURDFモデルを使用
- **GPU加速**: CUDA対応による高速学習
- **完全な学習環境**: 訓練・評価・可視化まで一貫したワークフロー
- **PPO強化学習**: 最新の強化学習アルゴリズムによる歩行学習
- **リアルタイム可視化**: TensorBoardとGenesisビューアーによる学習進捗確認

## 📋 システム要件

- **OS**: Ubuntu 22.04 LTS (推奨)
- **Python**: 3.8以上
- **GPU**: NVIDIA GeForce RTX 2070 SUPER以上 (CUDA対応)
- **メモリ**: 16GB以上推奨
- **ストレージ**: 10GB以上の空き容量

## 🛠️ 環境構築

### クイックスタート

```bash
# リポジトリのクローン
git clone git@github.com:kakimochi/Genesis-Humanoid-Diary.git
cd Genesis-Humanoid-Diary

# 仮想環境の作成と有効化
python3 -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install genesis-world
pip install rsl-rl-lib==2.2.4 tensorboard

# サブモジュールの取得
git submodule update --init --recursive

# 環境チェック
python 00_check_env.py
```

### 詳細な手順

詳細な環境構築手順については、[SETUP_GUIDE.md](SETUP_GUIDE.md)を参照してください。

## 📁 プロジェクト構成

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
├── SETUP_GUIDE.md                  # 環境構築手順書
├── prompt.md                       # プロジェクト指示書
└── README.md                       # このファイル
```

## 🔧 環境チェック

環境が正しく構築されているかを確認するには：

```bash
source venv/bin/activate
python 00_check_env.py
```

成功時の出力：
```
Genesis Humanoid Diary 環境チェック
成功: 9/9
✓ 全てのチェックが成功しました！
環境構築が完了しています。
```

## 🤖 H1ヒューマノイド強化学習

### 基本的な使用方法

```bash
# 仮想環境の有効化
source venv/bin/activate

# 学習環境のテスト（2環境、2イテレーション）
python h1_train.py --exp_name test --num_envs 2 --max_iterations 2

# 本格的な学習の実行
python h1_train.py --exp_name h1-walking --num_envs 4096 --max_iterations 1000

# 学習結果の評価
python h1_eval.py --exp_name h1-walking

# TensorBoardでの学習進捗確認
tensorboard --logdir logs/
```

### 学習環境の仕様

- **観測空間**: 39次元（角速度、重力、コマンド、関節状態、前回アクション）
- **アクション空間**: 10次元（H1の脚関節制御）
- **報酬関数**: 速度追跡、姿勢維持、エネルギー効率を考慮した複合報酬
- **アルゴリズム**: PPO（Proximal Policy Optimization）

成功: 9/9
✓ 全てのチェックが成功しました！
環境構築が完了しています。
```

## 🎮 使用技術

- **[Genesis](https://genesis-world.readthedocs.io/)**: 高性能物理シミュレーションエンジン
- **[PyTorch](https://pytorch.org/)**: 深層学習フレームワーク
- **[Unitree H1](https://www.unitree.com/)**: ヒューマノイドロボットプラットフォーム
- **CUDA**: GPU加速コンピューティング

## 📊 開発ロードマップ

- [x] 環境構築
- [x] 基本シミュレーション環境の準備
- [ ] H1モデルの基本動作確認
- [ ] 強化学習環境の実装
- [ ] 歩行学習アルゴリズムの開発
- [ ] 学習結果の評価・可視化
- [ ] 実機での検証
### 利用可能なスクリプト

- `h1_env.py`: H1ヒューマノイド学習環境
- `h1_train.py`: PPO強化学習訓練スクリプト
- `h1_eval.py`: 訓練済みモデル評価スクリプト

## � 使用技術

- **[Genesis](https://genesis-world.readthedocs.io/)**: 高性能物理シミュレーションエンジン
- **[PyTorch](https://pytorch.org/)**: 深層学習フレームワーク
- **[Unitree H1](https://www.unitree.com/)**: ヒューマノイドロボットプラットフォーム
- **[rsl-rl-lib](https://github.com/leggedrobotics/rsl_rl)**: 強化学習ライブラリ
- **TensorBoard**: 学習進捗可視化
- **CUDA**: GPU加速コンピューティング

## � 開発ロードマップ

- [x] 環境構築とシステム要件確認
- [x] Genesis物理シミュレーション環境構築
- [x] Unitree H1モデル統合
- [x] H1専用強化学習環境実装
- [x] PPO学習アルゴリズム実装
- [x] 学習結果評価システム構築
- [x] TensorBoard可視化システム
- [ ] 長期学習による歩行性能向上
- [ ] 実機での検証準備
==================================================
成功: 9/9
✓ 全てのチェックが成功しました！
環境構築が完了しています。
```

## 🎮 使用技術

- **[Genesis](https://genesis-world.readthedocs.io/)**: 高性能物理シミュレーションエンジン
- **[PyTorch](https://pytorch.org/)**: 深層学習フレームワーク
- **[Unitree H1](https://www.unitree.com/)**: ヒューマノイドロボットプラットフォーム
- **CUDA**: GPU加速コンピューティング

## 📊 開発ロードマップ

- [x] 環境構築
- [x] 基本シミュレーション環境の準備
- [ ] H1モデルの基本動作確認
- [ ] 強化学習環境の実装
- [ ] 歩行学習アルゴリズムの開発
- [ ] 学習結果の評価・可視化
- [ ] 実機での検証

## 🐛 トラブルシューティング

よくある問題と解決策については、[SETUP_GUIDE.md](SETUP_GUIDE.md)のトラブルシューティングセクションを参照してください。

## � ログ

プロジェクトの進捗は `log_task/` ディレクトリで管理されています：
- [log_task_00.md](log_task/log_task_00.md): 環境構築ログ

## 🤝 コントリビューション

このプロジェクトへの貢献を歓迎します。以下の手順でコントリビュートできます：

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトのライセンス情報については、LICENSEファイルを参照してください。

## 🔗 関連リンク

- [Genesis Documentation](https://genesis-world.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Unitree Robotics](https://www.unitree.com/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

## � お問い合わせ

プロジェクトに関する質問や提案がある場合は、Issueを作成してください。

---

**注意**: このプロジェクトは研究・教育目的で開発されています。商用利用については、各ライブラリのライセンスを確認してください。
