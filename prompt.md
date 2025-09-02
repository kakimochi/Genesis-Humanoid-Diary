# prompt

## コミットメッセージ
- 英文で簡潔にしてください
- コミットするファイルは私が git add します

## 作業ログ
- ファイル名は log_task_00.md のようにしてください
- 保存先は ./log_task/ にしてください
- 既存ファイルに追記してください
- 書式は下記の通りにしてください

```markdown
# 2025-09-02 17:49:38 <タイトル>
- <内容>
```

## 目的
- Genesisを使って、Unitree H1 ヒューマノイドモデルの歩行の強化学習を行います

## 環境構築
- venvを使います
  - 仮想環境名は "venv"
- PyTorchのGPU版をインストールします
- Genesisをインストールします
- Unitree H1 ヒューマノイドモデルを使用します
  - https://github.com/unitreerobotics/unitree_ros.git
  - 上記の h1.urdf を使用します
- 環境構築のチェックを行います
  - 00_check_env.py スクリプトを作成する
    - 基本パッケージ確認
    - Genesis初期化テスト
