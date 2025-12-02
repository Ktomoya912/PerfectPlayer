# Mini 2048 (3x3) 教師あり学習プロジェクト

3x3版の2048ゲームに対する深層学習ベースのAIエージェント。
完全なゲーム状態データベース（`db2.out`）を用いた教師あり学習により、最適な手を予測するニューラルネットワークを訓練します。

## プロジェクト概要

- **ゲーム**: Mini 2048 (3x3グリッド)
- **学習方法**: 教師あり学習（回帰）
- **データセット**: 31,431,374個の完全なゲーム状態と評価値
- **モデル**: CNN + 全結合層による評価値予測
- **正規化**: 8方向（回転・鏡面）の盤面を正規化して学習

## プロジェクト構成

```
.
├── src/
│   ├── agent.py          # ゲームエージェント（行動選択ロジック）
│   ├── config.py         # 設定管理
│   ├── dataset.py        # データセットクラス（遅延読み込み）
│   ├── game_2048_3_3.py  # ゲームロジック
│   ├── logger.py         # ロギング
│   ├── model.py          # ニューラルネットワークモデル
│   ├── play.py           # モデルを使ってゲームをプレイ
│   ├── train.py          # 学習スクリプト
│   └── utils.py          # ユーティリティ関数
├── db2.out               # ゲーム状態データベース（バイナリ）
├── checkpoints/          # 学習済みモデル
├── logs/                 # 学習ログ・評価ログ
└── results/              # プレイ結果統計
```

## セットアップ

### 必要要件

- Python 3.12以上
- [uv](https://github.com/astral-sh/uv) (Pythonパッケージマネージャー)
- CUDA対応GPU（推奨、CPUでも動作可能）

### インストール

1. **uvのインストール**（未インストールの場合）
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **依存関係のインストール**
   ```bash
   uv sync
   ```

これにより、`pyproject.toml`に記載された依存関係（PyTorch、NumPyなど）が自動的にインストールされます。

## 使い方

### 1. モデルの学習

```bash
uv run src/train.py
```

**学習設定のカスタマイズ**:
`src/config.py`を編集してハイパーパラメータを調整できます。

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 1024
    epochs: int = 10
    device: str = "cuda"  # "cuda" or "cpu"
```

**学習の進行状況**:
- 100バッチごとに損失値をログ出力
- 1000バッチごとに10ゲームプレイして平均スコアを評価
- エポックごとにチェックポイントを`checkpoints/`に保存
- ログは`logs/metrics_*.csv`と`logs/evaluation_*.csv`に記録

### 2. 学習済みモデルでゲームをプレイ

**1ゲームをプレイ（詳細表示）**:
```bash
uv run src/play.py --model checkpoints/model_epoch_3.pth --single
```

**100ゲームをプレイして統計を取得**:
```bash
uv run src/play.py --model checkpoints/model_epoch_3.pth --num-games 100
```

統計は`results/`ディレクトリに保存されます。

## 技術詳細

### データベース（db2.out）

- **形式**: バイナリファイル
- **構造**:
  - ゲーム状態数（int, 4バイト）
  - 状態ID配列（int配列）
  - 評価値配列（double配列）
- **正規化**: 各ゲーム状態は8方向（回転・鏡面変換）のうち最小のインデックスに正規化

### モデルアーキテクチャ

```
入力: 11x3x3 one-hotエンコーディング（11種類のタイル値）
  ↓
Conv2d(11→128, kernel=1x1) + BatchNorm + ReLU
  ↓
Conv2d(128→256, kernel=2x2) + BatchNorm + ReLU
  ↓
Conv2d(256→512, kernel=2x2) + BatchNorm + ReLU
  ↓
Flatten → FC(512→256) + BatchNorm + ReLU + Dropout(0.3)
  ↓
FC(256→1) + Sigmoid
  ↓
出力: 正規化されたスコア（0-1）
```

### 盤面の正規化

ゲームの対称性を利用して、8方向の回転・鏡面変換のうち最小のインデックスに正規化:
- 元の向き
- 左右反転
- 90度回転
- 270度回転
- 180度回転
- 180度回転 + 左右反転
- 90度回転 + 左右反転
- 270度回転 + 左右反転

これにより、データセットのサイズを削減し、学習効率を向上させています。

### 損失関数

- **MSE Loss**: 正規化されたスコア（0-1範囲）に対する平均二乗誤差

### 最適化

- **メモリ効率**: 遅延読み込み（lazy loading）で大規模データセットに対応
- **高速化**: NumPyのベクトル化演算を活用
- **安定化**: BatchNormalizationとDropoutで過学習を防止

## パフォーマンス

学習の進捗は以下で確認できます:
- `logs/metrics_*.csv`: バッチごとの損失値
- `logs/evaluation_*.csv`: 1000バッチごとの評価スコア

評価スコアの例（epoch 2時点）:
- 平均スコア: 600-1000
- 最大スコア: 2000-3000
- 最小スコア: 100-500

## トラブルシューティング

### CUDA out of memory

バッチサイズを減らしてください:
```python
# src/config.py
batch_size: int = 512  # デフォルト: 1024
```

### 学習が遅い

- GPUが利用可能か確認: `torch.cuda.is_available()`
- `num_workers=0`（マルチプロセス無効）でメモリリーク防止

## ライセンス

このプロジェクトは研究目的で作成されています。

## 参考

- ゲームロジック: `src/game_2048_3_3.py`
- データベース読み込み: `src/utils.py`の`readDB2()`関数
