"""
Mini 2048 学習設定ファイル
"""

from dataclasses import dataclass


@dataclass
class DataConfig:
    """データセット関連の設定"""

    # データベースファイル
    db_path: str = "db2.out"


@dataclass
class TrainingConfig:
    """学習関連の設定"""

    # 学習パラメータ
    learning_rate: float = 1e-2
    batch_size: int = 2048
    epochs: int = 20

    # デバイス設定
    device: str = "cuda"  # "cuda" or "cpu"

    # チェックポイント設定
    save_interval: int = 1  # エポックごとに保存
    checkpoint_dir: str = "checkpoints"

    # ログ設定
    log_interval: int = 100  # バッチごとにログ出力


@dataclass
class Config:
    """全体の設定"""

    data: DataConfig
    training: TrainingConfig

    @classmethod
    def default(cls):
        """デフォルト設定を返す"""
        return cls(data=DataConfig(), training=TrainingConfig())

    def __repr__(self):
        lines = ["Configuration:"]
        lines.append("  Data:")
        for key, value in self.data.__dict__.items():
            lines.append(f"    {key}: {value}")
        lines.append("  Training:")
        for key, value in self.training.__dict__.items():
            lines.append(f"    {key}: {value}")
        return "\n".join(lines)
