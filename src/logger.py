"""
ロギング機能
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class Logger:
    """学習ログを管理するクラス"""

    def __init__(self, log_dir="logs", name="mini2048"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # ログファイル名（タイムスタンプ付き）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{timestamp}.log"

        # ロガーの設定
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # フォーマッター
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # ファイルハンドラ
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # コンソールハンドラ
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # ハンドラを追加
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Logging to {log_file}")

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)


class MetricsLogger:
    """学習メトリクスを記録するクラス"""

    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.csv"
        self.eval_file = self.log_dir / f"evaluation_{timestamp}.csv"

        # メトリクスCSVヘッダーを書き込み
        with open(self.metrics_file, "w") as f:
            f.write("epoch,batch,loss,avg_loss\n")

        # 評価CSVヘッダーを書き込み
        with open(self.eval_file, "w") as f:
            f.write("epoch,avg_score,max_score,min_score\n")

    def log_batch(self, epoch, batch, loss, avg_loss=None):
        """バッチごとのメトリクスを記録"""
        with open(self.metrics_file, "a") as f:
            avg_loss_str = f"{avg_loss:.6f}" if avg_loss is not None else ""
            f.write(f"{epoch},{batch},{loss:.6f},{avg_loss_str}\n")

    def log_epoch(self, epoch, avg_loss):
        """エポックごとのメトリクスを記録"""
        with open(self.metrics_file, "a") as f:
            f.write(f"{epoch},-1,-1,{avg_loss:.6f}\n")

    def log_evaluation(self, epoch, avg_score, max_score, min_score):
        """エポックごとの評価結果を記録"""
        with open(self.eval_file, "a") as f:
            f.write(f"{epoch},{avg_score:.2f},{max_score},{min_score}\n")
