"""
listup.pyで列挙されたafterstateを使った教師あり学習
"""

import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from agent import get_values
from config import Config
from game_2048_3_3 import State
from logger import Logger, MetricsLogger
from model import Mini2048_SV_Predictor
from utils import index_to_onehot


class AfterstateFileDataset(Dataset):
    """
    perfect_afterstates_*.txtファイルから読み込んだafterstateを使用するデータセット
    """

    def __init__(self, afterstate_files_pattern: str, db_path: Path):
        """
        Args:
            afterstate_files_pattern: afterstateファイルのパターン (例: "perfect_afterstates_*.txt")
            db_path: 評価値を取得するSQLiteデータベースのパス
        """
        self.db_path = db_path

        # 1. ファイルからafterstateインデックスを読み込み
        base_dir = Path(__file__).parent.parent
        afterstate_files = sorted(base_dir.glob(afterstate_files_pattern))

        if not afterstate_files:
            raise ValueError(
                f"No files found matching pattern: {afterstate_files_pattern}"
            )

        print(f"Found {len(afterstate_files)} afterstate files")

        # 全てのafterstateインデックスを収集（重複を保持）
        all_indices = []
        for file in afterstate_files:
            with file.open("r") as f:
                for line in f:
                    indices = [int(idx) for idx in line.strip().split(",") if idx]
                    all_indices.extend(indices)

        self.board_indices = all_indices
        unique_count = len(set(all_indices))
        print(f"Total afterstates (with duplicates): {len(self.board_indices)}")
        print(f"Unique afterstates: {unique_count}")

        # 2. データベースから評価値を取得
        self._load_evaluations()

    def _load_evaluations(self):
        """データベースから評価値を一括取得"""
        print("Loading evaluation scores from database...")

        # 先にnumpy配列化（重複を保持）
        self.board_indices = np.array(self.board_indices, dtype=np.int64)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ユニークなインデックスを取得（データベースクエリの効率化のため）
        unique_indices = np.unique(self.board_indices).tolist()

        # バッチサイズで分割して取得（SQLiteの制限を回避）
        batch_size = 10000
        index_to_value = {}

        for i in range(0, len(unique_indices), batch_size):
            batch_indices = unique_indices[i : i + batch_size]
            placeholders = ",".join("?" * len(batch_indices))
            cursor.execute(
                f"SELECT board_index, evaluation_score FROM board_data WHERE board_index IN ({placeholders})",
                batch_indices,
            )
            index_to_value.update(dict(cursor.fetchall()))

            if (i // batch_size + 1) % 10 == 0:
                print(
                    f"  Loaded {i + len(batch_indices)}/{len(unique_indices)} unique evaluations"
                )

        conn.close()

        # 元の重複を含むboard_indicesに対して評価値をマッピング
        print("Mapping evaluation scores to afterstates...")
        evaluations_list = [
            index_to_value.get(int(idx), 0.0) for idx in self.board_indices
        ]
        self.evaluations = np.array(evaluations_list, dtype=np.float32)

        # 統計情報
        self.max_score = float(np.max(self.evaluations))
        self.min_score = float(np.min(self.evaluations))
        self.mean_score = float(np.mean(self.evaluations))

        print(
            f"Evaluation scores loaded: Max={self.max_score:.2f}, Min={self.min_score:.2f}, Mean={self.mean_score:.2f}"
        )
        print(f"Total training samples (with duplicates): {len(self.evaluations)}")
        print(f"Unique afterstates: {len(unique_indices)}")

    def __len__(self):
        return len(self.board_indices)

    def __getitem__(self, idx):
        """
        Returns:
            (input_tensor, target_value)
        """
        board_index = int(self.board_indices[idx])
        evaluation = float(self.evaluations[idx])

        # board_indexからone-hot表現に変換
        onehot = index_to_onehot(board_index)
        input_tensor = torch.from_numpy(onehot.flatten()).float()
        target_tensor = torch.tensor(evaluation, dtype=torch.float32)

        return input_tensor, target_tensor


def get_initial_evaluation_score(model, device):
    """初期状態の評価値を取得"""
    state = State()
    state.initGame()
    values, _ = get_values(state, model, device)
    return max(values)


def evaluate_model(model, device, num_games=10):
    """
    モデルの性能を評価（複数ゲームをプレイ）

    Args:
        model: 評価するモデル
        device: デバイス
        num_games: プレイするゲーム数

    Returns:
        (average_score, max_score, min_score)
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for _ in range(num_games):
            state = State()
            state.initGame()

            while not state.isGameOver():
                values, _ = get_values(state, model, device)
                best_action = np.argmax(values)

                if best_action is None:
                    break

                state.play(best_action)
                state.putNewTile()

            scores.append(state.score)

    model.train()
    return np.mean(scores), np.max(scores), np.min(scores)


def train():
    # 設定の読み込み
    config = Config.default()

    # ロガーの初期化
    logger = Logger()
    metrics_logger = MetricsLogger()

    # 設定を表示
    logger.info("\n" + str(config))

    # 1. データローダーの準備
    logger.info("Preparing dataset from afterstate files...")
    train_dataset = AfterstateFileDataset(
        afterstate_files_pattern="perfect_afterstates_*.txt", db_path=Path("perfect.db")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    logger.info(f"Dataset size: {len(train_dataset)}")
    logger.info(f"Max score in dataset: {train_dataset.max_score:.2f}")
    logger.info(f"Mean score in dataset: {train_dataset.mean_score:.2f}")

    # 2. モデル、損失関数、オプティマイザの初期化
    logger.info("Initializing model...")
    model = Mini2048_SV_Predictor()
    criterion = nn.MSELoss()
    logger.info("Using MSE Loss for regression")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # デバイスの設定
    device = torch.device(
        config.training.device if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    logger.info(f"Using device: {device}")

    # チェックポイントディレクトリの作成
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # チェックポイントから再開
    start_epoch = 0
    global_batch_count = 0
    checkpoint_files = sorted(
        checkpoint_dir.glob("model_afterstate_epoch_*.pth"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        logger.info(f"Loading checkpoint from {latest_checkpoint}...")
        checkpoint = torch.load(
            latest_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_batch_count = start_epoch * len(train_loader)
        logger.info(f"Resuming from epoch {start_epoch}, batch {global_batch_count}")
    else:
        logger.info("No checkpoint found. Starting from scratch.")

    # 3. トレーニングの実行
    logger.info("Starting Supervised Learning Training from Afterstates...")

    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)

            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value

            del outputs, loss

            global_batch_count += 1

            # バッチごとのログ出力
            if (batch_idx + 1) % config.training.log_interval == 0:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                eval_score = get_initial_evaluation_score(model, device)
                logger.info(
                    f"Epoch [{epoch + 1}/{config.training.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss_value:.4f} "
                    f"Avg Loss: {avg_loss_so_far:.4f} "
                    f"Eval Score: {eval_score:.2f}"
                )
                metrics_logger.log_batch(
                    epoch + 1, batch_idx + 1, loss_value, avg_loss_so_far
                )

            # 1000バッチごとに評価
            if global_batch_count % 1000 == 0:
                logger.info(
                    f"[Batch {global_batch_count}] Evaluating model performance..."
                )
                avg_score, max_score_eval, min_score_eval = evaluate_model(
                    model, device, num_games=10
                )
                logger.info(
                    f"[Batch {global_batch_count}] Evaluation: Avg Score={avg_score:.2f}, Max={max_score_eval}, Min={min_score_eval}"
                )
                metrics_logger.log_evaluation(
                    epoch + 1, avg_score, max_score_eval, min_score_eval
                )

        # エポック終了時のログ
        avg_loss = total_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch + 1}/{config.training.epochs} completed. Avg Loss: {avg_loss:.4f}"
        )
        metrics_logger.log_epoch(epoch + 1, avg_loss)

        # チェックポイントの保存
        if (epoch + 1) % config.training.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"model_afterstate_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "max_score": train_dataset.max_score,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    logger.info("Training finished.")

    # 最終モデルの保存
    final_model_path = "mini2048_sv_predictor_afterstate.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")


if __name__ == "__main__":
    train()
