"""
強化学習によるMini 2048の学習 (Ray マルチプロセス版)
TD学習（Temporal Difference Learning）を使用
"""

import argparse
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import ray
import torch
from torch import nn, optim

from agent import get_values
from game_2048_3_3 import State
from model import Mini2048_SV_Predictor
from utils import board_to_index, index_to_onehot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def select_action_from_perfect_player(
    state: State, db_path: Path = Path("perfect.db")
) -> tuple[int, float]:
    """
    パーフェクトプレイヤーのSQLiteデータベースから最適な行動を選択

    Args:
        state: 現在のゲーム状態
        db_path: SQLiteデータベースのパス

    Returns:
        (action, value): 選択された行動と評価値
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    best_value = -float("inf")
    best_action = -1

    for action in range(4):
        if not state.canMoveTo(action):
            continue

        temp_state = state.clone()
        temp_state.play(action)

        state_index = board_to_index(temp_state.board)

        # SQLiteから評価値を取得
        cursor.execute(
            "SELECT evaluation_score FROM board_data WHERE board_index = ?",
            (state_index,),
        )
        result = cursor.fetchone()
        score = result[0] if result else 0.0

        if score > best_value:
            best_value = score
            best_action = action

    conn.close()
    return best_action, best_value


@dataclass
class RLConfig:
    """強化学習の設定"""

    num_workers: int = os.cpu_count() or 4
    batch_size: int = 1024
    learning_rate: float = 0.001
    checkpoint_interval: int = 1000  # バッチごとにチェックポイント保存
    checkpoint_dir: str = "checkpoints_rl"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_games: Optional[int] = None  # 最大ゲーム数(Noneの場合は無制限)
    use_perfect_player_action: bool = False  # パーフェクトプレイヤーで行動選択するか
    use_perfect_player_target: bool = (
        False  # パーフェクトプレイヤーの評価値を学習ターゲットにするか
    )
    perfect_db_path: Path = Path("perfect.db")


class ExperienceBuffer:
    """経験データを管理するバッファ"""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.board_indices = []  # board_indexを保存
        self.values = []

    def add(self, board_index: int, value: float):
        """経験データを追加"""
        self.board_indices.append(board_index)
        self.values.append(value)

    def is_full(self) -> bool:
        """バッファが満杯かチェック"""
        return len(self.board_indices) >= self.batch_size

    def get_batch(self):
        """バッチデータを取得してクリア"""
        board_indices = self.board_indices.copy()
        values = self.values.copy()
        self.clear()
        return board_indices, values

    def clear(self):
        """バッファをクリア"""
        self.board_indices.clear()
        self.values.clear()

    def __len__(self):
        return len(self.board_indices)


class RLTrainer:
    """強化学習のトレーナークラス"""

    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        self.device = torch.device(self.config.device)

        # モデルとオプティマイザ
        self.model = Mini2048_SV_Predictor().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.criterion = nn.MSELoss()

        # 統計情報
        self.total_games = 0
        self.total_batches = 0

        # チェックポイントディレクトリ
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized RL Trainer with device: {self.device}")
        logger.info(f"Config: {self.config}")

    def get_model_weights(self):
        """モデルの重みを取得（CPUに移動）"""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def train_batch(self, board_indices: list, values: list):
        """バッチ学習"""
        self.model.train()

        # board_indexからone-hot表現に変換
        batch_onehots = []
        for board_idx in board_indices:
            onehot = index_to_onehot(board_idx)
            batch_onehots.append(onehot.flatten())

        # テンソルに変換
        input_tensor = torch.tensor(
            np.array(batch_onehots), dtype=torch.float32, device=self.device
        )
        target_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        # 勾配計算と更新
        self.optimizer.zero_grad()
        outputs = self.model(input_tensor)
        loss = self.criterion(outputs.squeeze(), target_tensor)
        loss.backward()
        self.optimizer.step()

        self.total_batches += 1
        logger.info(f"Batch {self.total_batches}: Loss={loss.item():.6f}")

        return loss.item()

    def save_checkpoint(self, suffix: str = ""):
        """チェックポイントを保存"""
        checkpoint_path = (
            self.checkpoint_dir / f"rl_model_batch_{self.total_batches}{suffix}.pth"
        )
        torch.save(
            {
                "batch": self.total_batches,
                "total_games": self.total_games,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイントを読み込み"""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_batches = checkpoint["batch"]
        self.total_games = checkpoint.get("total_games", 0)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(
            f"Resuming from batch {self.total_batches}, total games {self.total_games}"
        )


@ray.remote
class GameWorker:
    """ゲームをプレイするワーカー（Rayアクター）"""

    def __init__(
        self,
        worker_id: int,
        model_weights: dict,
        use_perfect_player_action: bool = False,
        use_perfect_player_target: bool = False,
        perfect_db_path: Path = Path("perfect.db"),
        device: str = "cpu",
    ):
        self.worker_id = worker_id
        self.games_played = 0
        self.use_perfect_player_action = use_perfect_player_action
        self.use_perfect_player_target = use_perfect_player_target
        self.perfect_db_path = perfect_db_path
        self.device = torch.device(device)

        # モデルを初期化して重みをロード
        self.model = Mini2048_SV_Predictor().to(self.device)
        self.model.load_state_dict(model_weights)
        self.model.eval()

        logger.info(f"Worker {worker_id} initialized on device: {self.device}")

    def update_model(self, model_weights: dict):
        """モデルの重みを更新"""
        self.model.load_state_dict(model_weights)
        self.model.eval()

    def play_game(self):
        """1ゲームをプレイしてTD学習データを生成"""
        state = State()
        state.initGame()
        turn = 0
        experiences = []  # (board_index, td_target)のリスト
        initial_eval = None

        while True:
            turn += 1

            # NNから評価値を計算(常に必要)
            values, afterstate_bonus = get_values(state, self.model, self.device)

            # 行動選択
            if self.use_perfect_player_action:
                # パーフェクトプレイヤーから行動を選択
                action, _ = select_action_from_perfect_player(
                    state, self.perfect_db_path
                )
            else:
                # 学習中のモデルから行動を選択
                action = np.argmax(values)

            # 学習ターゲットの計算
            if self.use_perfect_player_target:
                # パーフェクトプレイヤーの評価値をターゲットに使用
                _, td_target = select_action_from_perfect_player(
                    state, self.perfect_db_path
                )
            else:
                # NNの出力から計算されたTDターゲット
                td_target = values[action] + afterstate_bonus[action]

            # 前の状態があればTD学習データとして追加
            if len(experiences) > 0:
                # 前の状態のターゲットを現在の評価値で更新
                prev_board_index = experiences[-1][0]
                experiences[-1] = (prev_board_index, float(td_target))

            # 初期評価値を記録
            if turn == 1:
                initial_eval = td_target

            # 行動を実行
            state.play(action)
            # 正規化されたboard_indexを保存（ターゲットは次のステップで設定）
            current_board_index = board_to_index(state.board.copy())
            experiences.append((current_board_index, 0.0))  # 一時的に0.0
            state.putNewTile()

            # ゲーム終了チェック
            if state.isGameOver():
                # 終端状態の価値は0
                experiences[-1] = (current_board_index, 0.0)

                self.games_played += 1

                action_type = "PP" if self.use_perfect_player_action else "AI"
                target_type = "PP" if self.use_perfect_player_target else "TD"
                logger.info(
                    f"Worker {self.worker_id:02d} (Action:{action_type}/Target:{target_type}): "
                    f"Game {self.games_played} finished | "
                    f"Score={state.score:04d}, Turns={turn:04d}, "
                    f"Initial eval={initial_eval:.2f}"
                )
                break

        return experiences, state.score

    def get_stats(self):
        """統計情報を取得"""
        return {"worker_id": self.worker_id, "games_played": self.games_played}


def main(config: Optional[RLConfig] = None):
    """メイン関数"""
    if config is None:
        config = RLConfig()

    # Rayの初期化
    if not ray.is_initialized():
        ray.init(num_cpus=config.num_workers)
        logger.info(f"Ray initialized with {config.num_workers} CPUs")

    trainer = RLTrainer(config)
    buffer = ExperienceBuffer(batch_size=config.batch_size)

    # 初期モデルの重みを取得
    model_weights = trainer.get_model_weights()

    # ワーカーアクターを作成
    workers = [
        GameWorker.remote(
            i,
            model_weights,
            use_perfect_player_action=config.use_perfect_player_action,
            use_perfect_player_target=config.use_perfect_player_target,
            perfect_db_path=config.perfect_db_path,
            device="cpu",  # 各ワーカーはCPUを使用
        )
        for i in range(config.num_workers)
    ]

    action_type = (
        "Perfect Player" if config.use_perfect_player_action else "Learning AI"
    )
    target_type = (
        "Perfect Player" if config.use_perfect_player_target else "TD Learning"
    )
    logger.info(f"Starting {config.num_workers} worker processes...")
    logger.info(f"  Action selection: {action_type}")
    logger.info(f"  Learning target: {target_type}")
    if config.use_perfect_player_action or config.use_perfect_player_target:
        logger.info(f"  SQLite database: {config.perfect_db_path}")
    if config.max_games:
        logger.info(f"  Training will stop after {config.max_games} games")

    try:
        # 各ワーカーに最初のゲームを開始させる
        pending_games = {worker.play_game.remote(): worker for worker in workers}

        while True:
            # 最大ゲーム数チェック
            if config.max_games and trainer.total_games >= config.max_games:
                logger.info(
                    f"Reached maximum games ({config.max_games}). Stopping training..."
                )
                break

            # 完了したゲームを待機
            if pending_games:
                ready_refs, remaining_refs = ray.wait(
                    list(pending_games.keys()), num_returns=1, timeout=1.0
                )

                for ready_ref in ready_refs:
                    # ゲーム結果を取得
                    experiences, score = ray.get(ready_ref)
                    worker = pending_games.pop(ready_ref)

                    # 経験データをバッファに追加
                    for board_index, td_target in experiences:
                        buffer.add(board_index, td_target)

                    trainer.total_games += 1

                    # バッファが満杯になったら学習
                    if buffer.is_full():
                        board_indices, values = buffer.get_batch()
                        trainer.train_batch(board_indices, values)

                        # 定期的にチェックポイント保存とモデル更新
                        if trainer.total_batches % config.checkpoint_interval == 0:
                            trainer.save_checkpoint()
                            # 全ワーカーのモデルを更新
                            model_weights = trainer.get_model_weights()
                            for w in workers:
                                w.update_model.remote(model_weights)
                            logger.info("Updated model weights for all workers")

                    # 最大ゲーム数に達していなければ、次のゲームを開始
                    if not config.max_games or trainer.total_games < config.max_games:
                        pending_games[worker.play_game.remote()] = worker

    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down...")
    except Exception as e:
        logger.exception(f"Training error: {e}")
    finally:
        # 残りのデータを学習
        if len(buffer) > 0:
            board_indices, values = buffer.get_batch()
            trainer.train_batch(board_indices, values)

        # 最終チェックポイント保存
        trainer.save_checkpoint(suffix="_final")

        # 統計情報を収集
        stats = ray.get([worker.get_stats.remote() for worker in workers])
        for stat in stats:
            logger.info(
                f"Worker {stat['worker_id']}: {stat['games_played']} games played"
            )

        logger.info("All workers terminated. Training complete.")
        logger.info(
            f"Total games: {trainer.total_games}, Total batches: {trainer.total_batches}"
        )

        # Rayのシャットダウン
        ray.shutdown()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mini 2048 Reinforcement Learning (Ray)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to play (default: unlimited)",
    )
    parser.add_argument(
        "--use-perfect-player-action",
        action="store_true",
        help="Use perfect player for action selection",
    )
    parser.add_argument(
        "--use-perfect-player-target",
        action="store_true",
        help="Use perfect player evaluation as learning target",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N batches (default: 1000)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--perfect-db",
        type=str,
        default="perfect.db",
        help="Path to perfect player SQLite database (default: perfect.db)",
    )

    args = parser.parse_args()

    # 設定を作成
    config = RLConfig(
        max_games=args.max_games,
        use_perfect_player_action=args.use_perfect_player_action,
        use_perfect_player_target=args.use_perfect_player_target,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.num_workers or (os.cpu_count() or 4),
        perfect_db_path=args.perfect_db,
    )

    main(config)
