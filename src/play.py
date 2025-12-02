"""
学習済みモデルを使用してMini 2048をプレイする
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from agent import select_best_action
from config import Config
from game_2048_3_3 import State
from model import Mini2048_SV_Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GamePlayer:
    def __init__(self, model_path, config=None):
        """
        Args:
            model_path: 学習済みモデルのパス
            config: 設定（Noneの場合はデフォルト）
        """
        if config is None:
            config = Config.default()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # モデルの読み込み
        logger.info(f"Loading model from {model_path}...")
        self.model = Mini2048_SV_Predictor()

        # チェックポイントを読み込む
        checkpoint = torch.load(model_path, map_location=self.device)

        # チェックポイント形式かどうかを判定
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # チェックポイント形式（epoch情報などを含む）
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
            )
        else:
            # 通常のstate_dict形式
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def select_action(self, state: State):
        """
        最適な行動を選択

        Args:
            state: Stateオブジェクト

        Returns:
            選択された行動（0:上, 1:右, 2:下, 3:左）
        """
        return select_best_action(state, self.model, self.device)

    def play_game(self, verbose=False):
        """
        1ゲームをプレイ

        Args:
            verbose: ログを詳細に表示するか

        Returns:
            (最終スコア, ターン数, 最大タイル)
        """
        state = State()
        state.initGame()
        turn = 0

        if verbose:
            logger.info("Game started!")
            state.print()

        while not state.isGameOver():
            turn += 1

            # 最適な行動を選択
            action, value = self.select_action(state)

            if action is None:
                break

            # 行動を実行
            state.play(action)
            state.putNewTile()

            if verbose and turn % 10 == 0:
                action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
                logger.info(
                    f"Turn {turn}: {action_names[action]} (Expected: {value:.2f})"
                )
                state.print()

        # 最大タイルを計算
        max_tile = 2 ** np.max(state.board) if np.max(state.board) > 0 else 0

        if verbose:
            logger.info(
                f"Game Over! Score: {state.score}, Turns: {turn}, Max Tile: {max_tile}"
            )

        return state.score, turn, max_tile


def play_multiple_games(model_path, num_games=100, save_stats=True):
    """
    複数ゲームをプレイして統計を取る

    Args:
        model_path: モデルのパス
        num_games: プレイするゲーム数
        save_stats: 統計を保存するか
    """
    player = GamePlayer(model_path)

    scores = []
    turns = []
    max_tiles = []

    logger.info(f"Playing {num_games} games...")

    for i in range(num_games):
        score, turn, max_tile = player.play_game(verbose=False)
        scores.append(score)
        turns.append(turn)
        max_tiles.append(max_tile)

        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{num_games} games")

    # 統計を計算
    scores_array = np.array(scores)
    turns_array = np.array(turns)
    max_tiles_array = np.array(max_tiles)

    logger.info("\n=== Game Statistics ===")
    logger.info(f"Games played: {num_games}")
    logger.info(
        f"Average Score: {np.mean(scores_array):.2f} ± {np.std(scores_array):.2f}"
    )
    logger.info(f"Median Score: {np.median(scores_array):.0f}")
    logger.info(f"Max Score: {np.max(scores_array)}")
    logger.info(f"Min Score: {np.min(scores_array)}")
    logger.info(f"Average Turns: {np.mean(turns_array):.2f}")
    logger.info("Max Tile Distribution:")
    unique_tiles, counts = np.unique(max_tiles_array, return_counts=True)
    for tile, count in zip(unique_tiles, counts):
        logger.info(f"  {tile}: {count} ({count / num_games * 100:.1f}%)")

    # 統計をファイルに保存
    if save_stats:
        stats_dir = Path("results")
        stats_dir.mkdir(exist_ok=True)

        stats_file = stats_dir / "play_stats.txt"
        with open(stats_file, "w") as f:
            f.write(f"Model: {model_path}\n")
            f.write(f"Games: {num_games}\n\n")
            f.write(
                f"Average Score: {np.mean(scores_array):.2f} ± {np.std(scores_array):.2f}\n"
            )
            f.write(f"Median Score: {np.median(scores_array):.0f}\n")
            f.write(f"Max Score: {np.max(scores_array)}\n")
            f.write(f"Min Score: {np.min(scores_array)}\n")
            f.write(f"Average Turns: {np.mean(turns_array):.2f}\n\n")
            f.write("Max Tile Distribution:\n")
            for tile, count in zip(unique_tiles, counts):
                f.write(f"  {tile}: {count} ({count / num_games * 100:.1f}%)\n")

        # 各ゲームの詳細を保存
        details_file = stats_dir / "game_details.csv"
        with open(details_file, "w") as f:
            f.write("game_id,score,turns,max_tile\n")
            for i, (score, turn, max_tile) in enumerate(zip(scores, turns, max_tiles)):
                f.write(f"{i + 1},{score},{turn},{max_tile}\n")

        logger.info(f"\nStatistics saved to {stats_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Play Mini 2048 with trained model")
    parser.add_argument(
        "--model",
        type=str,
        default="mini2048_sv_predictor.pth",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--num-games", type=int, default=100, help="Number of games to play"
    )
    parser.add_argument(
        "--single", action="store_true", help="Play a single game with verbose output"
    )

    args = parser.parse_args()

    if args.single:
        # 1ゲームを詳細に表示
        player = GamePlayer(args.model)
        score, turns, max_tile = player.play_game(verbose=True)
        logger.info(
            f"\nFinal Result: Score={score}, Turns={turns}, Max Tile={max_tile}"
        )
    else:
        # 複数ゲームをプレイして統計を取る
        play_multiple_games(args.model, num_games=args.num_games)


if __name__ == "__main__":
    main()
