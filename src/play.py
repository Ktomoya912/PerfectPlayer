import argparse
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import numpy as np
import torch

from agent import get_values
from config import Config
from game_2048_3_3 import State
from model import Mini2048_SV_Predictor
from utils import calc_progress

stop_event = threading.Event()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

queue = Queue()
tasks = os.cpu_count() or 4

SAVE_DIR = Path("board_data") / "test"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class GamePlayer:
    def __init__(self, eval_func: callable):  # type: ignore
        """
        Args:
            eval_func: 評価関数（state -> (values[4])を返す関数）
        """
        self.eval_func = eval_func
        self.turn = 0
        self.states = []
        self.after_states = []
        self.evals = []

    def play_game(self, thread_id=0):
        """
        1ゲームをプレイ

        Args:
            thread_id: スレッドID（ログ用）

        Returns:
            None（結果はqueueに格納される）
        """
        state = State()
        state.initGame()
        self.turn = 0
        self.states = []
        self.after_states = []
        self.evals = []

        while not state.isGameOver() and not stop_event.is_set():
            self.turn += 1

            # 最適な行動を選択
            values = self.eval_func(state)
            progress = calc_progress(state.board.copy())
            self.states.append((state.board.copy(), progress))
            action = np.argmax(values)
            self.evals.append((values, progress))

            if action is None:
                break

            # 行動を実行
            state.play(action)
            self.after_states.append(
                (state.board.copy(), calc_progress(state.board.copy()))
            )
            state.putNewTile()

        # ゲーム終了時の情報を収集
        max_tile = 2 ** np.max(state.board) if np.max(state.board) > 0 else 0
        gameover_progress = calc_progress(state.board.copy())

        # 結果をキューに格納
        game_data = {
            "gameover_turn": self.turn,
            "gameover_score": state.score,
            "gameover_progress": gameover_progress,
            "max_tile": max_tile,
            "states": self.states,
            "after_states": self.after_states,
            "evals": self.evals,
        }

        if not stop_event.is_set():
            queue.put(game_data)

        logger.debug(
            f"Thread {thread_id}: Game finished. Score={state.score}, Turns={self.turn}"
        )


def create_model_eval_func(model_path, config=None):
    """
    モデルから評価関数を作成するヘルパー関数

    Args:
        model_path: 学習済みモデルのパス
        config: 設定（Noneの場合はデフォルト）

    Returns:
        eval_func: 評価関数（state -> (action, value)を返す）
    """
    if config is None:
        config = Config.default()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの読み込み
    logger.info(f"Loading model from {model_path}...")
    model = Mini2048_SV_Predictor()

    # チェックポイントを読み込む
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # チェックポイント形式かどうかを判定
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    # 評価関数を返す
    def eval_func(state):
        return get_values(state, model, device)

    return eval_func


def play_multiple_games(eval_func, num_games=100, save_stats=True, model_name=None):
    """
    複数ゲームをプレイして統計を取る

    Args:
        eval_func: 評価関数（state -> (values[4])を返す）
        num_games: プレイするゲーム数
        save_stats: 統計を保存するか
        model_name: モデル名（統計保存時に使用、Noneの場合は"Custom eval function"）
    """

    logger.info(f"Playing {num_games} games with {tasks} threads...")

    # 各スレッドが1ゲームずつプレイし続ける
    def worker(thread_id):
        while not stop_event.is_set():
            player = GamePlayer(eval_func)
            player.play_game(thread_id)

    executor = ThreadPoolExecutor(max_workers=tasks)
    futures = []
    for i in range(tasks):
        futures.append(executor.submit(worker, i))

    data = []
    try:
        while len(data) < num_games:
            game_data = queue.get(timeout=300)  # 5分のタイムアウト
            data.append(game_data)
            if len(data) % 10 == 0 or len(data) == num_games:
                logger.info(f"Collected {len(data)}/{num_games} games.")
    except Exception as e:
        logger.error(f"Error collecting game data: {e}")
    finally:
        # 停止シグナルを送信
        stop_event.set()
        executor.shutdown(wait=True, cancel_futures=True)

    logger.info(f"Played {num_games} games.")

    if not data:
        logger.error("No game data collected!")
        return

    scores = [game_data["gameover_score"] for game_data in data]
    turns = [game_data["gameover_turn"] for game_data in data]
    max_tiles = [game_data["max_tile"] for game_data in data]

    # 統計を表示
    logger.info("\n=== Game Statistics ===")
    logger.info(f"Games played: {len(data)}")
    logger.info(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    logger.info(f"Median Score: {np.median(scores):.0f}")
    logger.info(f"Max Score: {np.max(scores)}")
    logger.info(f"Min Score: {np.min(scores)}")
    logger.info(f"Average Turns: {np.mean(turns):.2f}")
    logger.info("Max Tile Distribution:")
    unique_tiles, counts = np.unique(max_tiles, return_counts=True)
    for tile, count in zip(unique_tiles, counts):
        logger.info(f"  {tile}: {count} ({count / len(data) * 100:.1f}%)")

    # ゲームデータを保存
    if save_stats:
        try:
            with (
                open(SAVE_DIR / "state.txt", "w") as f_state,
                open(SAVE_DIR / "eval.txt", "w") as f_eval,
                open(SAVE_DIR / "after-state.txt", "w") as f_after,
            ):
                for i, game_data in enumerate(data):
                    states = game_data["states"]
                    after_states = game_data["after_states"]
                    evals = game_data["evals"]
                    gameover_info = f"gameover_turn: {game_data['gameover_turn']}; game: {i + 1}; progress: {game_data['gameover_progress']}; score: {game_data['gameover_score']}"

                    state_strs = []
                    eval_strs = []
                    after_state_strs = []
                    for state, after_state, eval_data in zip(
                        states, after_states, evals
                    ):
                        state_str = f"{' '.join(map(str, state[0]))} {state[1]}"
                        after_state_str = (
                            f"{' '.join(map(str, after_state[0]))} {after_state[1]}"
                        )
                        # eval_data[0]がリストか単一値かによって処理を分岐
                        if isinstance(eval_data[0], (list, tuple, np.ndarray)):
                            eval_str = f"{' '.join(str(float(ev)) for ev in eval_data[0])} {eval_data[1]}"
                        else:
                            eval_str = f"{float(eval_data[0])} {eval_data[1]}"
                        state_strs.append(state_str)
                        eval_strs.append(eval_str)
                        after_state_strs.append(after_state_str)
                    # 1ゲーム分のデータを追記
                    f_state.write("\n".join(state_strs) + f"\n{gameover_info}\n")
                    f_eval.write("\n".join(eval_strs) + f"\n{gameover_info}\n")
                    f_after.write("\n".join(after_state_strs) + f"\n{gameover_info}\n")

            logger.info(f"Game data saved to {SAVE_DIR}/")
        except Exception as e:
            logger.error(f"Error saving game data: {e}")
            return

    logger.info("End Free Play")


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

    args = parser.parse_args()
    eval_func = create_model_eval_func(args.model)
    play_multiple_games(eval_func, num_games=args.num_games, model_name=args.model)


if __name__ == "__main__":
    main()
