import logging
import sqlite3
from pathlib import Path

import numpy as np
import ray

from game_2048_3_3 import State
from utils import board_to_index

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
base_dir = Path(__file__).parent.parent


def get_afterstates(state: State, cursor: sqlite3.Cursor):
    """
    パーフェクトプレイヤーのSQLiteデータベースから各行動の評価値を取得

    Args:
        state: 現在のゲーム状態
        cursor: SQLiteカーソル

    Returns:
        index_to_value: 各afterstateの評価値の辞書
        best_action: 最も評価値の高い手
    """

    # 有効なアクションのインデックスを一度に計算
    valid_actions = [action for action in range(4) if state.canMoveTo(action)]

    if not valid_actions:
        return {}, None

    # 各アクションのafterstate indexを計算
    afterstate_indices = []
    for action in valid_actions:
        temp_state = state.clone()
        temp_state.play(action)
        afterstate_indices.append(board_to_index(temp_state.board))

    # プレースホルダーを動的に生成してクエリ実行
    placeholders = ",".join("?" * len(afterstate_indices))
    cursor.execute(
        f"SELECT board_index, evaluation_score FROM board_data WHERE board_index IN ({placeholders})",
        afterstate_indices,
    )

    # board_index -> evaluation_score のマッピング
    index_to_value = dict(cursor.fetchall())

    # 最も評価値の高いアクションを1回のループで選択
    best_value = -float("inf")
    best_action = valid_actions[0]

    for action, idx in zip(valid_actions, afterstate_indices):
        value = index_to_value.get(idx, 0.0)
        if value > best_value:
            best_value = value
            best_action = action

    return index_to_value, best_action


@ray.remote
def run_parallel_games(num_games: int, idx: int):
    conn = sqlite3.connect(base_dir / "perfect.db", check_same_thread=False)
    cursor = conn.cursor()
    savefile = base_dir / f"perfect_afterstates_{idx}.txt"
    if savefile.exists():
        savefile.unlink()
    savefile.touch()
    scores = []
    for game_id in range(num_games):
        bd = State()
        bd.initGame()
        afterstates = []
        while not bd.isGameOver():
            index_to_values, best_action = get_afterstates(bd, cursor)
            if best_action is None:
                break
            afterstates.extend(index_to_values.keys())
            bd.play(best_action)
            bd.putNewTile()
        with savefile.open("a") as f:
            f.write(",".join(map(str, afterstates)) + "\n")
        scores.append(bd.score)
        print(f"Game {game_id + 1} completed with score {bd.score}")
    conn.close()
    return (
        f"Process {idx} completed {num_games} games. Average score: {np.mean(scores)}"
    )


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    num_processes = 10
    games_per_process = 10000
    futures = [
        run_parallel_games.remote(games_per_process, idx)
        for idx in range(num_processes)
    ]
    results = ray.get(futures)
    for res in results:
        print(res)
    ray.shutdown()
    print("All games completed.")
