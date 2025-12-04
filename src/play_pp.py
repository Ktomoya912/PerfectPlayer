from game_2048_3_3 import State
from play import play_multiple_games
from utils import board_to_index, readDB2

# データベースを読み込み
db = readDB2()


def get_values_from_db(state: State):
    """
    データベースから直接評価値を取得して最適な行動を選択

    Args:
        state: 現在の状態

    Returns:
        (action, value): 選択された行動と評価値
    """
    values = [-1e10] * 4

    for action in range(4):
        if not state.canMoveTo(action):
            continue

        temp_state = state.clone()
        temp_state.play(action)

        state_index = board_to_index(temp_state.board)
        score = db.get(state_index, 0.0)
        values[action] = score
    return values


# カスタム評価関数を使用してゲームをプレイ
if __name__ == "__main__":
    play_multiple_games(
        eval_func=get_values_from_db,
        num_games=10000,
        model_name="Perfect Play (DB)",
    )
