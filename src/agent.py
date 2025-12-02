"""
Mini 2048のゲームエージェント
モデルを使用して最適な行動を選択する
"""

import torch

from game_2048_3_3 import State
from utils import board_to_index, index_to_onehot


def select_best_action(state, model, device, max_score=None):
    """
    モデルを使用して最適な行動を選択

    Args:
        state: Stateオブジェクト
        model: 学習済みモデル
        device: デバイス（'cuda' or 'cpu'）
        max_score: 最大スコア（クリッピングに使用、Noneの場合はクリッピングなし）

    Returns:
        (best_action, best_value): 最適な行動とその評価値
    """
    best_value = -float("inf")
    best_action = None

    # 各方向について評価
    for action in range(4):
        if not state.canMoveTo(action):
            continue

        # 仮想的に行動を実行
        temp_state = state.clone()
        temp_state.play(action)

        # 正規化された盤面をone-hotエンコーディングに変換
        state_index = board_to_index(temp_state.board)
        onehot = index_to_onehot(state_index)
        state_tensor = torch.from_numpy(onehot.flatten()).unsqueeze(0)
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            # モデルで予測（生のスコア値）
            value = model(state_tensor).item()

            # max_scoreが指定されている場合はクリッピング
            if max_score is not None:
                value = max(0, min(value, max_score))

        if value > best_value:
            best_value = value
            best_action = action

    return best_action, best_value


def play_single_game(model, device):
    """
    モデルを使用して1ゲームをプレイ

    Args:
        model: 学習済みモデル
        device: デバイス（'cuda' or 'cpu'）

    Returns:
        final_score: 最終スコア
    """
    state = State()
    state.initGame()

    while not state.isGameOver():
        best_action, _ = select_best_action(state, model, device)

        if best_action is None:
            break

        # 行動を実行
        state.play(best_action)
        state.putNewTile()

    return state.score
