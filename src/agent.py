import torch

from game_2048_3_3 import State
from utils import board_to_index, index_to_onehot


def get_values(state: State, model, device):
    """
    モデルを使用して最適な行動を選択

    Args:
        state: Stateオブジェクト
        model: 学習済みモデル
        device: デバイス（'cuda' or 'cpu'）

    Returns:
        (best_action, best_value): 最適な行動とその評価値
    """
    values = [-1e10] * 4
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
            values[action] = value

    return values
