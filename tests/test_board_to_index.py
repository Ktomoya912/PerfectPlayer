"""
board_to_index関数が8方向の回転・鏡面で同じindexを返すかテストする
"""

import sys
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from utils import ROTATE3, board_to_index


class TestBoardToIndexInvariance:
    """board_to_indexの不変性をテストするクラス"""

    @pytest.fixture
    def test_boards(self):
        """テスト用の盤面を提供"""
        return [
            # 空の盤面
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
            # 対称な盤面
            np.array([1, 2, 1, 2, 3, 2, 1, 2, 1], dtype=np.int32),
            # 非対称な盤面
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32),
            # ランダムな盤面
            np.array([2, 0, 3, 1, 4, 0, 5, 2, 1], dtype=np.int32),
            np.array([3, 5, 2, 1, 0, 4, 2, 1, 3], dtype=np.int32),
            # 実際のゲーム盤面の例
            np.array([2, 3, 2, 3, 4, 3, 2, 3, 2], dtype=np.int32),
        ]

    def test_all_transformations_same_index(self, test_boards):
        """すべての変換で同じindexを返すことをテスト"""
        for board in test_boards:
            original_index = board_to_index(board)

            for transform_idx, rotation in enumerate(ROTATE3):
                transformed_board = board[rotation]
                transformed_index = board_to_index(transformed_board)

                assert transformed_index == original_index, (
                    f"Transform {transform_idx} failed: {transformed_index} != {original_index} for board {board}"
                )

    def test_specific_board_transformations(self):
        """特定の盤面で各変換を個別にテスト"""
        board = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
        original_index = board_to_index(board)

        transform_names = [
            "Original",
            "Horizontal flip",
            "90° rotation",
            "270° rotation",
            "180° rotation",
            "180° rotation + H flip",
            "90° rotation + H flip",
            "270° rotation + H flip",
        ]

        for rotation, name in zip(ROTATE3, transform_names):
            transformed = board[rotation]
            transformed_index = board_to_index(transformed)
            assert transformed_index == original_index, (
                f"{name} failed: {transformed_index} != {original_index}"
            )

    def test_empty_board(self):
        """空の盤面でテスト"""
        board = np.zeros(9, dtype=np.int32)
        indices = [board_to_index(board[rotation]) for rotation in ROTATE3]
        assert len(set(indices)) == 1, (
            "Empty board should have the same index for all transformations"
        )

    def test_symmetric_board(self):
        """対称な盤面でテスト"""
        board = np.array([1, 2, 1, 2, 3, 2, 1, 2, 1], dtype=np.int32)
        indices = [board_to_index(board[rotation]) for rotation in ROTATE3]
        assert len(set(indices)) == 1, (
            "Symmetric board should have the same index for all transformations"
        )


def test_manual_rotation_visual(capsys):
    """
    手動で回転を確認するテスト（視覚的確認用）
    pytest -v -s でログを表示
    """
    board = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)

    print("\n" + "=" * 70)
    print("Manual rotation test:")
    print("=" * 70)
    print("\nOriginal board (3x3):")
    print(board.reshape(3, 3))
    original_index = board_to_index(board)
    print(f"Index: {original_index}\n")

    transform_names = [
        "Original",
        "Horizontal flip",
        "90° rotation",
        "270° rotation",
        "180° rotation",
        "180° rotation + H flip",
        "90° rotation + H flip",
        "270° rotation + H flip",
    ]

    all_same = True
    for rotation, name in zip(ROTATE3, transform_names):
        transformed = board[rotation]
        transformed_index = board_to_index(transformed)
        status = "✓" if transformed_index == original_index else "❌"
        print(f"{status} {name}:")
        print(transformed.reshape(3, 3))
        print(f"Index: {transformed_index}\n")
        if transformed_index != original_index:
            all_same = False

    assert all_same, "Some transformations produced different indices"


if __name__ == "__main__":
    # pytest実行
    pytest.main([__file__, "-v", "-s"])
