import struct
import sys
from pathlib import Path

import numpy as np

base_dir = Path(__file__).resolve().parent.parent

# 8方向の回転・鏡面変換パターン（numpy配列化）
ROTATE3 = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [2, 1, 0, 5, 4, 3, 8, 7, 6],
        [2, 5, 8, 1, 4, 7, 0, 3, 6],
        [0, 3, 6, 1, 4, 7, 2, 5, 8],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
        [6, 7, 8, 3, 4, 5, 0, 1, 2],
        [6, 3, 0, 7, 4, 1, 8, 5, 2],
        [8, 5, 2, 7, 4, 1, 6, 3, 0],
    ],
    dtype=np.int32,
)

# pow11をnumpy配列化
POW11 = np.array([11**i for i in range(9)], dtype=np.int64)


def index_to_onehot(index):
    """indexから3x3x11のone-hotエンコーディングに変換する関数(numpyを用いた高速な関数)"""
    onehot = np.zeros((11, 3, 3), dtype=np.float32)

    # indexをPythonのintに変換（numpy scalarの場合に対応）
    index = int(index)

    # indexから9つのタイル値を抽出してベクトル化で設定
    tile_values = np.empty(9, dtype=np.int32)
    for i in range(9):
        tile_values[i] = index % 11
        index //= 11

    # ベクトル化された操作でone-hotを設定
    rows = np.arange(9) // 3
    cols = np.arange(9) % 3
    onehot[tile_values, rows, cols] = 1.0

    return onehot


def index_to_board(index):
    """indexから盤面を復元する関数(numpyを用いた高速な関数)"""
    board = np.empty(9, dtype=np.int32)

    for i in range(9):
        board[i] = index % 11
        index //= 11

    return board


def board_to_index(board):
    """正規化されていないboardから正規化を行った後のindexを計算する関数(numpyを用いた高速な関数)"""

    # boardをnumpy配列に変換（既にnumpy配列の場合はそのまま）
    if not isinstance(board, np.ndarray):
        board = np.array(board, dtype=np.int32)

    # 全ての回転パターンでindexを計算（ベクトル化）
    # rotate3[i, j]はi番目の回転パターンでのj番目の位置
    # board[rotate3]で全回転パターンの盤面を取得 (8x9)
    rotated_boards = board[ROTATE3]  # shape: (8, 9)

    # 各回転パターンのindexを計算（ベクトル化）
    indices = np.sum(rotated_boards * POW11, axis=1)  # shape: (8,)

    # 最小のindexを取得してPythonのintとして返す
    return int(np.min(indices))


def readDB2(file_path=base_dir / "db2.out"):
    try:
        with open(file_path, "rb") as fp:
            # Read count (int, 4 bytes)
            count_data = fp.read(4)
            if len(count_data) != 4:
                print("error reading DB: size mismatch", file=sys.stderr)
                sys.exit(1)
            count = struct.unpack("i", count_data)[0]

            # Read ids array (int array)
            ids_data = fp.read(4 * count)
            if len(ids_data) != 4 * count:
                print(
                    f"error reading DB: size mismatch {len(ids_data) // 4} != {count}",
                    file=sys.stderr,
                )
                sys.exit(1)
            ids = struct.unpack(f"{count}i", ids_data)

            # Read evs array (double array)
            evs_data = fp.read(8 * count)
            if len(evs_data) != 8 * count:
                print(
                    f"error reading DB: size mismatch {len(evs_data) // 8} != {count}",
                    file=sys.stderr,
                )
                sys.exit(1)
            evs = struct.unpack(f"{count}d", evs_data)

            # Populate db dictionary
            db = {}
            for i in range(count):
                db[ids[i]] = evs[i]

            return db

    except FileNotFoundError:
        print(f"error opening file: {file_path}", file=sys.stderr)
        sys.exit(1)


def calc_progress(board: np.ndarray):
    """
    ボード状態から進捗度（progress）を計算
    各タイルの値の2の累乗和を2で割った値を返す。
    """
    return sum(2**i for i in board if i > 0) // 2
