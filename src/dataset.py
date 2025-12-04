import numpy as np
import torch
from torch.utils.data import Dataset

from utils import index_to_onehot, readDB2


class AfterstateDataset(Dataset):
    def __init__(self, db_path="db2.out"):
        """
        db2.outから遅延ロードでデータを読み込むデータセット

        Args:
            db_path: データベースファイルのパス
            max_score: 最大スコア（正規化に使用）
        """
        # データベースを読み込み（indexとscoreのマッピング）
        print(f"Loading database from {db_path}...")
        db = readDB2(db_path)

        # numpy配列に変換（高速アクセスのため）
        self.indices = np.array(list(db.keys()), dtype=np.int64)
        self.scores = np.array(list(db.values()), dtype=np.float32)
        print(f"Loaded {len(self.indices)} states from database")

        # スコアの統計情報を表示
        print(
            f"Score statistics: min={self.scores.min():.2f}, max={self.scores.max():.2f}, mean={self.scores.mean():.2f}"
        )
        self.max_score = self.scores.max()
        # 辞書を削除してメモリを解放
        del db

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 該当するindexとscoreを取得（numpy配列から直接）
        board_index = int(self.indices[idx])
        score = float(self.scores[idx])

        # indexから盤面を復元
        onehot = index_to_onehot(board_index)

        # フラット化して99次元のベクトルにし、torchテンソルに変換
        state_tensor = torch.from_numpy(onehot.flatten())

        # 生のスコアをそのまま返す（正規化なし）
        return state_tensor, torch.tensor(score, dtype=torch.float32)
