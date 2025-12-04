from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agent import get_values
from config import Config
from dataset import AfterstateDataset
from game_2048_3_3 import State
from logger import Logger, MetricsLogger
from model import Mini2048_SV_Predictor


def get_initial_evaluation_score(model, device):
    state = State()
    state.initGame()
    values, _ = get_values(state, model, device)
    return max(values)


def evaluate_model(model, device, num_games=10):
    """
    モデルの性能を評価（複数ゲームをプレイ）

    Args:
        model: 評価するモデル
        device: デバイス
        num_games: プレイするゲーム数

    Returns:
        (average_score, max_score, min_score)
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for _ in range(num_games):
            state = State()
            state.initGame()

            while not state.isGameOver():
                values, _ = get_values(state, model, device)
                best_action = np.argmax(values)

                if best_action is None:
                    break

                # 行動を実行
                state.play(best_action)
                state.putNewTile()

            scores.append(state.score)

    model.train()
    return np.mean(scores), np.max(scores), np.min(scores)


def train():
    # 設定の読み込み
    config = Config.default()

    # ロガーの初期化
    logger = Logger()
    metrics_logger = MetricsLogger()

    # 設定を表示
    logger.info("\n" + str(config))

    # 1. データローダーの準備
    logger.info("Preparing dataset...")
    train_dataset = AfterstateDataset(db_path=config.data.db_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,  # GPU転送の高速化
    )
    logger.info(f"Dataset size: {len(train_dataset)}")

    # データセットの最大スコアを取得
    max_score = train_dataset.max_score
    logger.info(f"Max score in dataset: {max_score:.2f}")

    # 2. モデル、損失関数、オプティマイザの初期化
    logger.info("Initializing model...")
    model = Mini2048_SV_Predictor()

    # MSE Loss
    criterion = nn.MSELoss()
    logger.info("Using MSE Loss for regression")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # デバイスの設定
    device = torch.device(
        config.training.device if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    logger.info(f"Using device: {device}")

    # チェックポイントディレクトリの作成
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # チェックポイントから再開
    start_epoch = 0
    global_batch_count = 0
    checkpoint_files = sorted(
        checkpoint_dir.glob("model_epoch_*.pth"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        logger.info(f"Loading checkpoint from {latest_checkpoint}...")
        checkpoint = torch.load(
            latest_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        # グローバルバッチカウントを推定（概算）
        global_batch_count = start_epoch * len(train_loader)
        logger.info(f"Resuming from epoch {start_epoch}, batch {global_batch_count}")
    else:
        logger.info("No checkpoint found. Starting from scratch.")

    # 3. トレーニングの実行
    logger.info("Starting Supervised Learning Training...")

    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 順伝播
            outputs = model(inputs)

            # 損失の計算（直接MSE）
            loss = criterion(outputs, targets)  # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # メモリリーク防止：lossをitemで取得してグラフから切り離す
            loss_value = loss.item()
            total_loss += loss_value

            # 明示的にテンソルを削除
            del outputs, loss

            # グローバルバッチカウントを増加
            global_batch_count += 1

            # バッチごとのログ出力
            if (batch_idx + 1) % config.training.log_interval == 0:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                eval_score = get_initial_evaluation_score(model, device)
                logger.info(
                    f"Epoch [{epoch + 1}/{config.training.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss_value:.4f} "
                    f"Avg Loss: {avg_loss_so_far:.4f}"
                    f" Eval Score: {eval_score:.2f}"
                )
                metrics_logger.log_batch(
                    epoch + 1, batch_idx + 1, loss_value, avg_loss_so_far
                )

            # 1000バッチごとに評価
            if global_batch_count % 1000 == 0:
                logger.info(
                    f"[Batch {global_batch_count}] Evaluating model performance..."
                )
                avg_score, max_score_eval, min_score_eval = evaluate_model(
                    model, device, num_games=10
                )
                logger.info(
                    f"[Batch {global_batch_count}] Evaluation: Avg Score={avg_score:.2f}, Max={max_score_eval}, Min={min_score_eval}"
                )
                metrics_logger.log_evaluation(
                    epoch + 1, avg_score, max_score_eval, min_score_eval
                )

        # エポック終了時のログ
        avg_loss = total_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch + 1}/{config.training.epochs} completed. Avg Loss: {avg_loss:.4f}"
        )
        metrics_logger.log_epoch(epoch + 1, avg_loss)

        # チェックポイントの保存
        if (epoch + 1) % config.training.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "max_score": max_score,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    logger.info("Training finished.")

    # 最終モデルの保存
    final_model_path = "mini2048_sv_predictor.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")


if __name__ == "__main__":
    train()
