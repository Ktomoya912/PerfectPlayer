import torch
import torch.nn as nn


class Mini2048_SV_Predictor(nn.Module):
    def __init__(self):
        super(Mini2048_SV_Predictor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(11, 64, kernel_size=1),  # 3x3 -> 3x3
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=2, padding=1
            ),  # 3x3 -> 4x4
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=232, kernel_size=2),  # 4x4 -> 3x3
            nn.ReLU(),
            nn.Conv2d(in_channels=232, out_channels=256, kernel_size=2),  # 3x3 -> 2x2
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2),  # 2x2 -> 1x1
            nn.ReLU(),
        )
        # Conv層の出力: 256 channels × 1 × 1 = 256
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor):
        # 入力を (Batch, Channels, Height, Width) の形式に整形
        x = x.view(-1, 11, 3, 3)

        x = self.conv_layers(x)

        # Flatten（バッチ次元以外を平坦化）
        x = x.view(x.size(0), -1)

        x = self.fc_layers(x)

        # スカラー値を返す
        return x.squeeze(-1)
