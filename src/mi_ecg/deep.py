from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - allows importing package without torch.
    torch = None
    nn = None


if nn is not None:

    class ResidualBlock1D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1, dropout: float = 0.1):
            super().__init__()
            padding = kernel_size // 2
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels:
                self.skip = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                self.skip = nn.Identity()
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.skip(x))


    class ResNet1D(nn.Module):
        def __init__(self, in_channels: int = 12, base_channels: int = 64, dropout: float = 0.2):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, base_channels, kernel_size=15, padding=7, bias=False),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(
                ResidualBlock1D(base_channels, base_channels, stride=1, dropout=dropout),
                ResidualBlock1D(base_channels, base_channels * 2, stride=2, dropout=dropout),
                ResidualBlock1D(base_channels * 2, base_channels * 2, stride=1, dropout=dropout),
                ResidualBlock1D(base_channels * 2, base_channels * 4, stride=2, dropout=dropout),
                ResidualBlock1D(base_channels * 4, base_channels * 4, stride=1, dropout=dropout),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(base_channels * 4, 1),
            )

        def forward(self, x):
            return self.head(self.blocks(self.stem(x))).squeeze(-1)


    class InceptionBlock1D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, bottleneck: int = 32):
            super().__init__()
            mid = min(bottleneck, in_channels)
            self.bottleneck = nn.Conv1d(in_channels, mid, 1, bias=False)
            branch_channels = out_channels // 4
            self.branches = nn.ModuleList(
                [
                    nn.Conv1d(mid, branch_channels, k, padding=k // 2, bias=False)
                    for k in (9, 19, 39)
                ]
            )
            self.pool_branch = nn.Sequential(
                nn.MaxPool1d(3, stride=1, padding=1),
                nn.Conv1d(in_channels, branch_channels, 1, bias=False),
            )
            self.bn = nn.BatchNorm1d(branch_channels * 4)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            z = self.bottleneck(x)
            out = [branch(z) for branch in self.branches]
            out.append(self.pool_branch(x))
            return self.act(self.bn(torch.cat(out, dim=1)))


    class Inception1D(nn.Module):
        def __init__(self, in_channels: int = 12, channels: int = 128, depth: int = 5, dropout: float = 0.2):
            super().__init__()
            layers = []
            current = in_channels
            for _ in range(depth):
                layers.append(InceptionBlock1D(current, channels))
                current = channels
            self.encoder = nn.Sequential(*layers)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(channels, 1),
            )

        def forward(self, x):
            return self.head(self.encoder(x)).squeeze(-1)


    class ResidualBlock2D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            )
            if stride != 1 or in_channels != out_channels:
                self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            else:
                self.skip = nn.Identity()
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.skip(x))


    class Spectrogram2DNet(nn.Module):
        """Thesis-inspired STFT image CNN for wearable/limited-lead ECG."""

        def __init__(self, in_channels: int, base_channels: int = 32, dropout: float = 0.3):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 5, padding=2, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.blocks = nn.Sequential(
                ResidualBlock2D(base_channels, base_channels * 2, stride=2, dropout=dropout),
                ResidualBlock2D(base_channels * 2, base_channels * 4, stride=2, dropout=dropout),
                ResidualBlock2D(base_channels * 4, base_channels * 4, stride=1, dropout=dropout),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(base_channels * 4, 1),
            )

        def forward(self, x):
            return self.head(self.blocks(self.stem(x))).squeeze(-1)


def make_deep_model(name: str, in_channels: int = 12):
    if nn is None:
        raise ImportError("torch is required for deep models. Install with `pip install torch`.")
    if name == "resnet1d":
        return ResNet1D(in_channels=in_channels)
    if name == "inception1d":
        return Inception1D(in_channels=in_channels)
    if name == "spectrogram2d":
        return Spectrogram2DNet(in_channels=in_channels)
    raise ValueError(f"Unknown deep model: {name}")
