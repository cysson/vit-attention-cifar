import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth=2, patch_size=(2, 2), num_heads=2, dropout=0.0):
        super(MobileViTBlock, self).__init__()
        self.patch_h, self.patch_w = patch_size
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)

        self.transformer = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True
            ) for _ in range(depth)
        ])

        self.conv2 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def unfolding(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_h == 0 and W % self.patch_w == 0
        x = x.view(B, C, H // self.patch_h, self.patch_h, W // self.patch_w, self.patch_w)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, H', W', ph, pw, C
        x = x.view(B, -1, self.patch_h * self.patch_w, C)  # B, N, P, C
        return x.view(B, -1, C)  # B, N*P, C

    def folding(self, x, H, W):
        B, _, C = x.shape
        x = x.view(B, H // self.patch_h, W // self.patch_w, self.patch_h, self.patch_w, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H, W)
        return x

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)

        B, C, H, W = y.shape
        tokens = self.unfolding(y)
        tokens = self.transformer(tokens)
        y = self.folding(tokens, H, W)

        out = torch.cat([x, y], dim=1)
        out = self.conv2(out)
        out = self.bn2(out)
        return out


class MobileViTResNet(nn.Module):
    def __init__(self, num_classes=10, base_channels=64):
        super(MobileViTResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.layer1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=1),
            MobileViTBlock(dim=base_channels * 2, depth=2, patch_size=(2, 2), num_heads=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def test():
    model = MobileViTResNet(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    test()
