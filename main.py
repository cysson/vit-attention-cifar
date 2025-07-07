# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import yaml
import os
import wandb

from models.resnet import ResNet18, ResNet50
from models.resnet_mobilevit import MobileViTResNet
from utils.progress_bar import progress_bar


def get_model(config):
    if config["model"] == "resnet18":
        return ResNet18(
            num_classes=config["num_classes"],
            attention=config.get("attention", None)
        )
    elif config["model"] == "resnet50":
        return ResNet50(
            num_classes=config["num_classes"],
            attention=config.get("attention", None)
        )
    elif config["model"] == "mobilevit":
        from models.resnet_mobilevit import MobileViTResNet
        return MobileViTResNet(
            num_classes=config["num_classes"],
            base_channels=config.get("base_channels", 64)
        )
    else:
        raise ValueError("Unknown model type")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/baseline/mobilevit.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化wandb
    wandb_cfg = config.get("wandb", {})
    wandb_run = wandb.init(
        project=wandb_cfg.get("project", "default_project"),
        name=wandb_cfg.get("run_name", None),
        config=config,
        entity=wandb_cfg.get("entity", None),
        reinit=True
    )

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.get("batch_size", 128), shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    print('==> Building model..')
    net = get_model(config).to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def train(epoch):
        print('\nEpoch:', epoch)
        net.train()
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        
        # 记录wandb训练数据
        wandb.log({
            "train_loss": train_loss / len(trainloader),
            "train_accuracy": 100. * correct / total,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr']
        })

    def test(epoch):
        net.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # 记录wandb测试数据
        wandb.log({
            "test_loss": test_loss / len(testloader),
            "test_accuracy": 100. * correct / total,
            "epoch": epoch,
        })

    for epoch in range(0, config.get("epochs", 100)):
        train(epoch)
        test(epoch)
        scheduler.step()

    # 保存模型权重（可选）
    model_save_path = wandb_cfg.get("model_save_path", None)
    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)
        save_path = os.path.join(model_save_path, f"model_epoch_{epoch}.pth")
        torch.save(net.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        wandb.save(save_path)


if __name__ == '__main__':
    main()
