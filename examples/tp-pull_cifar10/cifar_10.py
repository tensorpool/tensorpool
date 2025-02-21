import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision.utils import make_grid
import argparse


def main(num_epochs):
    # ---------------------------
    # 1. Set Up Environment
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create directories for outputs
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ---------------------------
    # 2. Define the CNN Model
    # ---------------------------
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)  # flatten
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleCNN().to(device)

    # ---------------------------
    # 3. Data Preparation
    # ---------------------------
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                               shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                              shuffle=False, num_workers=4)

    classes = train_dataset.classes

    # ---------------------------
    # 4. Training Setup
    # ---------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    log_list = []

    # ---------------------------
    # 5. Training and Evaluation Loop
    # ---------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Evaluation on the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Accuracy = {test_accuracy:.2f}%")

        # Log the metrics
        log_list.append({"epoch": epoch, "train_loss": avg_train_loss, "test_accuracy": test_accuracy})

        # Save a checkpoint of the model
        checkpoint_path = os.path.join("checkpoints", f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    # Save the training log to a CSV file
    log_df = pd.DataFrame(log_list)
    log_csv_path = os.path.join("outputs", "training_log.csv")
    log_df.to_csv(log_csv_path, index=False)
    print("Training log saved to", log_csv_path)

    # ---------------------------
    # 6. Plot Training Curves
    # ---------------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(log_df['epoch'], log_df['train_loss'], marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(log_df['epoch'], log_df['test_accuracy'], marker='o', color='green')
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()

    training_plot_path = os.path.join("outputs", "training_curves.png")
    plt.savefig(training_plot_path)
    plt.close()
    print("Training curves saved to", training_plot_path)

    # ---------------------------
    # 7. Generate and Save Sample Predictions
    # ---------------------------
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    images = images.cpu()

    def denormalize(img):
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.247, 0.243, 0.261])
        img = img.permute(1, 2, 0).numpy() * std + mean
        return np.clip(img, 0, 1)

    grid_images = []
    for idx in range(16):
        img = denormalize(images[idx])
        grid_images.append(torch.tensor(img).permute(2, 0, 1))

    grid = make_grid(torch.stack(grid_images), nrow=4, padding=2)
    np_grid = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid)
    plt.title("Sample Predictions (P: Predicted, A: Actual)")
    plt.axis('off')
    sample_pred_path = os.path.join("outputs", "sample_predictions.png")
    plt.savefig(sample_pred_path)
    plt.close()
    print("Sample predictions saved to", sample_pred_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN on CIFAR-10 with configurable epochs')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    main(args.epochs)
