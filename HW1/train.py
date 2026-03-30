import torch
import torch.nn as nn
import torch.optim as optim  # As first use for optimizer but I use another
import time
import os
import csv
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

CONFIG = {
    "experiment_name": "Exp6_RexNet50_RandAug_224",
    "model_type": "resnet50",  # resnext50, resnet50
    "img_size": 224,
    "batch_size": 32,
    "epochs": 1,
    "learning_rate": 1e-3,
    "aug_strategy": "rand_augment",  # standard, rand_augment, "trivial"
    "save_dir": "./checkpoints",
    "resume_weight": "",  # If doing fine-tuning. Input the model
    "save_all_after_epoch": 60,
}

Train_dir = "./data/train"
Val_dir = "./data/val"

if __name__ == "__main__":
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # According to the aug_strategy to process the dataset
    if CONFIG["aug_strategy"] == "rand_augment":
        train_aug = transforms.Compose(
            [
                transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif CONFIG["aug_strategy"] == "trivial":
        train_aug = transforms.Compose(
            [
                transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        train_aug = transforms.Compose(
            [
                transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(root=Train_dir, transform=train_aug)
    val_dataset = datasets.ImageFolder(root=Val_dir, transform=val_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if CONFIG["model_type"] == "resnext50":
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    num_ftrs = (
        model.fc.in_features
    )  # model_fc: final connected layer, in_features: features counts
    model.fc = nn.Sequential(  # nn.Sequential: multiple processes
        nn.Dropout(p=0.5), nn.Linear(num_ftrs, 100)
    )

    if CONFIG["resume_weight"] and os.path.exists(CONFIG["resume_weight"]):
        model.load_state_dict(torch.load(CONFIG["resume_weight"], map_location=device))
    elif CONFIG["resume_weight"]:
        print("wrong path.")
        exit()
    model = model.to(device)

    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()

    # Prepare the csv file and the scoreboard for the record.
    Scoreboard = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0

    csv_filename = f"{CONFIG['experiment_name']}_training_log.csv"
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Epoch", "Train_Loss", "Train_Acc(%)", "Val_Loss", "Val_Acc(%)"]
        )

    print(f"Start training: ...")
    for epoch in range(CONFIG["epochs"]):
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        with tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=False
        ) as train_bar:
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += (
                    loss.item()
                )  # .item() take the value from tensor, sum the total loss
                _, predicted = torch.max(
                    outputs.data, 1
                )  # Take the answer for current cal.
                total += labels.size(0)
                correct += (predicted == labels).sum().item()  # If correct, correct++

        train_acc = 100 * correct / total  # Cal the accurancy.
        train_loss = running_loss / len(train_loader)  # Average loss.

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        end_time = time.time()
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{CONFIG['epochs']}] {end_time - start_time:.0f}s | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )

        Scoreboard["train_loss"].append(train_loss)
        Scoreboard["train_acc"].append(train_acc)
        Scoreboard["val_loss"].append(val_loss)
        Scoreboard["val_acc"].append(val_acc)

        with open(csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    f"{train_loss:.4f}",
                    f"{train_acc:.2f}",
                    f"{val_loss:.4f}",
                    f"{val_acc:.2f}",
                ]
            )

        # Save model if good performance.
        acc_str = f"{val_acc:.2f}".replace(".", "_")
        loss_str = f"{val_loss:.4f}".replace(".", "_")

        save_path = os.path.join(
            CONFIG["save_dir"],
            f"{CONFIG['experiment_name']}_ep{epoch+1}_acc{acc_str}_loss{loss_str}.pth",
        )

        saved = False
        # New highest Acc socre
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Save model: {save_path}")
            saved = True
        # New Lowest loss score
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not saved:
                torch.save(model.state_dict(), save_path)
                print(f"Save Model: {save_path}")
                saved = True
        # After some epoch, save all model.
        if (epoch + 1) >= CONFIG["save_all_after_epoch"] and not saved:
            if not saved:
                torch.save(model.state_dict(), save_path)
                print(f"Save Model: {save_path}")

    # Draw the plot
    epochs_range = range(1, CONFIG["epochs"] + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)

    plt.plot(epochs_range, Scoreboard["train_acc"], label="Train_Acc")
    plt.plot(epochs_range, Scoreboard["val_acc"], label="Val Acc")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, Scoreboard["train_loss"], label="Train Loss")
    plt.plot(epochs_range, Scoreboard["val_loss"], label="Val Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plot_path = f"{CONFIG['experiment_name']}_learning_curve.png"
    plt.savefig(plot_path)
    print("Program end. ")
