import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = [
            f for f in os.listdir(root_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    Train_dir = "./data/train"
    Test_dir = "./data/test"
    output_csv = "prediction.csv"

    train_dataset = datasets.ImageFolder(root=Train_dir)
    class_mapping = train_dataset.classes

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = TestDataset(root_dir=Test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    model1 = models.resnext50_32x4d(weights=None)
    model1.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model1.fc.in_features, 100))
    model1.load_state_dict(torch.load("./model1.pth", map_location=device))
    model1 = model1.to(device)
    model1.eval()

    model2 = models.resnext50_32x4d(weights=None)
    model2.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model2.fc.in_features, 100))
    model2.load_state_dict(torch.load("./model2.pth", map_location=device))
    model2 = model2.to(device)
    model2.eval()

    model3 = models.resnet50(weights=None)
    model3.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model3.fc.in_features, 100))
    model3.load_state_dict(torch.load("./model3.pth", map_location=device))
    model3 = model3.to(device)
    model3.eval()

    # Start test.
    results = []
    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            for images, img_names in tqdm(test_loader, desc="Predicting", leave=False):
                images = images.to(device)
                images_flipped = torch.flip(images, dims=[3])

                images_256 = TF.resize(images, [256, 256], antialias=True)
                images_256_flipped = torch.flip(images_256, dims=[3])

                images_288 = TF.resize(images, [288, 288], antialias=True)
                images_288_flipped = torch.flip(images_288, dims=[3])

                outputs1 = model1(images_256)
                outputs2 = model2(images_288)
                outputs3 = model3(images)

                # use the softmax finc
                prob1 = F.softmax(outputs1, dim=1)
                prob2 = F.softmax(outputs2, dim=1)
                prob3 = F.softmax(outputs3, dim=1)

                # flip the image and test again
                outputs1_f = model1(images_256_flipped)
                outputs2_f = model2(images_288_flipped)
                outputs3_f = model3(images_flipped)

                prob1_f = F.softmax(outputs1_f, dim=1)
                prob2_f = F.softmax(outputs2_f, dim=1)
                prob3_f = F.softmax(outputs3_f, dim=1)

                # different weight to get the mean score
                # w1, w2, w3 = 0.5, 0.3, 0.2
                # final_outputs = (
                #     outputs1 * w1 * 0.5 + outputs1_f * w1 * 0.5 +
                #    outputs2 * w2 * 0.5 + outputs2_f * w2 * 0.5 +
                #     outputs3 * w3 * 0.5 + outputs3_f * w3 * 0.5
                # )

                final_outputs = (
                    outputs1
                    + outputs2
                    + outputs3
                    + outputs1_f
                    + outputs2_f
                    + outputs3_f
                ) / 6.0
                # final_outputs = (prob1 + prob2 + prob3 + prob1_f + prob2_f + prob3_f) / 6.0

                # final_outputs = (
                #     (prob2 * 0.5 * 0.5) + (prob2_f * 0.5 * 0.5) +
                #     (prob3 * 0.5 * 0.5) + (prob3_f * 0.5 * 0.5)
                # )

                _, predicted = torch.max(final_outputs.data, 1)

                for i in range(len(img_names)):
                    clean_name = os.path.splitext(img_names[i])[0]
                    pred_idx = predicted[i].item()
                    real_label = class_mapping[pred_idx]

                    results.append({"image_name": clean_name, "pred_label": real_label})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
