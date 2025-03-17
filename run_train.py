import numpy as np 
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from sklearn.model_selection import train_test_split

import csv
import tqdm


from model import SmallResNet, QResNet

ckpt_path = './best_resnet3.pth'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar_dir = "./cifar-10-python/cifar-10-batches-py"
if not os.path.exists(cifar_dir):
    raise FileNotFoundError(f"Dataset path {cifar_dir} not found. Ensure CIFAR-10 is downloaded.")

meta_data = unpickle(os.path.join(cifar_dir, 'batches.meta'))
label_names = [name.decode("utf-8") for name in meta_data[b'label_names']]
print("Labels:", label_names)

train_images, train_labels = [], []
for i in range(1, 6): 
    batch_dict = unpickle(os.path.join(cifar_dir, f"data_batch_{i}"))
    train_images.append(batch_dict[b'data'])
    train_labels.extend(batch_dict[b'labels'])

train_images = np.vstack(train_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
train_labels = np.array(train_labels)

# train_images, val_images, train_labels, val_labels = train_test_split(
#     train_images, train_labels, test_size=0.1, stratify=train_labels, random_state=42
# )

val_batch_file = os.path.join(cifar_dir, "test_batch")
val_dict = unpickle(val_batch_file)
val_images = val_dict[b'data']
val_images = val_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
val_labels = val_dict[b'labels']

print("Training dataset shape:", train_images.shape)
print("Validation dataset shape:", val_images.shape)

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.3),
    # transforms.RandomVerticalFlip(p=0.3),
    # transforms.RandomRotation(5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# transform_train = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CIFARDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = self.labels[idx]
        return image, label

train_dataset = CIFARDataset(train_images, train_labels, transform_train)
val_dataset = CIFARDataset(val_images, val_labels, transform_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = QResNet().to(device)

total_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")
print("Within limit:", total_params <= 5_000_000)


num_epochs = 100

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

best_acc = 0

for epoch in tqdm.tqdm(range(num_epochs)):
    resnet.train()
    correct_train, total_train = 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    scheduler.step()


    resnet.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet(inputs)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    print(f"\nEpoch [{epoch+1}/{num_epochs}], Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(resnet.state_dict(), ckpt_path)

