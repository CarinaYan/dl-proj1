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

from model import SmallResNet, QResNet
from sklearn.metrics import confusion_matrix
import seaborn as sns

ckpt_path = "best_resnet3-acc90.7.pth"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# resnet = QResNet().to(device)

best_model = QResNet().to(device)
best_model.load_state_dict(torch.load(ckpt_path))
best_model.eval()

print(f"Model loaded successfully from {ckpt_path}")


# test_batch_file = os.path.join(cifar_dir, "test_batch")
# test_dict = unpickle('cifar-10-python/cifar-10-batches-py/test_batch')
# test_images = test_dict[b'data']
# test_ids = test_dict[b'ids']
# test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)

cifar_dir = "./cifar-10-python/cifar-10-batches-py"
val_batch_file = os.path.join(cifar_dir, "test_batch")
val_dict = unpickle(val_batch_file)
val_images = val_dict[b'data']
val_images = val_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
val_labels = val_dict[b'labels']



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

val_dataset = CIFARDataset(val_images, val_labels, transform_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)#, num_workers=2

# with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = resnet(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total_val += labels.size(0)
#             correct_val += (predicted == labels).sum().item()

correct_val = 0
total_val = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

accuracy = correct_val / total_val
conf_matrix = confusion_matrix(all_labels, all_predictions)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

