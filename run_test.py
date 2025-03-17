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
test_dict = unpickle('cifar_test_nolabel.pkl')
test_images = test_dict[b'data']
test_ids = test_dict[b'ids']
# test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TestDataset(Dataset):
    def __init__(self, images, ids, transform):
        self.images = images
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        test_id = self.ids[idx]
        return image, test_id

test_dataset = TestDataset(test_images,test_ids, transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



predictions = []
ids = []
with torch.no_grad():
    for images, idx in test_loader:
        images = images.to(device)
        outputs = best_model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        ids.extend(idx.cpu().numpy())

submission_df = pd.DataFrame({"ID": ids, "Labels": predictions})
submission_df.to_csv("submission_4.csv", index=False)

print("saved to submission_4.csv")
