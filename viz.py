import numpy as np 
import pandas as pd
import pickle
import torch


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# ...existing code...
test_dict = unpickle('cifar_test_nolabel.pkl')
test_images = test_dict[b'data']
test_ids = test_dict[b'ids']


Labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# 打印labels和对应的数字
print("Labels:")
for i, label in enumerate(Labels):
    print(f"{i}: {label}")

# 可视化前20张图像
import matplotlib.pyplot as plt

# 创建一个figure
fig, axes = plt.subplots(4, 5, figsize=(15, 10))

# 可视化前20张图像
for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i])
    ax.set_title(f"Image {i}")
    ax.axis('off')

plt.tight_layout()
plt.show()
# ...existing code...
