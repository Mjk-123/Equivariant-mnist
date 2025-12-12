import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 데이터셋 & DataLoader 설정
pin_memory = torch.cuda.is_available()
tfm = transforms.ToTensor()
train_val = datasets.MNIST("./data", train=True, download=True, transform=tfm)
test_ds   = datasets.MNIST("./data", train=False, download=True, transform=tfm)

n_train = int(0.8 * len(train_val))
n_val   = len(train_val) - n_train
train_ds, val_ds = torch.utils.data.random_split(train_val, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=0, pin_memory=pin_memory, persistent_workers=False)

# 첫 배치 샘플 보기
X, y = next(iter(train_loader))
print(f"X shape: {X.shape}")      # torch.Size([128, 1, 28, 28])
print(f"y shape: {y.shape}")      # torch.Size([128])
print(f"First lablel: {y[0]}")
print("First image:")
print(X[0].squeeze())

# 실제 이미지 시각화 (5개)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 2))

i = 0
plt.subplot(1, 5, i + 1)
plt.imshow(X[i].squeeze(), cmap="gray")
plt.title(f"Label: {y[i].item()}")
plt.axis("off")
plt.show()
