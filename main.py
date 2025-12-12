import numpy as np
import torch
from torch import nn, optim    
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

from models.mlp import MLPClassifier, mlp
from models.simple_cnn import SimpleCNN, simpleCNN
from models.p4m_cnn import P4M_EquivariantCNN, P4M_Block, p4m_cnn

from typing import Optional, List, Iterable, List
import os
import wandb
import tqdm

SEED = 42
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed_all(SEED) if torch.cuda.is_available() else None
cudnn.benchmark = True

pin_memory = (device.type == "cuda")
tfm = transforms.ToTensor()
train_val = datasets.MNIST("./data", train=True, download=True, transform=tfm)
test_ds   = datasets.MNIST("./data", train=False, download=True, transform=tfm)

n_train = int(0.8 * len(train_val))
n_val   = len(train_val) - n_train
train_ds, val_ds = torch.utils.data.random_split(train_val, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=0, pin_memory=pin_memory, persistent_workers=False)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False,
                          num_workers=0, pin_memory=pin_memory, persistent_workers=False)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False,
                          num_workers=0, pin_memory=pin_memory, persistent_workers=False)



# model = mlp().to(device)
# model = simpleCNN().to(device)
model = p4m_cnn().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = F.cross_entropy

num_epochs = 50
batch_size = 128


# -------------------------
# Training and Evaluation Functions
# -------------------------
def train(model, loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    total, correct = 0, 0

    for batch_idx, (data, target) in enumerate(loader):
        # Flatten 
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)                  # logits [B,10]
        loss = loss_fn(output, target)  # cross-entropy loss
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        running_loss += float(loss) * batch_size
        total += batch_size
        correct += (output.argmax(dim=1) == target).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total
    return train_loss, train_acc

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    val_loss = 0
    correct = 0
    for data, target in tqdm.tqdm(loader, desc="Eval", leave=False):
        data, target = data.to(device), target.to(device)
        output = model(data)
        val_loss += loss_fn(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return val_loss / len(loader.dataset), correct / len(loader.dataset)

@torch.no_grad()
def test(model, loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in tqdm.tqdm(loader, desc="Test", leave=False):
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += loss_fn(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return test_loss / len(loader.dataset), correct / len(loader.dataset)

'''
import sys
from PIL import Image, ImageOps

def load_and_preprocess(img_path, invert=False):
    """
    - Reads an image from <img_path>, resizes it to 28x28, converts to grayscale.
    - Normalizes pixel values to [0, 1].
    - Returns a tensor of shape [1, 1, 28, 28].
    """
    img = Image.open(img_path).convert("L")          # grayscale
    img = img.resize((28, 28), Image.BILINEAR)
    if invert:
        img = ImageOps.invert(img)                   # 배경/글자색 반전 옵션
    x = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))).float()
    x = x.view(28, 28) / 255.0                       # [H,W] in [0,1]
    x = x.unsqueeze(0).unsqueeze(0)                  # [1,1,28,28]
    return x.to(device)
'''

@torch.no_grad()
def classify(model, x):
    """
    Classifies a single image tensor x.
    - x: [1, 1, 28, 28]
    - Returns (predicted class, confidence, probabilities).
    """
    x = x.to(device)
    output = model(x.view(1, -1))                   # [1, 10] logits
    probs = torch.softmax(output, dim=-1).squeeze(0)
    pred = int(probs.argmax().item())                # predicted class
    conf = float(probs.max().item())                 # confidence
    return pred, conf, probs.cpu().numpy()

'''
def print_help():
    print(
"""Commands:
  <path>           classify image at <path>
  invert <path>    classify with inversion (white digit on black vs black on white)
  quit/exit        leave
Tips:
  - Images can be of any size. They will be resized to 28x28 grayscale.
  - If the background and digit colors are reversed, try the 'invert' command.
"""
    )

def shell():
    print("MNIST classifier shell (type 'help' for commands)")
    while True:
        try:
            cmd = input("classify> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not cmd:
            continue
        if cmd.lower() in {"quit", "exit"}:
            break
        if cmd.lower() == "help":
            print_help()
            continue

        # Check for invert command
        if cmd.startswith("invert "):
            img_path = cmd.split(maxsplit=1)[1]
            x = load_and_preprocess(img_path, invert=True)
        else:
            img_path = cmd
            x = load_and_preprocess(img_path)

        pred, conf, probs = classify(model2, x)
        print(f"Predicted: {pred}, Confidence: {conf:.4f}")
        print(f"Probabilities: {probs}")
'''

if __name__ == "__main__":
    wandb.init(
        project = "mlp-classifier",
        name = "p4m-equivariant-cnn",
        config = {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "channel": [32, 64, 128],
            "activation": "relu",
            "dropout": 0.0,
            "use_batchnorm": True,
            "final_bias": 0.0
        }
    )
    cfg = wandb.config

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc":  val_acc
        })
        print("Epoch {}: Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}".format(
            epoch, train_loss, train_acc, val_loss, val_acc
        ))

        if val_acc > 0.99:  # Early stopping condition
            print("Early stopping at epoch {}".format(epoch))
            break
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
        
    test_loss, test_acc = test(model, test_loader, loss_fn)
    wandb.log({
        "test_loss": test_loss,
        "test_acc": test_acc
    })
    print("Test Loss: {:.4f}, Test Acc: {:.4f}".format(test_loss, test_acc))
    wandb.finish()

    '''
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as e: 
        pass
    shell()  # Start the interactive shell
    print("Exiting MNIST classifier shell.")
    '''
    

