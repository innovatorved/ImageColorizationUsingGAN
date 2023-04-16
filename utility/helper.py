import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import requests
import gdown

SIZE = 256


def download_from_drive():
    url = "https://drive.google.com/uc?id=1EhuMET76c02VFyRW8Pie7BwNCDHmQiad"
    try:
        output = "model/ImageColorizationModel.pth"
        gdown.download(url, output, quiet=False)
        return True
    except:
        print("Error Occured in Downloading model from Gdrive")
        return False


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {
        "loss_D_fake": loss_D_fake,
        "loss_D_real": loss_D_real,
        "loss_D": loss_D,
        "loss_G_GAN": loss_G_GAN,
        "loss_G_L1": loss_G_L1,
        "loss_G": loss_G,
    }


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap="gray")
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")


def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")


def create_lab_tensors(image):
    """
    This function receives an image path or a direct image input and creates a dictionary of L and ab tensors.
    Args:
    - image: either a path to the image file or a direct image input.
    Returns:
    - lab_dict: dictionary containing the L and ab tensors.
    """
    if isinstance(image, str):
        # Open the image and convert it to RGB format
        img = Image.open(image).convert("RGB")
    else:
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        img = img.convert("RGB")

    custom_transforms = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE), Image.BICUBIC),
            transforms.RandomHorizontalFlip(),  # A little data augmentation!
        ]
    )
    img = custom_transforms(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
    L = L.unsqueeze(0)
    ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1
    return {"L": L, "ab": ab}


def predict_and_visualize_single_image(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    fake_color = model.fake_color.detach()
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(L[0][0].cpu(), cmap="gray")
    axs[0].set_title("Grey Image")
    axs[0].axis("off")

    axs[1].imshow(fake_imgs[0])
    axs[1].set_title("Colored Image")
    axs[1].axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")


def predict_color(model, image, save=False):
    """
    This function receives an image path or a direct image input and creates a dictionary of L and ab tensors.
    Args:
    - model : Pytorch Gray Scale to Colorization Model
    - image: either a path to the image file or a direct image input.
    """
    data = create_lab_tensors(image)
    predict_and_visualize_single_image(model, data, save)


def load_model_with_cpu(model_class, file_path):
    """
    Load PyTorch model from file.

    Args:
        model_class (torch.nn.Module): PyTorch model class to load.
        file_path (str): File path to load the model from.

    Returns:
        model (torch.nn.Module): Loaded PyTorch model.
    """
    model = model_class()
    model.load_state_dict(torch.load(file_path, map_location=torch.device("cpu")))
    return model


def load_model_with_gpu(model_class, file_path):
    """
    Load PyTorch model from file.

    Args:
        model_class (torch.nn.Module): PyTorch model class to load.
        file_path (str): File path to load the model from.

    Returns:
        model (torch.nn.Module): Loaded PyTorch model.
    """
    model = model_class()
    model.load_state_dict(torch.load(file_path))
    return model
