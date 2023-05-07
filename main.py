import warnings

warnings.filterwarnings("ignore")

import os
import sys
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


from utility import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "model/ImageColorizationModel.pth"


model = None
if not os.path.exists(model_path):
    print("Model not find")
    download_from_drive()
    print("Model Downloaded")
else:
    model = load_model_with_cpu(model_class=MainModel, file_path=model_path)
    print("Model Loaded")


def predict_and_return_image(image):
    if image is None:
        return None
    data = create_lab_tensors(image)
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    fake_color = model.fake_color.detach()
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    return fake_imgs[0]


import gradio as gr

title = "Black&White to Color image"
description = "Transforming Black & White Image in to colored image. Upload a black and white image to see it colorized by our deep learning model."

gr.Interface(
    fn=predict_and_return_image,
    title=title,
    description=description,
    inputs=[gr.Image(label="Gray Scale Image")],
    outputs=[gr.Image(label="Predicted Colored Image")],
).launch()
