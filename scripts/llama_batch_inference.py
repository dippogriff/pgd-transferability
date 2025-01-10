# %%
%load_ext autoreload
%autoreload 2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleCNN, ResNet18, VitB16, ViT
from train import get_dataloader, simple_train
from eval import test, pgd_attack
from pathlib import Path
import seaborn as sns
from models import LlamaVision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# %%
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    
dataset = "CIFAR-10"
if dataset == "CIFAR-10":
    dataset = datasets.CIFAR10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
# %%
transform_list = [transforms.ToTensor()]
    
if transform_list:
    transform = transforms.Compose(transform_list)

# %%
data = dataset(root='./data', train=False, download=True, 
                transform=transform)

dataloader = DataLoader(data, batch_size=16, shuffle=False, num_workers=2) # shuffle if training

# %%
data2 = dataset(root='./data', train=False, download=True, 
                transform=None)

# %%
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch

# %%
# Load the model and processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct" 

# %%
processor = AutoProcessor.from_pretrained(model_id)

# %%
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# %%
classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

prompt = f"""
    Use exactly one word for the output. 
    Classify the image into one of the following labels: {", ".join(classes)}.
"""
    
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]}
]

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
# %%
# Prepare the text input
images = [*data2.data][:10]

#%%
o = [*next(iter(dataloader))[0]]
#%%
inputs = processor(o, 
                   [input_text]*len(o), 
                   add_special_tokens=False, 
                   return_tensors="pt", 
                   padding=True).to(dev)

            #%%
# Generate the output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
    
# %%
# Decode the output
generated_text = processor.batch_decode(outputs)
# %%
