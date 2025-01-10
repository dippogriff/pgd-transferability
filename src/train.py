import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def get_dataloader(dataset_name, batch_size=32, train=True, norm=True, 
                   resize=None, stratified_sample=None):
    if dataset_name == "CIFAR-10":
        dataset = datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.243, 0.261)
    
    transform_list = []   
                    
    if resize is not None:
        transform_list.append(transforms.Resize(resize))
        
    transform_list.append(transforms.ToTensor())
    
    if norm:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
    if transform_list:
        transform = transforms.Compose(transform_list)
    
    data = dataset(root='./data', train=train, download=True, 
                   transform=transform)
    
    targets = data.targets
    if stratified_sample is not None:
        _, subset_idxs = train_test_split(np.arange(len(targets)),
                                          test_size=stratified_sample,
                                          random_state=123,
                                          shuffle=True,
                                          stratify=targets)
        data = Subset(data, subset_idxs)
    
    dataloader = DataLoader(data, batch_size=batch_size, 
                            shuffle=train, num_workers=2) # shuffle if training
    
    return dataloader

def simple_train(trainloader, model, dev, save_path, lr=0.01, epochs=5):
    criterion_a = nn.CrossEntropyLoss()
    optimizer_a = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer_a)
        
    for epoch in range(epochs): 
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            
            optimizer_a.zero_grad()
            outputs = model(inputs)
            loss = criterion_a(outputs, labels)
            
            # backprop and adjust
            loss.backward()
            optimizer_a.step()

            running_loss += loss.item()
            
            if i % 2000 == 1999: # print stats every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] '
                      'loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        lr_decay.step(running_loss)
    
    torch.save(model.state_dict(), save_path)