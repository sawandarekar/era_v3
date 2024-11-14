import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class ConvNet(nn.Module):
    def __init__(self, filters, kernels):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, filters[0], kernels[0], padding=kernels[0]//2)
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernels[1], padding=kernels[1]//2)
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernels[2], padding=kernels[2]//2)
        self.conv4 = nn.Conv2d(filters[2], filters[3], kernels[3], padding=kernels[3]//2)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(filters[3] * 1 * 1, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train_model(filters, kernels, optimizer_name, batch_size, epochs, log_queue=None, model_name="Model"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = ConvNet(filters, kernels).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{epochs}')
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            current_loss = running_loss/(batch_idx+1)
            current_acc = 100.*correct/total
            
            if log_queue and batch_idx % 5 == 0:  # Update every 5 batches
                log_queue.put({
                    'model': model_name,
                    'epoch': epoch + 1,
                    'progress': batch_idx / len(train_loader) * 100,
                    'train_loss': current_loss,
                    'train_acc': current_acc,
                    'timestamp': datetime.now().isoformat()
                })
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100.*correct/total
        
        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss = test_loss/len(test_loader)
        test_acc = 100.*correct/total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if log_queue:
            log_queue.put({
                'model': model_name,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': test_loss,
                'val_acc': test_acc,
                'final_epoch': True,
                'timestamp': datetime.now().isoformat()
            })
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }

def evaluate_random_samples(model, n_samples=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
    indices = np.random.choice(len(test_dataset), n_samples, replace=False)
    
    results = []
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            pred = output.argmax(dim=1).item()
            
            results.append({
                'image': image.cpu().numpy(),
                'true_label': classes[label],
                'predicted_label': classes[pred],
                'correct': label == pred
            })
    
    return results 