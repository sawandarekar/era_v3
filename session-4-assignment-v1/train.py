import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from model import MNISTConvNet
import random
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')
if torch.cuda.is_available():
    logging.info(f'GPU Name: {torch.cuda.get_device_name(0)}')

def save_training_stats(epoch, train_loss, train_accuracy, test_accuracy):
    stats = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }
    with open('static/training_stats.json', 'a') as f:
        f.write(json.dumps(stats) + '\n')

def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def train():
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    logging.info('Loading MNIST dataset...')
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True)
    logging.info(f'Dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')

    # Initialize model and move to GPU if available
    model = MNISTConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    logging.info('Model initialized and moved to device')

    # Clear previous training stats
    if os.path.exists('static/training_stats.json'):
        os.remove('static/training_stats.json')

    # Training loop
    num_epochs = 10
    logging.info('Starting training...')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Create progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            train_accuracy = 100 * correct_train / total_train
            
            # Update progress bar description
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'train_acc': f'{train_accuracy:.2f}%'
            })
            
            if batch_idx % 20 == 19:  # Reduced frequency of evaluation
                # Calculate test accuracy
                test_accuracy = calculate_accuracy(model, test_loader)
                avg_loss = running_loss / 20
                
                # Save stats
                save_training_stats(epoch, avg_loss, train_accuracy, test_accuracy)
                running_loss = 0.0
                
                # Log progress
                logging.info(
                    f'Epoch: {epoch+1}, Batch: {batch_idx+1}, '
                    f'Loss: {avg_loss:.4f}, '
                    f'Train Accuracy: {train_accuracy:.2f}%, '
                    f'Test Accuracy: {test_accuracy:.2f}%'
                )
                
                model.train()

    logging.info('Training completed')
    
    # Final evaluation
    final_test_accuracy = calculate_accuracy(model, test_loader)
    logging.info(f'Final Test Accuracy: {final_test_accuracy:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    logging.info('Model saved to mnist_cnn.pth')

    # Generate and save sample predictions
    logging.info('Generating test samples...')
    model.eval()
    test_samples = []
    with torch.no_grad():
        for i in range(10):
            idx = random.randint(0, len(test_dataset) - 1)
            img, label = test_dataset[idx]
            img = img.unsqueeze(0).to(device)
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True).item()
            
            # Convert tensor to image
            plt.figure(figsize=(2, 2))
            plt.imshow(img.cpu().squeeze(), cmap='gray')
            plt.axis('off')
            
            # Save to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            test_samples.append({
                'image': img_str,
                'predicted': pred,
                'actual': label
            })

    # Save test samples
    with open('static/test_samples.json', 'w') as f:
        json.dump(test_samples, f)
    logging.info('Test samples saved')

if __name__ == '__main__':
    train()