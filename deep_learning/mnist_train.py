import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# 确保可视化目录存在
os.makedirs('./deep_learning/visualization', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 784         # 28*28像素展开
hidden_size = 128        # 隐藏层维度
num_classes = 10         # 0-9数字分类
learning_rate = 0.001    # 学习率
num_epochs = 10          # 训练轮数
batch_size = 64          # 批次大小

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./deep_learning/data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./deep_learning/data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class MNISTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = x.view(-1, input_size)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = MNISTClassifier(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 用于记录训练过程的指标
train_loss_list = []
train_acc_list = []
test_acc_list = []

print("开始训练MNIST模型...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    test_acc_list.append(test_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}], 平均损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%')

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_classes': num_classes
}, './deep_learning/model/mnist_model.pth')
print("\n模型已保存为: mnist_model.pth")

# 结果可视化 
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_loss_list, 'b-', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_acc_list, 'r-', label='Train Accuracy')
plt.plot(range(1, num_epochs+1), test_acc_list, 'g--', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training vs Test Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('./deep_learning/visualization/mnist_result.png') 
plt.close()

model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    plt.figure(figsize=(16, 2))
    for i in range(8):
        plt.subplot(1, 8, i+1)
        img = images[i].cpu().numpy().squeeze() * 0.3081 + 0.1307
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {predicted[i].item()}\nTrue: {labels[i].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./deep_learning/visualization/mnist_samples.png')
    plt.close()

print("可视化结果已保存至 ./deep_learning/visualization/")
print("训练完成")