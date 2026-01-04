import torch
import torch.nn as nn
import numpy as np
import tkinter as tk
from tkinter import Label, Button, Canvas
from PIL import Image, ImageDraw
import os

def load_mnist_model():
    """加载预训练的MNIST模型"""
    model_path = './deep_learning/model/mnist_model.pth'
    if not os.path.exists(model_path):
        print("错误：未找到模型文件 mnist_model.pth！")
        print("请先运行 mnist_train.py 训练模型。")
        exit(1)
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu')) 
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_classes = checkpoint['num_classes']
    
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
    
    model = MNISTClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_mnist_model()
device = torch.device('cpu') 

class MNISTDrawingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST手写数字识别工具")
        self.root.geometry("500x450")
        
        self.canvas_width = 280 
        self.canvas_height = 280
        self.bg_color = "white"
        self.pen_color = "black"
        self.line_width = 15 
        
        self.canvas = Canvas(
            root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=self.bg_color,
            bd=2,
            relief="solid"
        )
        self.canvas.pack(pady=10)
        
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.predict)
        
        self.result_label = Label(root, text="识别结果: 未检测到数字", font=("Arial", 16))
        self.result_label.pack(pady=5)
        
        self.confidence_label = Label(root, text="置信度: 0%", font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        
        self.clear_btn = Button(
            root,
            text="清除画布",
            command=self.clear_canvas,
            font=("Arial", 12),
            width=10
        )
        self.clear_btn.pack(pady=10)
        
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
    
    def draw(self, event):
        """在画布上写字"""
        x1, y1 = (event.x - self.line_width//2), (event.y - self.line_width//2)
        x2, y2 = (event.x + self.line_width//2), (event.y + self.line_width//2)
        
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.pen_color, outline=self.pen_color)
        self.draw.ellipse([x1, y1, x2, y2], fill=0, outline=0)
    
    def center_digit(self, img):
        """将图像中的数字自动居中"""
        img_np = np.array(img)
        
        non_white_pixels = np.where(img_np < 255)
        if len(non_white_pixels[0]) == 0:
            return img
        
        min_row, max_row = np.min(non_white_pixels[0]), np.max(non_white_pixels[0])
        min_col, max_col = np.min(non_white_pixels[1]), np.max(non_white_pixels[1])
        
        digit_center_row = (min_row + max_row) // 2
        digit_center_col = (min_col + max_col) // 2
        
        img_center_row = img_np.shape[0] // 2
        img_center_col = img_np.shape[1] // 2
        
        row_offset = img_center_row - digit_center_row
        col_offset = img_center_col - digit_center_col
        
        new_img = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        new_draw = ImageDraw.Draw(new_img)
        
        for i in range(img_np.shape[0]):
            for j in range(img_np.shape[1]):
                if img_np[i, j] < 255:  
                    new_i = i + row_offset
                    new_j = j + col_offset
                    if 0 <= new_i < self.canvas_height and 0 <= new_j < self.canvas_width:
                        new_draw.point((new_j, new_i), fill=0)
        
        return new_img
    
    def preprocess_image(self):
        """预处理画布图像为模型输入格式"""
        centered_img = self.center_digit(self.image)
        img = centered_img.resize((28, 28), Image.Resampling.LANCZOS)
        img_np = np.array(img) / 255.0
        img_np = 1.0 - img_np
        img_np = (img_np - 0.1307) / 0.3081
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return img_tensor.to(device)
    
    def predict(self, event):
        """识别画布上的数字"""
        try:
            img_tensor = self.preprocess_image()
            
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                prob = torch.softmax(output, dim=1)[0][predicted.item()].item() * 100
            
            self.result_label.config(text=f"识别结果: {predicted.item()}")
            self.confidence_label.config(text=f"置信度: {prob:.2f}%")
        except Exception as e:
            self.result_label.config(text="识别失败: 请重新书写")
            self.confidence_label.config(text="置信度: 0%")
            print(f"预测错误: {e}")
    
    def clear_canvas(self):
        """清除画布所有内容"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="识别结果: 未检测到数字")
        self.confidence_label.config(text="置信度: 0%")

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTDrawingGUI(root)
    root.mainloop()