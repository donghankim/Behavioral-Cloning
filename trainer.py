import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os, pdb, time

class Trainer():
    def __init__(self, model, dataset):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.map_location=self.device
        else:
            self.device = torch.device('cpu')
            self.map_location= self.device

        self.model = model.to(self.device)
        print(f"Cuda Available: {next(self.model.parameters()).is_cuda}")

        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=50, shuffle=True)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.epochs = 25

        self.WEIGHT_PATH = 'model_weights/'

    def train(self, model_name):
        model_path = os.path.join(self.WEIGHT_PATH, model_name)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            print("No path exists. Training from scratch.")
            model_path = os.path.join(self.WEIGHT_PATH, 'base.pth')

        loss_hist = []
        start_time = time.time()
        for epoch in tqdm(range(self.epochs)):
            for imgs, labels in self.data_loader:
                x, y = imgs.to(self.device), labels.to(self.device)
                y = torch.unsqueeze(y, 0).view(-1, 1)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                loss_hist.append(loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"loss for epoch {epoch+1}: {loss[-1]}")

        end_time = time.time()
        torch.save(self.model.state_dict(), model_path)
        print(f"Training complete. Time taken: {round(end_time - start_time)//60} minuets")


    def val(model, dataset):
        pass



