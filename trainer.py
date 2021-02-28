import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
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

    def train(self, model_weight):
        model_path = os.path.join(self.WEIGHT_PATH, model_weight)
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

            print(f"loss for epoch {epoch+1}: {loss_hist[-1]}")

        end_time = time.time()
        torch.save(self.model.state_dict(), model_path)
        print(f"Training complete. Time taken: {round(end_time - start_time)//60} minuets")


    def val(self, dataset, model_weight):
        pass

    def test(self, dataset, model_weight):
        data_loader = DataLoader(dataset, batch_size = 50)
        model_path = os.path.join(self.WEIGHT_PATH, model_weight)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            print("No paths exists. Testing failed.")
            return

        predictions = []
        cnt = 0
        test_loss = 0
        self.model.eval()
        with torch.no_grad():
            print("Making predictions...")
            for images, labels in tqdm(data_loader):
                x, y = images.to(self.device), labels.to(self.device)
                y = torch.unsqueeze(y, 0).view(-1, 1)
                y_hat = self.model(x)
                for i in range(len(y_hat)):
                    predictions.append(y_hat[i].item())

                test_loss += mean_squared_error(y_hat.cpu(), y.cpu())
                cnt += 1

        print(f"Test Loss: {round(test_loss/cnt, 2)}")
        pdb.set_trace()




