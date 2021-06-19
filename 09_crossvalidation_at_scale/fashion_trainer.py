"""
Credit to: https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split
import optuna
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

class FashionCNN(nn.Module):
    
    def __init__(self,features,dropout):
        super(FashionCNN, self).__init__()
        self.features = features
        self.dropout = dropout

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(self.dropout)
        self.fc2 = nn.Linear(in_features=600, out_features=self.features)
        self.fc3 = nn.Linear(in_features=self.features, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

def train_and_test(lr,batch_size,features,dropout):
    train_set = FashionMNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = FashionMNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))


    # validation set
    # split data in training and validation set
    train_n = int(0.80 * len(train_set))
    val_n = len(train_set) - train_n
    train_data, val_data = random_split(train_set, [train_n, val_n])

    train_loader = DataLoader(train_data, batch_size= batch_size)
    val_loader = torch.utils.data.DataLoader( val_data, batch_size= batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    model = FashionCNN(features,dropout)
    model.to(device)
    
    error = nn.CrossEntropyLoss()

    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    print(model)
    
    num_epochs = 1
    # Lists for visualization of loss and accuracy 
    loss_list = []
    iteration_list = []
    accuracy_list = []
    
    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []
    
    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            # Forward pass 
            outputs = model(images)
            loss = error(outputs, labels)
            
            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()
            
            #Propagating the error backward
            loss.backward()
            
            # Optimizing the parameters
            optimizer.step()
                
            # Testing the model
            count = epoch * len(train_loader) + batch_idx
            if not (count % 50):    # It's same as "if count % 50 == 0"
                total = 0
                correct = 0
            
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)
                
                    outputs = model(images)
                
                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()
                
                    total += len(labels)
                
                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
            
            if not (count % 500):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
    return np.mean(accuracy_list)
            
                
    # class_correct = [0. for _ in range(10)]
    # total_correct = [0. for _ in range(10)]
    
    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         predicted = torch.max(outputs, 1)[1]
    #         c = (predicted == labels).squeeze()
            
    #         for i in range(100):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             total_correct[label] += 1
            
    # for i in range(10):
    #     print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))

def objective(trial):
    lr = trial.suggest_discrete_uniform("lr", 1e-6, 1e0,0.1) #optuna 1
    features = trial.suggest_int("features", 120, 600, 60) #optuna 2
    batch_size = trial.suggest_int("batch_size", 8, 128, 10) #optuna 3
    dropout = trial.suggest_discrete_uniform("dropout", 0.2, 0.5, 0.1) #optuna 4

    train_model =  train_and_test(lr,batch_size,features,dropout)

    return train_model


        
if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    study = optuna.create_study(direction="maximize",pruner = pruner)
    study.optimize(objective, n_trials=3) 
    study.best_trial
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study) ## this is important to figure out which hp is important
    optuna.visualization.plot_slice(study)   ## this gives a clear picture
    optuna.visualization.plot_parallel_coordinate(study)




