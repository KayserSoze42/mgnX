import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import LWM1Dataset, data_transforms
from model import LWM1CNN
from train_utils import train, validate

# yaml me up, sci.py
import yaml

with open("config/settings.yaml", "r") as j8ream:
    config = yaml.safe_load(j8ream)

# DS p

train_data_path = config["train_data_path"]
val_data_path = config["val_data_path"]

# L DS

train_datasets = LWM1Dataset(train_data_path, transform=data_transforms["train"])
val_datasets = LWM1Dataset(val_data_path, transform=data_transform["val"])

# L DL

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_siz"], shuffle=True)

# L mo && I mo

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LWM1CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# TLoop

num_epochs = config[epochs]

for epoch in range(num_epochs):

    train_loss = train(model, train_loader, optimizer, device)
    val_loss, val_accuracy = validate(model, val_loader, device)

    print(f"Epoch {epoch}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}, Validation accuracy: {val_accuracy}%")

print("Training completed!")

# STM

torch.save(model.state_dict(), config["model_save_path"])

