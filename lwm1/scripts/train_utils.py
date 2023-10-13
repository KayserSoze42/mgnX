import torch

def train_one_epoch(model, optimizer, train_loader, criterion, device, scheduler=None):
    
    model.train()
    total_loss = 0

    for images in train_loader:

        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backwards()
        optimizer.step()
        
        total_loss += loss.item()

    if scheduler:

        scheduler.step() # steady steps, idk

    return total_loss / len(train_loader)

def validate_model(model, val_loader, criterion, device):
    
    model.eval()
    total_loss = 0

    with torch.no_grad():

        for images in val_loader:

            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()

        return total_loss / len(val_loader)

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):

    checkpoint = {

            "epoch": epoch,
            "model_state_dict": model_state_dict(),
            "optimizer_state_dict": optimizer_state_dict(),

    }

    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    return epoch

