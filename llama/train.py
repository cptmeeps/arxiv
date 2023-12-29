import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

def train(model, data_loader, epochs, learning_rate, checkpoint_interval, checkpoint_path):
  # Example usage: train(model, data_loader, epochs=10, learning_rate=0.001, checkpoint_interval=5, checkpoint_path='/path/to/checkpoints')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=learning_rate)

  model.train()
  for epoch in range(epochs):
    for batch in data_loader:
      inputs, targets = batch
      inputs, targets = inputs.to(device), targets.to(device)

      outputs = model(inputs)
      loss = criterion(outputs, targets)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

    if (epoch + 1) % checkpoint_interval == 0:
      checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
      }
      checkpoint_filename = f'{checkpoint_path}/checkpoint_epoch_{epoch+1}.pth'
      torch.save(checkpoint, checkpoint_filename)

def fine_tune(model, data_loader, epochs, learning_rate, checkpoint_path):
  # Example usage: fine_tune(model, data_loader, epochs=5, learning_rate=0.0001, checkpoint_path='/path/to/checkpoint.pth')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=learning_rate)

  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate

  model.train()
  for epoch in range(epochs):
    for batch in data_loader:
      inputs, targets = batch
      inputs, targets = inputs.to(device), targets.to(device)

      outputs = model(inputs)
      loss = criterion(outputs, targets)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

