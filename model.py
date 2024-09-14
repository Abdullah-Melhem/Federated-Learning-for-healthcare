import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# Note the model and functions here defined do not have any FL-specific components.



class Net_LSTM(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=95, hidden_size=8, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(8, 256)  # Adjust input to match flattened LSTM output
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape x to (batch_size, input_shape, 1)
        x = x.unsqueeze(-2)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)  # Flatten the LSTM output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class Net_GRU(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net_GRU, self).__init__()
        self.lstm = nn.GRU(input_size=95, hidden_size=8, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(8, 256)  # Adjust input to match flattened LSTM output
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape x to (batch_size, input_shape, 1)
        x = x.unsqueeze(-2)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)  # Flatten the LSTM output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class Net(nn.Module):
    """A simple feedforward neural network suitable for tabular data."""

    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(95, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in tqdm(range(epochs)):
        for record, labels in trainloader:
            record, labels = record.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(record), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            record, labels = data[0].to(device), data[1].to(device)
            outputs = net(record)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
