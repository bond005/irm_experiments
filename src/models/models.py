import torch


class MultilayerPerceptronForMNIST(torch.nn.Module):
    def __init__(self):
        super(MultilayerPerceptronForMNIST, self).__init__()
        self.fc1 = torch.nn.Linear(3 * 28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 2)

    def forward(self, x):
        x = x.view(-1, 3 * 28 * 28)
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        logits = self.fc3(x).view(-1, 2)
        return logits


class ConvNetForMNIST(torch.nn.Module):
    def __init__(self):
        super(ConvNetForMNIST, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 5, 1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 256)
        self.fc2 = torch.nn.Linear(256, 2)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4 * 4 * 50)
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        logits = self.fc2(x).view(-1, 2)
        return logits


def init_weights(m: torch.nn.Module):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(tensor=m.weight, nonlinearity='leaky_relu')
        m.bias.data.fill_(0.0)
