from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.datasets import ColoredMNIST
from src.models.models import MultilayerPerceptronForMNIST, ConvNetForMNIST, init_weights
from src.training.evaluate import test_model


def erm_train(model: torch.nn.Module, device: Any, train_loader: DataLoader,
              optimizer: torch.optim.Optimizer, epoch: int):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        logits = model(data)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))


def train_and_test_erm(data_dir: str, model_type: str, num_epochs: int, training_batch_size: int, eval_batch_size: int):
    possible_types = {'mlp', 'convnet'}
    if model_type not in possible_types:
        raise ValueError(f'The {model_type} is unknown model type! Possible types are: {possible_types}.')

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda:0')
        print('CUDA is used!')
    else:
        device = torch.device('cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    all_train_loader = DataLoader(
        ColoredMNIST(
            root=data_dir,
            env='all_train',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
            ])
        ),
        batch_size=training_batch_size, shuffle=True, **kwargs
    )

    test_loader = DataLoader(
        ColoredMNIST(
            root=data_dir, env='test', transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
            ])
        ),
        batch_size=eval_batch_size, shuffle=True, **kwargs
    )

    if model_type == 'convnet':
        model = ConvNetForMNIST().to(device)
    else:
        model = MultilayerPerceptronForMNIST().to(device)
    model.apply(init_weights)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, num_epochs + 1):
        erm_train(model, device, all_train_loader, optimizer, epoch)
        test_model(model, device, all_train_loader, set_name='train set')
        test_model(model, device, test_loader)
    test_model(model, device, test_loader, sklearn_report=True)
