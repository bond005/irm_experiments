from typing import Any

import numpy as np
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader


def test_model(model: torch.nn.Module, device: Any, test_loader: DataLoader,
               set_name: str = 'test set', sklearn_report: bool = False):
    model.eval()
    test_loss = 0
    correct = 0
    reference = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).long()
            logits = model(data)
            test_loss += torch.nn.functional.cross_entropy(logits, target, reduction='sum').item()  # sum up batch loss
            pred = torch.argmax(logits, dim=-1)
            reference.append(target.cpu().numpy())
            predictions.append(pred.cpu().numpy())
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

    if sklearn_report:
        print('')
        print(classification_report(y_true=np.concatenate(reference), y_pred=np.concatenate(predictions), digits=4))
    else:
        print('\nPerformance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            set_name, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        ))

    return 100. * correct / len(test_loader.dataset)
