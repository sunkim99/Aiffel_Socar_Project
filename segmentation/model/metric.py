import torch

def IOUscore(model: torch.nn.Module, outputs: torch.Tensor, labels: torch.Tensor, device: str = 'cuda:0') -> torch.Tensor:
    """
    Description
     : A function to calculate IOU score.

    Parameters
     : outputs : predictions
     : labels : targets

    Return
     : IOU score
    """
    outputs = (torch.sigmoid(outputs.float()) > 0.5).float()
    outputs= outputs.squeeze(1)
    labels = labels.squeeze(1)

    intersection = torch.logical_and(outputs, labels).float()
    union = torch.logical_or(outputs, labels).float()

    return ((intersection.sum((1,2)) + 1e-6) / (union.sum((1,2)) + 1e-6)).mean()


def PixelAccuracy(model: torch.nn.Module, outputs: torch.Tensor, labels: torch.Tensor, device: str = 'cuda:0') -> torch.Tensor:
    """
    Description
     : A function to calculate pixel accuracy.

    Parameters
     : outputs : predictions
     : labels : targets

    Return
     : pixel accuracy
    """
    outputs = (torch.sigmoid(outputs.float()) > 0.5).float()
    
    outputs = outputs.view(-1)
    labels = labels.view(-1)

    num_correct = (outputs == labels).sum()
    num_pixels = torch.numel(labels)

    return num_correct / num_pixels
