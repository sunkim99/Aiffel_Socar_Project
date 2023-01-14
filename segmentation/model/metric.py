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

    iou = []
    outputs = (torch.sigmoid(outputs.float()) > 0.5).float()
    
    for idx in range(outputs.shape[0]):
        output = outputs[idx]
        target = labels[idx]

        intersection = torch.logical_and(output, target).float().sum((1,2))
        union = torch.logical_or(output, target).float().sum((1,2))

        if union == 0:
            pass
        else:
            iou.append(float(intersection) / float(max(union, 1)))

    return sum(iou)


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

    return (num_correct / num_pixels) * 100
