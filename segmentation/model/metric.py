import torch

def IOUscore(model: torch.nn.Module, pred: torch.Tensor, target: torch.Tensor, device: str = 'cuda:0') -> torch.Tensor:
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
    pred = (torch.sigmoid(pred) > 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)

    pred_inds = pred == 1
    target_inds = target == 1
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection

    if union == 0:
        iou.append(float('nan'))  # If there is no ground truth, do not include in evaluation
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
