# Loss function
# euclidean distance between predicted and gt flow for each pixel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torch.optim as optim

from FlowNetCorr import FlowNetCorrelation
from FlowNetS import FlowNetSimple

from torchvision.datasets import Sintel

def endpoint_error(pred_flow, gt_flow):
    """
    pred_flow: (B, 2, H, W)
    gt_flow: (B, 2, H, W)
    Returns: scalar loss (mean EPE over all pixels and batch)
    """
    # pred_flow: (B, 2, H_pred, W_pred)
    # flow_gt:   (B, 2, H_gt, W_gt)
    if pred_flow.shape[-2:] != flow_gt.shape[-2:]:
        pred_flow = F.interpolate(pred_flow, size=flow_gt.shape[-2:], mode='bilinear', align_corners=False)
    return torch.norm(pred_flow - gt_flow, dim=1).mean()

# ! custom transform functions on flow to sync with augmentation on images
class SynchronizedRandomCrop:
    def __init__(self, size):
        self.size = size  # (height, width)

    def __call__(self, img1, img2, flow, valid_flow_mask=None):
        # img1, img2: PIL Images
        # flow: numpy array (2, H, W) or torch tensor (2, H, W)
        i, j, h, w = T.RandomCrop.get_params(img1, output_size=self.size)
        img1 = TF.crop(img1, i, j, h, w)
        img2 = TF.crop(img2, i, j, h, w)
        if flow is not None:
            if isinstance(flow, np.ndarray):
                flow = torch.from_numpy(flow)
            flow = flow[:, i:i+h, j:j+w]
        return img1, img2, flow, valid_flow_mask



sync_crop = SynchronizedRandomCrop((192, 256))

def sintel_transform(img1, img2, flow, valid_flow_mask=None):
    img1 = T.Resize((384, 512))(img1)
    img2 = T.Resize((384, 512))(img2)
    img1, img2, flow, valid_flow_mask = sync_crop(img1, img2, flow, valid_flow_mask)
    img1 = TF.to_tensor(img1)
    img2 = TF.to_tensor(img2)
    img1 = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img1)
    img2 = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img2)
    return img1, img2, flow, valid_flow_mask


# Example usage:
dataset = Sintel(
    root='/pub/tyleryy/Models-from-Scratch/data',
    split='train',           # or 'test'
    pass_name='clean',       # or 'final', or 'both'
    transforms=sintel_transform
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

# ! model selection
model = FlowNetSimple()
# model = FlowNetCorrelation()

# Logs: first 100 epochs: 1e-4, no augmentation
# next 100 epochs: 1e-6, augmentation, loading from weights
model = model.cuda()  # if using GPU

optimizer = optim.Adam(model.parameters(), lr=1e-4) #
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# LR will drop by 10x every 100 epochs
num_epochs = 300

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (img1, img2, flow_gt) in enumerate(dataloader):

        # ! FlowNetS
        img = torch.concat((img1, img2), dim=1)
        img = img.cuda()

        
        # ! FlowNet Corr
        # img1 = img1.cuda()
        # img2 = img2.cuda()

        flow_gt = flow_gt.cuda()

        optimizer.zero_grad()
        pred_flow = model(img) # S
        # pred_flow = model(img1, img2) # C

        loss = endpoint_error(pred_flow, flow_gt)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Step the scheduler at the end of the epoch
    scheduler.step()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

# ! change model checkpoints' names
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"flownet_epoch_FlowNetS.pth")

torch.save(model.state_dict(), f"flownet_epoch_FlowNetS.pth")