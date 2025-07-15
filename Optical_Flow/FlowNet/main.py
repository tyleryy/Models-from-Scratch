# Loss function
# euclidean distance between predicted and gt flow for each pixel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
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

img_transform = T.Compose([
    # T.Resize((128, 128)),  # Resize to 128x128 for debugging
    T.RandomCrop((384, 512)),
    T.ToTensor(),          # Converts PIL Image to tensor and scales to [0,1]
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Optional: normalize to [-1, 1]
])

def sintel_transform(img1, img2, flow, valid_flow_mask=None):
    img1 = img_transform(img1)
    img2 = img_transform(img2)
    # flow is a numpy array (2, H, W); convert to tensor if needed
    if flow is not None:
        flow = torch.from_numpy(flow).float()
    return img1, img2, flow, valid_flow_mask


# Example usage:
dataset = Sintel(
    root='/pub/tyleryy/Models-from-Scratch/data',
    split='train',           # or 'test'
    pass_name='clean',       # or 'final', or 'both'
    transforms=sintel_transform
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

# model = FlowNetSimple()
model = FlowNetCorrelation()

# model.load_state_dict(torch.load('flownet_epoch100.pth', map_location='cuda'))

# Logs: first 100 epochs: 1e-4, no augmentation
# next 100 epochs: 1e-6, augmentation, loading from weights
model = model.cuda()  # if using GPU

optimizer = optim.Adam(model.parameters(), lr=1e-4) # 1e-4 changed after 100 epochs

num_epochs = 100  # or however many you want

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (img1, img2, flow_gt) in enumerate(dataloader):

        # FlowNetS
        # print(img1.shape, img2.shape)

        # img = torch.concat((img1, img2), dim=1)
        # img = img.cuda()

        # print(img.shape)
        
        # FlowNet Corr
        img1 = img1.cuda()
        img2 = img2.cuda()
        flow_gt = flow_gt.cuda()

        optimizer.zero_grad()
        # pred_flow = model(img) # S
        pred_flow = model(img1, img2) # C

        loss = endpoint_error(pred_flow, flow_gt)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"flownet_epoch_FlowNetC.pth")

torch.save(model.state_dict(), f"flownet_epoch_FlowNetC.pth")