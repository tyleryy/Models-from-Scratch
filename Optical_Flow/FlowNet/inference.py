import torch
from FlowNetCorr import FlowNetCorrelation  # or your model class
from FlowNetS import FlowNetSimple

import numpy as np
import cv2
import torch
from torchvision.utils import flow_to_image

import os
from torchvision.datasets import Sintel
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch

# Prepare output directory
output_dir = "predicted_flows"
os.makedirs(output_dir, exist_ok=True)

# Model loading (as before)
# model = FlowNetCorrelation()
model = FlowNetSimple()


model.load_state_dict(torch.load('flownet_epoch_FlowNetS.pth', map_location='cuda'))
model.eval()
model = model.cuda()

# Dataset and DataLoader (as before)
img_transform = T.Compose([
    T.Resize((384, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
def sintel_transform(img1, img2, flow, valid_flow_mask=None):
    img1 = img_transform(img1)
    img2 = img_transform(img2)
    if flow is not None:
        flow = torch.from_numpy(flow).float()
    return img1, img2, flow, valid_flow_mask

dataset = Sintel(
    root='/pub/tyleryy/Models-from-Scratch/data',
    split='train',
    pass_name='clean',
    transforms=sintel_transform
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Inference and save color images
for idx, (img1, img2, flow_gt) in enumerate(dataloader):

    # FlowNetCorr
    # img1 = img1.cuda()
    # img2 = img2.cuda()

    # FlowNetS
    img = torch.concat((img1, img2), dim=1)
    img = img.cuda()


    with torch.no_grad():
        # pred_flow = model(img1, img2) # FlowNetCorr

        pred_flow = model(img) # FlowNetS

        if pred_flow.shape[-2:] != flow_gt.shape[-2:]:
            pred_flow = torch.nn.functional.interpolate(
                pred_flow, size=flow_gt.shape[-2:], mode='bilinear', align_corners=False
            )
        pred_flow = pred_flow.cpu() # (2, H, W)
        color_img = flow_to_image(pred_flow)    # (H, W, 3), uint8
        # print(color_img.shape)

        # Save as image
        out_path = os.path.join(output_dir, f"flow_{idx:05d}.png")
        # print(color_img[0].permute(1,2,0).shape)
        # print(color_img.dtype)
        cv2.imwrite(out_path, color_img[0].permute(1,2,0).numpy())
        print(f"Saved {out_path}")