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

# ! I think that I "random crop" to 384, 512 getting a "zoomed in" image as train input
# ! But in inference, I "resize" to 384, 512 which is the whole image
# ! might be a mismatch between training and inference images resulting in noisy output
input_height, input_width = 384, 512

# Model loading (as before)
# model = FlowNetCorrelation()
model = FlowNetSimple()


model.load_state_dict(torch.load('flownet_epoch_FlowNetS.pth', map_location='cuda'))
model.eval()
model = model.cuda()

# Dataset and DataLoader (as before)
img_transform = T.Compose([
    T.Resize((input_height, input_width)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def sintel_transform_test(img1, img2, flow=None, valid_flow_mask=None):
    img1 = img_transform(img1)
    img2 = img_transform(img2)
    return img1, img2, flow, valid_flow_mask

dataset = Sintel(
    root='/pub/tyleryy/Models-from-Scratch/data',
    split='test',
    pass_name='clean',
    transforms=sintel_transform_test
)

def sintel_test_collate(batch):
    # Handles both (img1, img2, flow) and (img1, img2, flow, valid_flow_mask)
    if len(batch[0]) == 4:
        img1s, img2s, flows, valid_flow_masks = zip(*batch)
    elif len(batch[0]) == 3:
        img1s, img2s, flows = zip(*batch)
    else:
        raise ValueError("Unexpected number of elements in batch")
    img1s = torch.stack(img1s)
    img2s = torch.stack(img2s)
    return img1s, img2s

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=sintel_test_collate)

for idx, (img1, img2) in enumerate(dataloader):

    # FlowNetCorr
    # img1 = img1.cuda()
    # img2 = img2.cuda()

    # FlowNetS
    img = torch.concat((img1, img2), dim=1)
    img = img.cuda()


    with torch.no_grad():
        # pred_flow = model(img1, img2) # FlowNetCorr

        pred_flow = model(img) # FlowNetS

        pred_flow = pred_flow.cpu() # (2, H, W)
        color_img = flow_to_image(pred_flow)    # (H, W, 3), uint8
        # print(color_img.shape)

        # Save as image
        out_path = os.path.join(output_dir, f"flow_{idx:05d}.png")

        cv2.imwrite(out_path, color_img[0].permute(1,2,0).numpy())
        print(f"Saved {out_path}")