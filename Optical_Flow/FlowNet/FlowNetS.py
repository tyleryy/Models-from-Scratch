import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowNetSimple(nn.Module):
    """
    FlowNetSimple (FlowNetS) model for optical flow estimation.
    """
    def __init__(self):
        super(FlowNetSimple, self).__init__()

        # encoder layers

        # input channels = 6 (stacking two RGB images)
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        # decoder layers
        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.predict_flow6 = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)

        # Input: 1026 channels.
        # After upsampling, you concatenate:
        # The upsampled features from the previous deconv (512 channels from deconv5)
        # The skip connection from the encoder (conv5_1, 512 channels)
        # The upsampled flow prediction from the previous scale (2 channels)
        # 512 (deconv5) + 512 (conv5_1) + 2 (upsampled flow5) = 1026 channels

        # eq. 
        # upsampled layer (low res view) 
        # + skip connection (earlier high res view) 
        # + upsampled previous optical flow pred. for refinement
        self.deconv4 = nn.ConvTranspose2d(1026, 256, kernel_size=4, stride=2, padding=1)
        self.predict_flow5 = nn.Conv2d(1026, 2, kernel_size=3, stride=1, padding=1)
        
        self.deconv3 = nn.ConvTranspose2d(770, 128, kernel_size=4, stride=2, padding=1)
        self.predict_flow4 = nn.Conv2d(770, 2, kernel_size=3, stride=1, padding=1)
        
        self.deconv2 = nn.ConvTranspose2d(386, 64, kernel_size=4, stride=2, padding=1)
        self.predict_flow3 = nn.Conv2d(386, 2, kernel_size=3, stride=1, padding=1)

        self.predict_flow2 = nn.Conv2d(194, 2, kernel_size=3, stride=1, padding=1)

        
    
    def forward(self, x):
         # Encoder: extract features at multiple scales
        out_conv1 = F.relu(self.conv1(x))         # [B, 64, H/2, W/2]
        out_conv2 = F.relu(self.conv2(out_conv1)) # [B, 128, H/4, W/4]
        out_conv3 = F.relu(self.conv3(out_conv2)) # [B, 256, H/8, W/8]
        out_conv3_1 = F.relu(self.conv3_1(out_conv3))
        out_conv4 = F.relu(self.conv4(out_conv3_1))   # [B, 512, H/16, W/16]
        out_conv4_1 = F.relu(self.conv4_1(out_conv4))
        out_conv5 = F.relu(self.conv5(out_conv4_1))   # [B, 512, H/32, W/32]
        out_conv5_1 = F.relu(self.conv5_1(out_conv5))
        out_conv6 = F.relu(self.conv6(out_conv5_1))   # [B, 1024, H/64, W/64]
        out_conv6_1 = F.relu(self.conv6_1(out_conv6))

        # early high res layers capture smaller, fine movement
        # later low res layers capture large movement
        # used for skip connections in decoder (like U-net)

        # Decoder
        # upsampled, concatenate, predict flow, move on to next higher resolution
        # Deconvolution (coarse) + Skip connection (fine) + prev. flow prediction (refinement)
        

        flow6 = self.predict_flow6(out_conv6_1)
        
        # Upsample flow by 2 (each conv layer divided image dims by 2)
        flow6_up = F.interpolate(flow6, scale_factor=2, mode='bilinear', align_corners=False)

        # Deconvolution conv6_1 to conv5
        deconv5 = F.relu(self.deconv5(out_conv6_1))
        
        # Concat and predict next upsampled flow
        concat5 = torch.cat([deconv5, out_conv5_1, flow6_up], dim=1) # should all match now
        flow5 = self.predict_flow5(concat5)
        
        # upsample and continue
        flow5_up = F.interpolate(flow5, scale_factor=2, mode='bilinear', align_corners=False)
        deconv4 = F.relu(self.deconv4(concat5))

        concat4 = torch.cat([deconv4, out_conv4_1, flow5_up], dim=1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=False)
        deconv3 = F.relu(self.deconv3(concat4))

        concat3 = torch.cat([deconv3, out_conv3_1, flow4_up], dim=1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=False)
        deconv2 = F.relu(self.deconv2(concat3))

        concat2 = torch.cat([deconv2, out_conv2, flow3_up], dim=1)
        flow2 = self.predict_flow2(concat2)

        return flow2