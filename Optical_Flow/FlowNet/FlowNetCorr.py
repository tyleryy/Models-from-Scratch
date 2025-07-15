import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrelationLayer(nn.Module):
    def __init__(self, max_displacement=20):
        """
        Correlation layer for FlowNetC
        
        Args:
            max_displacement: maximum pixel displacement to consider
        """
        super(CorrelationLayer, self).__init__()
        self.max_displacement = max_displacement

    def forward(self, x1, x2):
        """
        Compute correlation between two feature maps
        
        Args:
            x1: first feature map (B, C, H, W)
            x2: second feature map (B, C, H, W)
            
        Returns:
            correlation volume (B, (2*max_displacement+1)^2, H, W)
        """

        # Get dimensions
        batch_size, channels, height, width = x1.size()

        # Pad x2 by max_displacement to handle boundary effects
        # replicate = copy edge values
        pad_size = self.max_displacement
        x2_padded = F.pad(x2, (pad_size, pad_size, pad_size, pad_size), mode='replicate')

        # Init correlation volume
        correlation_size = (2 * self.max_displacement + 1) ** 2
        correlation = torch.zeros(batch_size, correlation_size, height, width, device=x1.device)

        # For every pixel (i, j), iterate over search window 
        # (comparing every pixels in x1 to a window of pixels in x2)
        
        for i in range(height):
            for j in range(width):
                # Extract window from x2_padded
                window = x2_padded[:, :, 
                                 i:i + 2 * self.max_displacement + 1, 
                                 j:j + 2 * self.max_displacement + 1]
                
                # Compute correlation with pixel (i,j) from x1
                pixel_features = x1[:, :, i:i+1, j:j+1] # (B, C, 1, 1)

                # Reshape for broadcasting
                pixel_features = pixel_features.view(batch_size, channels, 1, 1)

                # Compute dot product (correlation)
                corr = torch.sum(pixel_features * window, dim=1) # (B, H_window, W_window)
                # sums along channel dimension

                # Flatten the correlation window
                corr_flat = corr.view(batch_size, -1) # (B, (2*max_displacement+1)^2)

                # Store in correlation volume
                correlation[:, :, i, j] = corr_flat 
                # each pixel has correlation dot product values with each pixel in window
                # stored along sort of the channel dim

        return correlation

class FlowNetCorrelation(nn.Module):

    def __init__(self, max_displacement=10, redir_dim=32):
        super(FlowNetCorrelation, self).__init__()

        # ! d = 20 in paper, but it should be d = 10 for the equation to satisfy and result in 441 correlation channels

        self.correlation_dim = (max_displacement * 2 + 1)**2
        self.conv_redir = redir_dim
        
        # 1. Feature extractors (same architecture for both images)
        # but seperately
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)

        # self.correlation_dim + self.conv_redir = 441 + 32 = 473 (in original paper)
        self.conv3_1 = nn.Conv2d(self.correlation_dim + self.conv_redir, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # add 32 from conv_redir
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        # Correlation layer / cost volume
        self.corr = CorrelationLayer(max_displacement=max_displacement)
        self.conv_redir = nn.Conv2d(256, 32, kernel_size=1, stride=1) # applied conv on image 1 that is concatenated to correlation layer
        self.corr_conv = nn.Conv2d(1681, 256, kernel_size=3, stride=1, padding=1)

        # Decoder layers
        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.predict_flow6 = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)

        # Input: 1026 channels
        # After upsampling, you concatenate:
        # - The upsampled features from the previous deconv (512 channels from deconv5)
        # - The skip connection from the encoder (conv5_1, 512 channels)
        # - The upsampled flow prediction from the previous scale (2 channels)
        # 512 + 512 + 2 = 1026 channels
        self.deconv4 = nn.ConvTranspose2d(1026, 256, kernel_size=4, stride=2, padding=1)
        self.predict_flow5 = nn.Conv2d(1026, 2, kernel_size=3, stride=1, padding=1)

        self.deconv3 = nn.ConvTranspose2d(770, 128, kernel_size=4, stride=2, padding=1)
        self.predict_flow4 = nn.Conv2d(770, 2, kernel_size=3, stride=1, padding=1)

        self.deconv2 = nn.ConvTranspose2d(386, 64, kernel_size=4, stride=2, padding=1)
        self.predict_flow3 = nn.Conv2d(386, 2, kernel_size=3, stride=1, padding=1)

        self.predict_flow2 = nn.Conv2d(194, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        # Extract features from both images (same architecture)
        out_conv1_1 = F.relu(self.conv1(x1))
        out_conv2_1 = F.relu(self.conv2(out_conv1_1))
        out_conv3_1 = F.relu(F.relu(self.conv3(out_conv2_1)))
        
        # Same for second image
        out_conv1_2 = F.relu(self.conv1(x2))
        out_conv2_2 = F.relu(self.conv2(out_conv1_2))
        out_conv3_2 = F.relu(F.relu(self.conv3(out_conv2_2)))

        # apply at the conv3 level
        correlation = self.corr(out_conv3_1, out_conv3_2)  # (B, 1681, H, W) if max_disp=20
        # (20 * 2 + 1)^2 = 1681 (correlation dim)

        # concat correlation_layer w/ conv_redir
        redir = F.relu(self.conv_redir(out_conv3_1))  # (B, redir_dim, H, W)
        concat = torch.cat([correlation, redir], dim=1) # (B, corr_dim + redir_dim, H, W)
        in_conv3_1 = self.conv3_1(concat) # (B, 256, H, W)

        # Decoder (upsampling and flow prediction at multiple scales)
        # The following is a typical decoder structure, similar to FlowNetS:
        out_conv4 = F.relu(self.conv4(in_conv3_1))
        out_conv4_1 = F.relu(self.conv4_1(out_conv4))
        out_conv5 = F.relu(self.conv5(out_conv4_1))
        out_conv5_1 = F.relu(self.conv5_1(out_conv5))
        out_conv6 = F.relu(self.conv6(out_conv5_1))
        out_conv6_1 = F.relu(self.conv6_1(out_conv6))

        # Predict flow at the coarsest level
        flow6 = self.predict_flow6(out_conv6_1)
        flow6_up = F.interpolate(flow6, scale_factor=2, mode='bilinear', align_corners=False)
        deconv5 = F.relu(self.deconv5(out_conv6_1))
        concat5 = torch.cat([deconv5, out_conv5_1, flow6_up], dim=1)
        flow5 = self.predict_flow5(concat5)
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
        concat2 = torch.cat([deconv2, out_conv2_1, flow3_up], dim=1)
        flow2 = self.predict_flow2(concat2)

        # Return the finest flow prediction
        return flow2
