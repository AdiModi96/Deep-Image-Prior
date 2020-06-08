import torch
import torch.nn as nn


class UNET_Heavy(nn.Module):
    input_channels = 1

    def __init__(self):
        super().__init__()

        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=UNET_Heavy.input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=192, out_channels=288, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=288),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=288, out_channels=288, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=288),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=288, out_channels=192, kernel_size=3, stride=2, padding=1, padding_mode='zeros', output_padding=1)
        )

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=3, stride=2, padding=1, padding_mode='zeros', output_padding=1)
        )

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=UNET_Heavy.input_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        )

    def __repr__(self):
        return 'UNET_Heavy'

    def forward(self, tensor):
        # Encoder
        encoder_block_1_tensor = self.encoder_block_1(tensor)
        encoder_block_2_tensor = self.encoder_block_2(encoder_block_1_tensor)

        # Bottleneck
        tensor = self.bottleneck(encoder_block_2_tensor)

        # Decoder
        tensor = self.decoder_block_2(torch.cat((tensor, encoder_block_2_tensor), dim=1))
        tensor = self.decoder_block_1(torch.cat((tensor, encoder_block_1_tensor), dim=1))

        return tensor


class UNET_Lite(nn.Module):
    input_channels = 1

    def __init__(self):
        super().__init__()

        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=UNET_Lite.input_channels, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=UNET_Lite.input_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        )

    def __repr__(self):
        return 'UNET_Lite'

    def forward(self, tensor):
        # Encoder
        encoder_block_1_tensor = self.encoder_block_1(tensor)
        encoder_block_2_tensor = self.encoder_block_2(encoder_block_1_tensor)

        # Bottleneck
        tensor = self.bottleneck(encoder_block_2_tensor)

        # Decoder
        tensor = self.decoder_block_2(torch.cat((tensor, encoder_block_2_tensor), dim=1))
        tensor = self.decoder_block_1(torch.cat((tensor, encoder_block_1_tensor), dim=1))

        return tensor


class FCNN(nn.Module):
    input_channels = 1

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=FCNN.input_channels, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),

            nn.Conv2d(in_channels=32, out_channels=FCNN.input_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
        )

    def __repr__(self):
        return 'FCNN'

    def forward(self, tensor):
        return self.decoder(self.encoder(tensor))