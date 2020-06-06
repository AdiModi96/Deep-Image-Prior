import torch
import torch.nn as nn


class UNET(nn.Module):

    input_channels = 1

    def __init__(self):
        super().__init__()

        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=UNET.input_channels, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=96),

            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=96),
        )

        self.encoder_block_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=192),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=192),
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=192, out_channels=288, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=288),

            nn.Conv2d(in_channels=288, out_channels=288, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=288),

            nn.ConvTranspose2d(in_channels=288, out_channels=192, kernel_size=3, stride=2, padding=1, padding_mode='zeros', output_padding=1)
        )

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=192),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=192),

            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=3, stride=2, padding=1, padding_mode='zeros', output_padding=1)
        )

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=96),

            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=96),

            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=48),

            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=16),

            nn.Conv2d(in_channels=16, out_channels=UNET.input_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        )


    def forward(self, tensor):
        # Encoder
        encoder_block_1_tensor = self.encoder_block_1(tensor)
        encoder_block_2_tensor = self.encoder_block_2(encoder_block_1_tensor)
        tensor = self.bottleneck(encoder_block_2_tensor)

        # Decoder
        tensor = self.decoder_block_2(torch.cat((tensor, encoder_block_2_tensor), dim=1))
        tensor = self.decoder_block_1(torch.cat((tensor, encoder_block_1_tensor), dim=1))

        return tensor