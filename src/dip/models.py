import torch
import torch.nn as nn


class UNET_D4(nn.Module):
    input_channels = 3
    convolutional_kernel_size = 3
    convolutional_padding = convolutional_kernel_size // 2

    def __init__(self):
        super().__init__()

        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=UNET_D4.input_channels, out_channels=64, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros', bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),
        )

        self.decoder_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros'),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros'),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
        )

        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
        )

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
        )

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=UNET_D4.input_channels, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode='zeros'),
        )

    def forward(self, tensor):
        # Encoder
        encoder_block_1_output = self.encoder_block_1(tensor)
        encoder_block_2_output = self.encoder_block_2(encoder_block_1_output)
        encoder_block_3_output = self.encoder_block_3(encoder_block_2_output)
        encoder_block_4_output = self.encoder_block_4(encoder_block_3_output)

        # Bottleneck
        tensor = self.bottleneck(encoder_block_4_output)

        # Decoder
        tensor = self.decoder_block_4(torch.cat((tensor, encoder_block_4_output), dim=1))
        tensor = self.decoder_block_3(torch.cat((tensor, encoder_block_3_output), dim=1))
        tensor = self.decoder_block_2(torch.cat((tensor, encoder_block_2_output), dim=1))
        tensor = self.decoder_block_1(torch.cat((tensor, encoder_block_1_output), dim=1))

        return tensor

    def __repr__(self):
        return 'UNET_D4'

