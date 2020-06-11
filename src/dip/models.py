import torch
import torch.nn as nn


class UNET_Heavy(nn.Module):
    input_channels = 1

    def __init__(self):
        super().__init__()

        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=UNET_Heavy.input_channels, out_channels=32, kernel_size=7, stride=1, padding=3, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=64),
        )

        self.encoder_block_2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=128),
        )

        self.encoder_block_3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=3, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=256),
        )

        self.encoder_block_4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=1, padding=3, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=512),
        )

        self.bottleneck = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=7, stride=1, padding=3, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=1024),

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=7, stride=1, padding=3, padding_mode='replicate', bias=True),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=512),

            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.decoder_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=512),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=256),

            nn.UpsamplingNearest2d(scale_factor=2),
        )

        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=256),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=128),

            nn.UpsamplingNearest2d(scale_factor=2),
        )

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=128),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=64),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=UNET_Heavy.input_channels, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
        )

    def forward(self, tensor):
        # Encoder
        encoder_block_1_output = self.encoder_block_1(tensor)
        encoder_block_2_output = self.encoder_block_2(encoder_block_1_output)
        encoder_block_3_output = self.encoder_block_3(encoder_block_2_output)
        encoder_block_4_output = self.encoder_block_4(encoder_block_3_output)
        # encoder_block_5_output = self.encoder_block_5(encoder_block_4_output)

        # Bottleneck
        tensor = self.bottleneck(encoder_block_4_output)

        # Decoder
        # tensor = self.decoder_block_5(torch.cat((tensor, encoder_block_4_output), dim=1))
        tensor = self.decoder_block_4(torch.cat((tensor, encoder_block_4_output), dim=1))
        tensor = self.decoder_block_3(torch.cat((tensor, encoder_block_3_output), dim=1))
        tensor = self.decoder_block_2(torch.cat((tensor, encoder_block_2_output), dim=1))
        tensor = self.decoder_block_1(torch.cat((tensor, encoder_block_1_output), dim=1))

        return tensor

    def __repr__(self):
        return 'UNET_Heavy'

