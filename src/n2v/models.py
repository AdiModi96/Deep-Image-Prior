import torch
import torch.nn as nn


class UNET_D4(nn.Module):
    input_channels = 3
    convolutional_kernel_size = 3
    convolutional_padding = convolutional_kernel_size // 2
    convolutional_padding_mode = 'circular'

    def __init__(self):
        super().__init__()

        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=UNET_D4.input_channels, out_channels=64, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode,
                      bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding, padding_mode=UNET_D4.convolutional_padding_mode,
                      bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),
        )

        self.decoder_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
        )

        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
        )

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
        )

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=UNET_D4.input_channels, kernel_size=UNET_D4.convolutional_kernel_size, stride=1, padding=UNET_D4.convolutional_padding,
                      padding_mode=UNET_D4.convolutional_padding_mode),
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


class UNET_D2(nn.Module):
    input_channels = 3
    convolutional_kernel_size = 3
    convolutional_padding = convolutional_kernel_size // 2
    convolutional_padding_mode = 'circular'

    def __init__(self):
        super().__init__()

        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=UNET_D2.input_channels, out_channels=96, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
        )

        self.encoder_block_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=2, stride=2),
        )

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=2, stride=2),
        )

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=96, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=UNET_D2.input_channels, kernel_size=UNET_D2.convolutional_kernel_size, stride=1, padding=UNET_D2.convolutional_padding, padding_mode=UNET_D2.convolutional_padding_mode),
        )

    def forward(self, tensor):
        # Encoder
        encoder_block_1_output = self.encoder_block_1(tensor)
        encoder_block_2_output = self.encoder_block_2(encoder_block_1_output)

        # Bottleneck
        tensor = self.bottleneck(encoder_block_2_output)

        # Decoder
        tensor = self.decoder_block_2(torch.cat((tensor, encoder_block_2_output), dim=1))
        tensor = self.decoder_block_1(torch.cat((tensor, encoder_block_1_output), dim=1))

        return tensor
