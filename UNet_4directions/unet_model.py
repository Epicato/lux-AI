
""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,global_length,bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.global_length = global_length

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)

        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024+global_length, 512 // factor, bilinear)
        self.up2 = Up(512+global_length, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_global):
        x1 = self.inc(x)    #output : 32*32*64
        x2 = self.down1(x1)     #output : 16*16*128
        x3 = self.down2(x2)     #output : 8*8*256
        x4 = self.down3(x3)     #output : 4*4*512
        # x5 = self.down4(x4)     #output : 2*2*512
        x4 = torch.cat([x4, x_global], dim=1)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
