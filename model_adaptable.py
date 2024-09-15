
import torch

class DoubleConv(torch.nn.Module):
    """
    Helper Class which implements the intermediate Convolutions
    """
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())
        
    def forward(self, X):
        return self.step(X)
    
class DownConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv = DoubleConv(in_channels, out_channels)

        if self.pooling:
            self.pool = torch.nn.MaxPool3d(2)

    def forward(self, X):
        x = self.conv(X)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool
    
class UpConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear")

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, X, before_pool):
        x = self.upsample(X)
        x = torch.cat([x, before_pool], dim=1)
        x = self.conv(x)
        return x
    



class UNet3D_adaptable(torch.nn.Module):
    """
    This class implements a UNet for the Segmentation
    We use 3 down- and 3 UpConvolutions and two Convolutions in each step
    """

    def __init__(self, num_classes=8, in_channels=1, depth=5, start_filts=64):
        """Sets up the U-Net Structure
        """
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.depth = depth
        self.start_filts = start_filts

        self.down_convs = []
        self.up_convs = []

        #Aqui vamos a crear las capas del encoder y meterlas en la lista
        for i in range(self.depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < self.depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        #Aqui vamos a crear las capas del decoder y meterlas en la lista
        for i in range(self.depth -1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins + outs, outs)
            self.up_convs.append(up_conv)
        
        self.down_convs = torch.nn.ModuleList(self.down_convs)
        self.up_convs = torch.nn.ModuleList(self.up_convs)

        #Ahora los sobrantes
        self.conv_final = torch.nn.Conv3d(outs, self.num_classes, 1)

    def forward(self, x):
        encoder_outs = []

        #Primero el forward del encoder
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        #Ahora el forward del decoder
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(x, before_pool)

        x = self.conv_final(x)

        return x