import torch
import torch.nn as nn
    
class IndexDecoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(IndexDecoder, self).__init__()
        #print(input_channels, output_channels)8 128 [10/08 18:54:29]
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class XyzMLP(nn.Module):
    def __init__(self, D=4, W=128, 
                 in_channels_xyz=63, out_channels_xyz=8):
        super(XyzMLP, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.out_channels_xyz = out_channels_xyz
        print(self.D, self.W, self.in_channels_xyz, self.out_channels_xyz)
        #4 96 63 8
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i}", layer)
        #self.skip = nn.Linear(in_channels_xyz + W, W)
        self.skip = nn.Linear(W, W)
        #self.xyz_encoding_final_1 = nn.Linear(W, 64)
        self.xyz_encoding_final = nn.Linear(W, out_channels_xyz)
        self.xyz_encoding_alpha = nn.Linear(W, 3)
        
    def forward(self, x, alpha):
        #print(x.shape)torch.Size([54275, 3])
        input_points = x
        for i in range(self.D):
            x = getattr(self, f"xyz_encoding_{i}")(x)
        #x = torch.cat([input_points, x], -1)
        x = self.skip(x)
        x = torch.relu(x)
        if alpha == True:
            alpha_encoding_final = self.xyz_encoding_alpha(x)
            return alpha_encoding_final
        xyz_encoding_final = self.xyz_encoding_final(x)
        #x = torch.relu(x)
        #xyz_encoding_final = self.xyz_encoding_final_2(x)
        #xyz_encoding_final = self.xyz_encoding_alpha(xyz_encoding_final)

        return xyz_encoding_final

'''
class FushionModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FushionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels * 2, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=output_channels, kernel_size=1)

    def forward(self, feature, xyzs):
        x = torch.cat([feature, xyzs], dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x
'''