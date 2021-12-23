import torch
import torch.nn as nn
import torch.nn.functional as F
#import PixelUnShuffle

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel=64, stride=1):
        super(ResidualBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(65, 64, kernel_size=3, stride=stride, padding=1, bias=False,dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=5, bias=False,dilation=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )
        self.block5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False,dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


        )





    def forward(self, x):
        y = x

        x = self.block1(x)

        x1_1 = self.block2(x)
        x1_2 = self.block2(x)

        x2_1 = self.block3(x1_1)
        x2_2 = self.block3(x1_1)
        x2_3 = self.block3(x1_2)
        x2_4 = self.block3(x1_2)


        x3_1 = self.block4(torch.add(x2_1,x2_2))
        x3_2 = self.block4(torch.add(x2_3,x2_4))
        x4_1 = self.block5(torch.add(x3_1,x3_2))



        return x+x4_1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inchannel = 64
        self.inchannel_1 = 64

        self.layer1_1 = ResidualBlock(64,64, stride=1)
        self.layer1_2 = ResidualBlock(64,64, stride=1)
        self.layer1_3 = ResidualBlock(64,64, stride=1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )

        self.conv5 = nn.Sequential(

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),

        )

        self.gate = nn.Conv2d(64 * 3, 3, 3, 1, 1, bias=True)


    def forward(self, x):   #batchSize * c * k*w * k*h   128*1*40*40
        y = x
        out = self.conv1(x)
        y1 = self.layer1_1(torch.cat((out, x), dim=1))
        y2 = self.layer1_2(torch.cat((y1, x), dim=1))
        y3 = self.layer1_3(torch.cat((y2, x), dim=1))

        gates = self.gate(torch.cat((y1, y2, y3), dim=1))
        gated_y = y1 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :]

        out = self.conv5(gated_y)

        #out = self.conv5(out)
       
     
	
        return y-out


#dn_net = Net()
#print(dn_net)



#from torchsummary import summary




#summary(dn_net, (1, 40, 40))
