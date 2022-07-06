import torch.nn as nn

#%% 3DED128 (baseline)

class N3DED128(nn.Module):
    def __init__(self, frames=128):  
        super(N3DED128, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.Conv4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.TrConv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.TrConv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock5 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.MaxpoolSpaTem_244_244 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x): # [batch, Features=3, Temp=128, Width=128, Height=128]
        # ENCODER
        x = self.Conv1(x)		            # [b, F=3, T=128, W=128, H=128]->[b, F=16, T=128, W=128, H=128]
        x = self.MaxpoolSpaTem_244_244(x)   # [b, F=16, T=128, W=128, H=128]->[b, F=16, T=64, W=32, H=32]

        x = self.Conv2(x)		            # [b, F=16, T=64, W=32, H=32]->[b, F=32, T=64, W=32, H=32]
        x = self.MaxpoolSpaTem_244_244(x)   # [b, F=32, T=64, W=32, H=32]->[b, F=32, T=32, W=8, H=8]
        
        x = self.Conv3(x)		            # [b, F=32, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        x = self.Conv4(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        # DECODER
        x = self.TrConv1(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=64, W=8, H=8]
        x = self.TrConv2(x)		            # [b, F=64, T=64, W=8, H=8]->[b, F=64, T=128, W=8, H=8]        
        x = self.poolspa(x)                 # [b, F=64, T=128, W=8, H=8]->[b, F=64, T=128, W=1, H=1]
        x = self.ConvBlock5(x)             # [b, F=64, T=128, W=1, H=1]->[b, F=1, T=128, W=1, H=1]
        
        rPPG = x.view(-1,x.shape[2])        # [b,128]
        return rPPG

#%% 3DED64

class N3DED64(nn.Module):
    def __init__(self, frames=128):  
        super(N3DED64, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.Conv4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.TrConv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.TrConv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock5 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.MaxpoolSpaTem_244_244 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        self.MaxpoolSpaTem_222_222 = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x): # [batch, Features=3, Temp=128, Width=64, Height=64]
        # ENCODER
        x = self.Conv1(x)		            # [b, F=3, T=128, W=64, H=64]->[b, F=16, T=128, W=64, H=64]
        x = self.MaxpoolSpaTem_222_222(x)   # [b, F=16, T=128, W=64, H=64]->[b, F=16, T=64, W=32, H=32]

        x = self.Conv2(x)		            # [b, F=16, T=64, W=32, H=32]->[b, F=32, T=64, W=32, H=32]
        x = self.MaxpoolSpaTem_244_244(x)   # [b, F=32, T=64, W=32, H=32]->[b, F=32, T=32, W=8, H=8]
        
        x = self.Conv3(x)		            # [b, F=32, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        x = self.Conv4(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        # DECODER
        x = self.TrConv1(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=64, W=8, H=8]
        x = self.TrConv2(x)		            # [b, F=64, T=64, W=8, H=8]->[b, F=64, T=128, W=8, H=8]        
        x = self.poolspa(x)                 # [b, F=64, T=128, W=8, H=8]->[b, F=64, T=128, W=1, H=1]
        x = self.ConvBlock5(x)             # [b, F=64, T=128, W=1, H=1]->[b, F=1, T=128, W=1, H=1]
        
        rPPG = x.view(-1,x.shape[2])        # [b,128]
        return rPPG

#%% 3DED32

class N3DED32(nn.Module):
    def __init__(self, frames=128):  
        super(N3DED32, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.Conv4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.TrConv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.TrConv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock5 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.MaxpoolSpaTem_244_244 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        self.MaxpoolTem_211_211 = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x): # [batch, Features=3, Temp=128, Width=32, Height=32]
        # ENCODER
        x = self.Conv1(x)		            # [b, F=3, T=128, W=32, H=32]->[b, F=16, T=128, W=32, H=32]
        x = self.MaxpoolTem_211_211(x)   # [b, F=16, T=128, W=32, H=32]->[b, F=16, T=64, W=32, H=32]

        x = self.Conv2(x)		            # [b, F=16, T=64, W=32, H=32]->[b, F=32, T=64, W=32, H=32]
        x = self.MaxpoolSpaTem_244_244(x)   # [b, F=32, T=64, W=32, H=32]->[b, F=32, T=32, W=8, H=8]
        
        x = self.Conv3(x)		            # [b, F=32, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        x = self.Conv4(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        # DECODER
        x = self.TrConv1(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=64, W=8, H=8]
        x = self.TrConv2(x)		            # [b, F=64, T=64, W=8, H=8]->[b, F=64, T=128, W=8, H=8]        
        x = self.poolspa(x)                 # [b, F=64, T=128, W=8, H=8]->[b, F=64, T=128, W=1, H=1]
        x = self.ConvBlock5(x)             # [b, F=64, T=128, W=1, H=1]->[b, F=1, T=128, W=1, H=1]
        
        rPPG = x.view(-1,x.shape[2])        # [b,128]
        return rPPG

#%% 3DED16

class N3DED16(nn.Module):
    def __init__(self, frames=128):  
        super(N3DED16, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.Conv4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.TrConv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.TrConv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock5 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.MaxpoolSpaTem_222_222 = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolTem_211_211 = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x): # [batch, Features=3, Temp=128, Width=16, Height=16]
        # ENCODER
        x = self.Conv1(x)		            # [b, F=3, T=128, W=16, H=16]->[b, F=16, T=128, W=16, H=16]
        x = self.MaxpoolTem_211_211(x)   # [b, F=16, T=128, W=16, H=16]->[b, F=16, T=64, W=16, H=16]

        x = self.Conv2(x)		            # [b, F=16, T=64, W=16, H=16]->[b, F=32, T=64, W=16, H=16]
        x = self.MaxpoolSpaTem_222_222(x)   # [b, F=32, T=64, W=16, H=16]->[b, F=32, T=32, W=8, H=8]
        
        x = self.Conv3(x)		            # [b, F=32, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        x = self.Conv4(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        # DECODER
        x = self.TrConv1(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=64, W=8, H=8]
        x = self.TrConv2(x)		            # [b, F=64, T=64, W=8, H=8]->[b, F=64, T=128, W=8, H=8]        
        x = self.poolspa(x)                 # [b, F=64, T=128, W=8, H=8]->[b, F=64, T=128, W=1, H=1]
        x = self.ConvBlock5(x)             # [b, F=64, T=128, W=1, H=1]->[b, F=1, T=128, W=1, H=1]
        
        rPPG = x.view(-1,x.shape[2])        # [b,128]
        return rPPG


#%% 3DED8 (RTrPPG) - Note that 3DED8, 3DED4, and 3DED2 are the same architecture.

class N3DED8(nn.Module):
    def __init__(self, frames=128):  
        super(N3DED8, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolTem = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolSpaTem1 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((128,1,1))

        
    def forward(self, x):	    	# x [6, T=128, 8, 8]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [6, T=128, 8, 8]->[16, T=128,  8, 8]
        x = self.MaxpoolTem(x)       # x [16, T=128, 8, 8]->[16, T=64, 8, 8]

        x = self.ConvBlock2(x)		     # x [16, T=64, 8, 8]->[32, T=64, 8, 8]
        x = self.MaxpoolTem(x)       # x [32, T=64, 8, 8]->[32, T=32, 8, 8]
        
        x = self.ConvBlock3(x)		    # x [32, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.ConvBlock4(x)		    # x [64, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.upsample(x)		    # x [64, T=32, 8, 8]->[64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [64, T=64, 8, 8]->[64, T=128, 8, 8]
        
        x = self.poolspa(x)     # x [64, T=128, 8, 8]->[64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [64, T=128, 1, 1]->[1, T=128, 1, 1]
        
        rPPG = x.view(-1,length) # [128]        rPPG = x.view(-1,length)   
        
        return rPPG
