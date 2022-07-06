import torch
import torch.nn as nn
#%% Negative Pearson's correlation
# Traken from https://github.com/ZitongYu/PhysNet/blob/master/NegPearsonLoss.py
'''
Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks' 
By Zitong Yu, 2019/05/05
If you use the code, please cite:
@inproceedings{yu2019remote,
    title={Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks},
    author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
    booktitle= {British Machine Vision Conference (BMVC)},
    year = {2019}
}
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019 
'''

class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):       # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))

            #if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #    loss += 1 - pearson
            #else:
            #    loss += 1 - torch.abs(pearson)
            
            loss += 1 - pearson
            
            
        loss = loss/preds.shape[0]
        return loss
        
#%% Negative Pearson's correlation + Signal-to-Noise-Ratio (NPSNR)

class NPSNR(nn.Module):
    def __init__(self,Lambda,LowF=0.7,upF=3.5,width=0.4):
        super(NPSNR,self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Lambda = Lambda
        self.LowF = LowF
        self.upF = upF
        self.width = width
        self.NormaliceK = 1/10.9 #Constant to normalize SNR between -1 and 1
        return
    
    def forward(self, sample:list):
        assert len(sample)>=3, print('=>[NP_NSNR] ERROR, sample must have 3 values [y_hat, y, time]')
        rppg = sample[0]
        gt = sample[1]
        time = sample[2]
        loss = 0        
        for i in range(rppg.shape[0]):
            ##############################
            # PEARSON'S CORRELATION
            ##############################
            sum_x = torch.sum(rppg[i])                # x
            sum_y = torch.sum(gt[i])               # y
            sum_xy = torch.sum(rppg[i]*gt[i])        # xy
            sum_x2 = torch.sum(torch.pow(rppg[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(gt[i],2)) # y^2
            N = rppg.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))   
            ##############################
            # SNR
            ##############################
            N = rppg.shape[-1]*3
            Fs = 1/time[i].diff().mean()
            freq = torch.arange(0,N,1,device=self.device)*Fs/N            
            fft = torch.abs(torch.fft.fft(rppg[i],dim=-1,n=N))**2
            gt_fft = torch.abs(torch.fft.fft(gt[i],dim=-1,n=N))**2
            fft = fft.masked_fill(torch.logical_or(freq>self.upF,freq<self.LowF).to(self.device),0)
            gt_fft = gt_fft.masked_fill(torch.logical_or(freq>self.upF,freq<self.LowF).to(self.device),0)
            PPG_peaksLoc = freq[gt_fft.argmax()]
            mask = torch.zeros(fft.shape[-1],dtype=torch.bool,device=self.device)
            mask = mask.masked_fill(torch.logical_and(freq<PPG_peaksLoc+(self.width/2),PPG_peaksLoc-(self.width/2)<freq).to(self.device),1)#Main signal
            mask = mask.masked_fill(torch.logical_and(freq<PPG_peaksLoc*2+(self.width/2),PPG_peaksLoc*2-(self.width/2)<freq).to(self.device),1)#Armonic
            power = fft*mask
            noise = fft*mask.logical_not().to(self.device)
            SNR = (10*torch.log10(power.sum()/noise.sum()))*self.NormaliceK
            ##############################
            # JOIN BOTH LOSS FUNCTION
            ##############################
            loss += 1 - (pearson+(self.Lambda*SNR))  
            
        loss = loss/rppg.shape[0]
        return loss 