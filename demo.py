import argparse
from networks import N3DED8
from pytorch_datasets import SubjectIndependentTestDataset
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

def normalize(data):    
    from sklearn.preprocessing import MinMaxScaler
    x = np.asarray(data)
    x = x.reshape(len(x), 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(x)
    scaled_x = scaler.transform(x)
    return scaled_x

def main_exp():
    
    """""""""
    START ARGPARSE
    """""""""
    parser = argparse.ArgumentParser()
    # EXPERIMENTAL SETUP  
    parser.add_argument('--run', type=str, choices=['rtrppg_demo','npsnr_demo'], default='none',required=False)

    # JOIN ALL ARGUMENTS
    args = parser.parse_args()
    if args.run in ['none']:
        print('[RTrPPG demo]=>Please select a valid option:\n--run rtrppg demo\n--run npsnr_demo')
    elif args.run in ['rtrppg_demo']:
        print('================================================================')
        print('                     Running RTrPPG demo                        ')
        print('================================================================') 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ## MODEL
        RTRPPG = N3DED8()
        RTRPPG.to(device)
        checkpoint = torch.load('weights.pth.tar',map_location=torch.device('cpu'))
        RTRPPG.load_state_dict(checkpoint['model_state_dict'])
        current_path = os.getcwd()
        ## DATA MANAGER
        path = os.path.abspath(os.path.join(current_path,'demo_subject\p1v1s1'))

        dataset = SubjectIndependentTestDataset(path) 
        dataloader = DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False)

        GT = dataset.y_file # GT
        time = dataset.t_file # time
        window = dataset.window # Slinding window length
        rPPG = []
        
        with torch.no_grad():
            for idx, sample in enumerate(dataloader):
                out = RTRPPG(sample['x'])
                out = out-torch.mean(out,keepdim=True,dim=1)/torch.std(out,keepdim=True,dim=1)
                rPPG.append(out.to('cpu').detach().numpy())
        
        rPPG = np.vstack(rPPG)
        
        ## OVERLAP-ADD PROCESS                
        y_hat = np.zeros(window+len(rPPG)-1)
        for i in range(len(rPPG)):
            y_hat[i:i+window] = y_hat[i:i+window]+rPPG[i]
        y_hat = np.squeeze(normalize(y_hat))
        
        # PLOT
        fig, ax = plt.subplots()
        plt.plot(time,GT),plt.plot(time,y_hat)
        plt.ylabel("Amplitude")
        plt.xlabel("Time [s]")
        plt.legend(['ground truth','rPPG'])     
        fig.savefig('Output.png', format='png', dpi=1200)
        print('[rtrppg demo]=>Output.png file saved in '+current_path)
        
    elif args.run in ['npsnr_demo']:
        print('================================================================')
        print('                     Running npsnr demo                         ')
        print('================================================================')
        # TODO
        print('[npsnr demo]=>Sorry, this demo is not available yet')



if __name__ == "__main__":
    main_exp()