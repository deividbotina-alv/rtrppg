import torch
import numpy as np
import os

class SubjectIndependentTestDataset(torch.utils.data.Dataset):
    """This dataset takes the path where the subject folder is located. After loading the data,
    it divides it through a sliding window of length=128, step=1.
    """
 
    def __init__(self, load_path:str=r'content/data'):
        """
        Args:
            load_path(str): Path where the data is located
            window(int): sliding window length
            step(int): sliding window step
            img_size(int): Squared image size
        """

        self.load_path = load_path
        self.name = load_path.split(os.path.sep)[-1]
        self.window = 128
        self.step = 1
        self.img_size = 8
        self.frames = np.load(os.path.join(self.load_path,self.name+'.npy'))
        self.Frames = np.zeros((self.frames.shape[0],self.img_size,self.img_size,3),dtype=np.float32)    
        
        for i in range(0,self.frames.shape[0]):
            frame = np.array(self.frames[i,:,:,:].copy(),dtype=np.float32)/255.0            
            self.Frames[i,:,:,:] = frame
       
        # Load ground truth file
        self.y_file = self.getFullGTfile()        
        # Load timestamp file
        self.t_file = self.getFulltimeFile()
        # Get windows index
        self.windows = self.getWindows()
            
    # RETURN NUMBER OF WINDOWS
    def __len__(self):
            return len(self.windows)

    # RETURN THE [i] FILE
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Take only current window from big tensor
        frames = self.take_frames_crt_window(self.windows[idx])
        GT = torch.tensor(self.y_file[self.windows[idx][0]:self.windows[idx][1]], dtype=torch.float32)
        time = torch.tensor(self.t_file[self.windows[idx][0]:self.windows[idx][1]], dtype=torch.float32)
        sample = {'x':frames, 'y':GT, 't':time}
        return sample    

    # FUNCTION TO GET THE WINDOWS INDEX
    def getWindows(self):
        windows = []
        for i in range(0,np.size(self.Frames,0)-self.window+1,self.step):
            windows.append((i,i+self.window))
        return windows

    # FUNCTION TO TAKE ONLY THE FRAMES OF THE CURRENT WINDOW AND RETURN A TENSOR OF IT
    def take_frames_crt_window(self,idx):
        frames = torch.zeros((3,self.window,self.img_size,self.img_size)) # list with all frames {3,T,128,128}
        
        # Load all frames in current window
        for j,i in enumerate(range(idx[0],idx[1])):
            frame = self.Frames[i,:,:,:]
            frame = torch.tensor(frame, dtype=torch.float32).permute(2,0,1)#In pythorch channels must be in position 0, cv2 has channels in 2
            frames[:,j,:,:] = frame 
        return frames

    # FUNCTION TO GET FULL GROUND TRUTH FILE
    def getFullGTfile(self):
        return np.loadtxt(os.path.join(self.load_path,self.name+'_gt.txt'))

    # FUNCTION TO GET THE FULL TIMESTAMP FILE  
    def getFulltimeFile(self):
        return np.loadtxt(os.path.join(self.load_path,self.name+'_timestamp.txt'))
        
