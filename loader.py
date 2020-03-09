import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from info import Info
import librosa

class Normalize(object):
    def __call__(self, sample):
        return (sample-Info.mini)/(Info.maxi-Info.mini)
        
class DeNorm:
    def __call__(self,sample):
        return sample*(Info.maxi-Info.mini)+Info.mini

class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)

# load spectrogram as input
class Music(Dataset): #Music dataset
    def __init__(self, data_path, file_arr, transform=None):
        self.data_path = data_path
        self.data = []
        for f in file_arr:
            # dat = np.load(self.data_path+f+'_mag.npy')
            dat = np.load(self.data_path+f)
            # print(dat.shape, dat)
            dat = librosa.amplitude_to_db(dat, ref=1.0)
            # print('dB: ',dat)
            self.data.extend(dat.T)
            # print(self.data)
        self.data = np.array(self.data).T
        # print('dataset shape=',self.data.shape)
        # quit()
        self.transform = transform
    def __len__(self): # num of items
        # return 20
        return len(self.data[0])
    def __getitem__(self, idx): # how to get one item
        sample = [self.data[:,idx]]
        if self.transform:
            sample = self.transform(sample)
        # print('after transform',sample)
        # quit()
        return sample
        
        
        
        
