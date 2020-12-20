import numpy as np
import torch
import os
from torch.utils.data.dataset import Dataset

def arr_at_each_location(image,direction,loc):
    arr=[]
    x_size,y_size,z_size=image.shape
    if direction=="X":
        assert 0<=loc<image.shape[0]
        arr=[
            image[loc-5,:,:] if loc-5>=0 else np.zeros((y_size,z_size)),
            image[loc-2,:,:] if loc-2>=0 else np.zeros((y_size,z_size)),
            image[loc,:,:],
            image[loc+2,:,:] if loc+2<x_size else np.zeros((y_size,z_size)),
            image[loc+5,:,:] if loc+5<x_size else np.zeros((y_size,z_size))
        ]
    elif direction=="Y":
        assert 0<=loc<image.shape[1]
        arr=[
            image[:,loc-5,:] if loc-5>=0 else np.zeros((x_size,z_size)),
            image[:,loc-2,:] if loc-2>=0 else np.zeros((x_size,z_size)),
            image[:,loc,:],
            image[:,loc+2,:] if loc+2<y_size else np.zeros((x_size,z_size)),
            image[:,loc+5,:] if loc+5<y_size else np.zeros((x_size,z_size))
        ]
    elif direction=="Z":
        assert 0<=loc<image.shape[2]
        arr=[
            image[:,:,loc-5] if loc-5>=0 else np.zeros((x_size,y_size)),
            image[:,:,loc-2] if loc-2>=0 else np.zeros((x_size,y_size)),
            image[:,:,loc],
            image[:,:,loc+2] if loc+2<z_size else np.zeros((x_size,y_size)),
            image[:,:,loc+5] if loc+5<z_size else np.zeros((x_size,y_size)),
        ]
    else:
        assert False
    arr=np.stack(arr,0)
    return arr.astype(np.float32)

class COVID19Dataset(Dataset):

    def __init__(self, data_dir, direction='X'):
        self.direction = direction
        self.images = []
        self.labels = []
        self.slices = []
        patient_ids = os.listdir(data_dir)
        for patient_id in patient_ids:
            scans = os.listdir(os.path.join(data_dir,patient_id))
            for scan in scans:
                # preload the image
                image = np.load(os.path.join(data_dir,patient_id,scan,scan+'_data.npy'))
                label = np.load(os.path.join(data_dir,patient_id,scan,scan+'_label.npz'))['array']
                x_size,y_size,z_size=image.shape
                if self.direction == 'X':
                    slices = x_size
                elif self.direction == 'Y':
                    slices = y_size
                else:
                    slices = z_size
                for i in range(slices):
                    self.images.append(image)
                    self.labels.append(label)
                    self.slices.append(i)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        # image = np.load(self.images[idx])
        # label = np.load(self.labels[idx])['array']
        image = self.images[idx]
        label = self.labels[idx]
        image_slice = arr_at_each_location(image, self.direction, self.slices[idx])
        if self.direction == 'X':
            label_slice = label[self.slices[idx],:,:]
        elif self.direction == 'Y':
            label_slice = label[:,self.slices[idx],:]
        else:
            label_slice = label[:,:,self.slices[idx]]
        return image_slice, label_slice

if __name__ == "__main__":
    train_dataset = COVID19Dataset(data_dir='/home/sbian/work/side/minibatch/mini_batch_version/codes/home/sbian/work/side/minibatch/mini_batch_version/dataset/train/', direction='X')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=5,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    drop_last=False)
    for batch_idx, tup in enumerate(train_dataloader):
        image, label = tup
        print(f'image shape:{image.shape}')
        print(f'label shape:{label.shape}')
        break
