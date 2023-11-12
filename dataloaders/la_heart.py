import h5py
import numpy as np
from torch.utils.data import Dataset


class LAHeart(Dataset):
    """ LA Dataset """

    def __init__(self, base_dir=None, split='train', transform=None):
        self._base_dir = base_dir
        self.transform = transform
        if split == 'train':
            with open('data/train.txt', 'r') as f:
                self.image_list = f.readlines()
        else:
            with open('data/test.txt', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.strip() for item in self.image_list]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # print(image_name)
        h5f = h5py.File(self._base_dir + "/data/" + image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        image = image.astype(np.float32)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample