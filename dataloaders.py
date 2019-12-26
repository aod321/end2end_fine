from torch.utils.data import DataLoader
from dataset import HelenDataset


class HelenLoader(object):
    def __init__(self, root_dir, transforms, batch_size, workers, mode='train'):        
        super(HelenLoader, self).__init__()
        self.transforms = transforms
        self.batch_size = batch_size
        self.workers = workers
        self.root_dir = root_dir
        self.mode = mode

    def get_dataloader(self):
        if self.mode == 'train':
            txt_file = 'exemplars.txt'
            shuffle = True

        if self.mode == 'val':
            txt_file = 'tuning.txt'
            shuffle = True

        if self.mode == 'test':
            txt_file = 'testing.txt'
            shuffle = False

        loader = DataLoader(HelenDataset(txt_file=txt_file,
                                    root_dir=self.root_dir,
                                    transform=self.transforms
                                    ), batch_size=self.batch_size,
                            shuffle=shuffle, num_workers=self.workers)
        
        return loader
