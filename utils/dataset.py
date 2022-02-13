from torch.utils.data import DataLoader
from TTS.tts.datasets.preprocess import load_meta_data
from TTS.utils.io import load_config
from torch.utils.data import Dataset

class Syncnet_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config_path, split = 'train'):
        """
        Args:
            c: TTS - Config
        """
        c = load_config(config_path)
        meta_data_train, meta_data_eval = load_meta_data(c.datasets)
        if split == 'train':
            self.items = meta_data_train
        else:
            self.items = meta_data_eval

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]