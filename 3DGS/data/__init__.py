from .dataset import Dataset
import torch.utils.data

class DataLoader():
    def __init__(self, opt, training=True) -> None:
        dataset = Dataset(opt, training)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=training,
            num_workers=4,
        )
    
    def load_data(self):
        return self.dataloader
