from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, img_lst, labels) -> None:
        super().__init__()
        self.img_lst = img_lst
        self.labels = labels

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, index):
        item = {
            "pixel_values": self.img_lst[index].squeeze(),
            "labels": self.labels[index],
        }
        return item
