from torch.utils.data import Dataset


class PoemDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.data.items()}
        item["labels"] = self.data.input_ids[index]
        return item

    def __len__(self):
        return len(self.data.input_ids)
