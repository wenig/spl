import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


def make_classfication(n=10000, centroids=((0,0), (3,3))):
    y = torch.cat((torch.zeros(int(n / 2)).long(), torch.zeros(int(n / 2)).long()+1))
    X = torch.cat(
        (
            torch.stack(
                (
                    torch.Tensor(np.random.normal(loc=centroids[0][0], size=int(n / 2))),
                    torch.Tensor(np.random.normal(loc=centroids[0][1], size=int(n / 2)))
                ),
                dim=1
            ),

            torch.stack(
                (
                    torch.Tensor(np.random.normal(loc=centroids[1][0], size=int(n / 2))),
                    torch.Tensor(np.random.normal(loc=centroids[1][1], size=int(n / 2)))
                ),
                dim=1
            )
        )
    )
    return X, y


def add_anomaly(X, y):
    swap_id = np.linalg.norm(X.numpy() - np.array((3, 3)), axis=1).argmax()
    y[swap_id] = np.abs(y[swap_id]-1)
    print(f"Anomaly at {swap_id}")
    return torch.Tensor(X), y


class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.X, self.y = add_anomaly(*make_classfication())
        print(self.X.shape)

    def __iter__(self):
        return list(zip(self.X, self.y))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return item, self.X[item], self.y[item]


def get_dataloader():
    dataset = MyDataset()

    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )