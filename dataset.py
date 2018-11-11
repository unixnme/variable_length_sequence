from torch.utils.data import Dataset
import numpy as np

class RandomDataset(Dataset):
    def __init__(self, N=100):
        self.N = N
        self.data = []
        for idx in range(N):
            length = np.random.randint(100, 200)
            self.data.append(np.random.randn(length))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.N

if __name__ == '__main__':
    dataset = RandomDataset()
    for x in dataset:
        print(x.shape)