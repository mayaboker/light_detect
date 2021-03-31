import torch


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.seperators = self.get_seperators(datasets)

    def __getitem__(self, index):
        dataset_ind, ind = self.get_actual_index(index)
        return self.datasets[dataset_ind].__getitem__(ind)

    def __len__(self):
        return self.seperators[-1]

    def get_actual_index(self, index):
        dataset_ind = -1
        ind = -1
        for i, seperator in enumerate(self.seperators):
            if index < seperator:
                dataset_ind = i
                pos = seperator - len(self.datasets[i]) -1
                ind = index - pos
                break
        return dataset_ind, ind

    def get_seperators(self, datasets):
        seperators = []
        count = 0
        for dataset in datasets:
            count += len(dataset)
            seperators.append(count)
        return seperators
