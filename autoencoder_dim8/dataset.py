import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir, test_set):
        data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
        self.data_dic = {}
        if test_set is None:
            for i in range(len(data_names)):
                features = np.load(data_names[i])
                name = data_names[i].split('/')[-1].split('.')[0]
                self.data_dic[name] = features.shape[0]
                if i == 0:
                    data = features
                else:
                    data = np.concatenate([data, features], axis=0)
            #print(data.shape)(39060, 512)
            self.data = data
        else:
            train_names = []
            test_names = []
            for i in range(len(data_names)):
                name = data_names[i].split('/')[-1].split('.')[0]
                #print(name)DSCF4714_f
                if name[:-2] in test_set:
                    test_names.append(data_names[i])
                else:
                    train_names.append(data_names[i])
            print(len(train_names))
            print(len(test_names))

            for i in range(len(train_names)):
                features = np.load(train_names[i])
                name = train_names[i].split('/')[-1].split('.')[0]
                self.data_dic[name] = features.shape[0]
                if i == 0:
                    data_train = features
                else:
                    data_train = np.concatenate([data_train, features], axis=0)

            for i in range(len(test_names)):
                features = np.load(test_names[i])
                name = test_names[i].split('/')[-1].split('.')[0]
                self.data_dic[name] = features.shape[0]
                if i == 0:
                    data_test = features
                else:
                    data_test = np.concatenate([data_test, features], axis=0)

            self.data = [data_train, data_test]

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data

    def __len__(self):
        return self.data.shape[0] 