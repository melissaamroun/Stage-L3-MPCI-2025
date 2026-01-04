# -*- coding: utf-8 -*-
"""Prepare dataset for training and testing the model.

@author: Frédéric Richard, 2025, AMU
"""
import yaml
from afbfdb import protocol

from numpy import amin, amax, mean, std, floor
from numpy.random import seed, permutation
from numpy.random import standard_normal as randn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class Texture_Database(Dataset):
    def __init__(self, rep, indices, config, name):
        """Initialization.

        Parameters
        ----------
        rep : str
            "path_to_the_database_directory".
        indices : array
            indices of examples to include in the dataset.
        params_transform : str, optional
            transforms to be applied to images.
        """
        self.rep = rep
        self.indices = indices
        self.dataset = protocol(self.rep)
        self.params_transforms = config['image_transforms']
        self.name = name
        if max(indices) >= self.dataset.nbexpe:
            raise Exception("The dataset is to small.")

    def __getitem__(self, idx):
        """Return the image and the features associated to an example.
        """
        image, features = self.dataset.ExportData(self.indices[idx])
        if self.params_transforms['discretization']:
            image = self. DiscretizeImageIntensity(image)
        if self.params_transforms['normalization']:
            image = self.NormalizeImage(image)
        if self.params_transforms['noise']:
            image = self.DegradeImage(image)
        if self.params_transforms['crop']:
            image = self.CropImage(image)
        return (Tensor(image).unsqueeze(dim=0), Tensor([features[0]]))

    def __len__(self):
        return len(self.indices)

    def DisplaySetting(self):
        """Display the setting of the dataset.
        """
        print(self.name + " : ")
        print("from " + self.rep)
        print("Number of examples: %d" % (self.__len__()))

    def NormalizeImage(self, image):
        """Centering and normalization of the image.
        """
        return (image - mean(image)) / std(image)

    def DiscretizeImageIntensity(self, image):
        """Discretization of image intensity.
        """
        immin = amin(image)
        immax = amax(image)
        imrange = immax - immin
        glmax = self.params_transforms['glmax']
        return floor((image - immin) / imrange * glmax)

    def DegradeImage(self, image):
        """Addition of Gaussian noise to the image.
        """
        return image + self.params_transforms['sigma'] * randn(image.shape)
    
    def CropImage(self, image):
        """Square crop image bottom left
        """
        crop_size = self.params_transforms['crop_size']
        H, W = image.shape
        if crop_size > min(H,W) :
            raise Exception("Image size is smaller than crop size")
        return image[0 : crop_size, 0 : crop_size]


def CreateDatasets(config):
    params_data = config['data']
    rep_data = params_data['rep_data']
    dataset = protocol(rep_data)
    n = dataset.nbexpe

    n_train = int(n * params_data['training'])
    n_val = n_train + int(n * params_data['validation'])
    n_test = n_val + int(n * params_data['test'])

    # Split the data into training, validation and test datasets.
    seed(params_data['seed'])
    indices = permutation(n)
    indices_train = indices[0:n_train]
    indices_val = indices[n_train:n_val]
    indices_test = indices[n_val:n_test]

    # Define the datasets
    train_set = Texture_Database(rep_data, indices_train, config,
                                 "Training set")
    val_set = Texture_Database(rep_data, indices_val, config,
                               "Validation set")
    test_set = Texture_Database(rep_data, indices_test, config,
                                "Test set")
    train_set.DisplaySetting()
    val_set.DisplaySetting()
    test_set.DisplaySetting()

    # Define the loaders.
    params_opti = config['optimization']
    train_load = DataLoader(train_set, batch_size=params_opti['batch_size'],
                            shuffle=True)
    val_load = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_load = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    return {"Training": train_set, "Validation": val_set, "Test": test_set}, \
        {"Training": train_load, "Validation": val_load, "Test": test_load}


if __name__ == '__main__':
    with open("./Experiments/Experiments-011/setting.yaml", "r") as f:
        expe_dir = yaml.safe_load(f)

    sets, loaders = CreateDatasets(expe_dir)
