import unittest
import torch
import numpy as np
from utils.helper_functions import set_seed, get_data_augmented, get_data
from utils.networks import LeNet5, AlexNet
from utils.train_tools import roc_auc_score_multiclass
from torchvision import transforms


class test_help_functions(unittest.TestCase):
    """ Class for testing the help_functions.py file functions """
    def test_networks(self):
        """
        Test self written networks in networks.py
        """
        model = LeNet5()
        x = torch.ones((1, 1, 299, 299))  # dummy tensor
        y = model(x)
        # test for 4 classes output size
        assert y.size() == (1, 4), \
            "Network LeNet5() has wrong number of output classes"
        model = AlexNet()
        y = model(x)
        assert y.size() == (1, 4), \
            "Network AlexNet() has wrong number of output classes"
        print("models passed")

    def test_set_seed(self):
        """Test whether seed is set correctly"""
        print("Test set_seed()")
        out = set_seed(42)
        assert out is True, "Didn't set the seed"

    def test_train_tools_roc_auc(self):
        """Test roc area under the curve metric function"""
        # perfectly correct output
        assert roc_auc_score_multiclass(
            np.array([1, 2, 0]), np.array([1, 2, 0])) == 1, \
            "maximum area not achieved when correct predictions"
        # Low output example, classifying wrong result well
        assert roc_auc_score_multiclass(
            np.array([1, 2, 0]), np.array([0, 1, 1])) == 0.25, \
            "incorrect prediction"

    def test_augmentation_dataset(self):
        """Test data augmentation function"""
        filepath_train = 'tests/test_dataset/xray-data/train'
        filepath_test = 'tests/test_dataset/xray-data/testouter'
        trainset, validset, test = get_data_augmented(filepath_train,
                                                      filepath_test, 0.2)
        # either 1 or 3 channels
        a = trainset[0][0].view(-1, 299, 299).shape[0] == 1
        b = trainset[0][0].view(-1, 299, 299).shape[0] == 3
        assert (a or b), "data read in with wrong number of channels"

    def test_get_data(self):
        """test whether classes names are read in as expected"""
        dat = get_data(
            'tests/test_dataset/xray-data/train',
            transforms.Compose([transforms.ToTensor()]))
        assert len(dat) == 16, "not all data was read in"
        assert dat[0][0][0].shape == (299, 299), \
            "images size has been compromised"
        assert dat.class_to_idx == {'0-covid': 0,
                                    '1-lung_opacity': 1,
                                    '2-normal': 2,
                                    '3-pneumonia': 3}, \
            "classes order has been compromised"


# Run using python tests.py in command line
unittest.main()
