# -*- coding: utf-8 -*-
'''
Compare computed metrics with results from original Matlab implementation.

The Matlab implementation is downloaded from http://saliency.mit.edu/
Bylinskii, Z., Judd, T., Durand, F., Oliva, A., & Torralba, A. (n.d.). MIT Saliency Benchmark.
'''

import unittest
from os import path
import glob
from skimage.io import imread
from skimage import img_as_float

from saliency.benchmark import metrics


class TestMetrics(unittest.TestCase):
    def test_AUC_Judd(self):
        maps = load_dataset('road')
        self.assertAlmostEqual(metrics.AUC_Judd(maps['saliency1'], maps['fixation']), 0.76289570, places=2)
        self.assertAlmostEqual(metrics.AUC_Judd(maps['large'], maps['fixation']), 0.76290159, places=2)

    def test_AUC_Borji(self):
        maps = load_dataset('road')
        self.assertAlmostEqual(metrics.AUC_Borji(maps['saliency1'], maps['fixation'], 10000), 0.75449380, places=2)
        self.assertAlmostEqual(metrics.AUC_Borji(maps['large'], maps['fixation'], 10000), 0.75554346, places=2)

    def test_AUC_shuffled(self):
        maps = load_dataset('road')
        self.assertAlmostEqual(metrics.AUC_shuffled(maps['saliency1'], maps['fixation'], maps['other'], 10000), 0.79630240, places=2)
        self.assertAlmostEqual(metrics.AUC_shuffled(maps['large'], maps['fixation'], maps['other'], 10000), 0.79756335, places=2)

    def test_NSS(self):
        maps = load_dataset('road')
        self.assertAlmostEqual(metrics.NSS(maps['saliency1'], maps['fixation']), 0.94992533, places=4)
        self.assertAlmostEqual(metrics.NSS(maps['large'], maps['fixation']), 0.95030310, places=4)

    def test_CC(self):
        maps = load_dataset('road')
        self.assertAlmostEqual(metrics.CC(maps['saliency1'], maps['saliency2']), 0.41267651)
        self.assertAlmostEqual(metrics.CC(maps['large'], maps['saliency2']), 0.41264365, places=2)

    def test_SIM(self):
        maps = load_dataset('road')
        self.assertAlmostEqual(metrics.SIM(maps['saliency1'], maps['saliency2']), 0.43001505)
        self.assertAlmostEqual(metrics.SIM(maps['large'], maps['saliency2']), 0.43025001, places=4)

    # def test_EMD(self):
    #     # TODO: The differences seem to be rooted at the shrinking imresize (no antialias at Matlab?)
    #     maps = load_dataset('road')
    #     self.assertAlmostEqual(metrics.EMD(maps['saliency1'], maps['saliency2']), 0.74947831)


def load_dataset(name):
    '''
    Load specified test dataset.

    Each image file in the dataset folder is read as numpy.array into a dict, indexed by filename.

    Parameters
    ----------
    name : string
        Usable datasets include: ['road', ]

    Returns
    -------
    maps : dict
    '''
    files = sorted(glob.glob(path.join(path.dirname(__file__), name, '*.*')))
    return {path.splitext(path.basename(f))[0]: img_as_float(imread(f)) for f in files}


if __name__ == '__main__':
    unittest.main()