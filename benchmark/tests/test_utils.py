# -*- coding: utf-8 -*-

import unittest
import numpy as np

from saliency.benchmark import utils


class TestUtils(unittest.TestCase):
    def test_normalize(self):
        x = np.tile(np.r_[1,2,3], [3,1]).T
        # Flatten
        y = utils.normalize(x, method='standard')
        self.assertEqual(x.shape, y.shape)
        self.assertAlmostEqual(np.mean(y), 0)
        self.assertAlmostEqual(np.std(y), 1)
        y = utils.normalize(x, method='range')
        self.assertAlmostEqual(np.min(y), 0)
        self.assertAlmostEqual(np.max(y), 1)
        y = utils.normalize(x, method='sum')
        self.assertAlmostEqual(np.sum(y), 1)
        # Perpendicular to axis
        y = utils.normalize(x, method='standard', axis=1)
        self.assertTrue(np.allclose(np.mean(y, axis=0), np.zeros(3)))
        self.assertTrue(np.allclose(np.std(y, axis=0), np.ones(3)))
        y = utils.normalize(x, method='range', axis=1)
        self.assertTrue(np.allclose(np.min(y, axis=0), np.zeros(3)))
        self.assertTrue(np.allclose(np.max(y, axis=0), np.ones(3)))
        y = utils.normalize(x, method='sum', axis=1)
        self.assertTrue(np.allclose(np.sum(y, axis=0), np.ones(3)))


if __name__ == '__main__':
    unittest.main()
