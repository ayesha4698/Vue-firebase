"""
Convention: append an integer to the end of the test, for multiple versions of
the same test at different difficulties.  Higher numbers are more difficult
(lower thresholds or accept fewer mistakes).  Example:
    test_all_equal1(self):
    ...
    test_all_equal2(self):
    ...
"""

import argparse
import json
import os
import math
import unittest

import numpy as np

import texture_gradient as tg

atol = 1e-03
class TestTextureGradients(unittest.TestCase):
    def setUp(self):
        self.test_data = np.load('part_c_test_data.npz')
        self.img = self.test_data['img'] # 64 x 64 random data
        self.theta = self.test_data['theta']
        self.theta2 = self.test_data['theta2']
        self.theta3 = self.test_data['theta3']

    def test_difference_of_gaussian(self):
        x, y = self.test_data['x'], self.test_data['y']
        dog = tg.getDoG(x, y)


        self.assertTrue(np.allclose(dog, self.test_data['dog'], atol=atol))

    def test_oriented_filter_theta(self):
        f1 = tg.getOrientedFilter(tg.getGaussDeriv, self.theta)

        f2 = tg.getOrientedFilter(tg.getGauss2Deriv, self.theta)
        fdiff = tg.getOrientedFilter(tg.getDoG, self.theta)
        self.assertTrue(np.allclose(f1, self.test_data['f1'], atol=atol))
        self.assertTrue(np.allclose(f2, self.test_data['f2'], atol=atol))
        self.assertTrue(np.allclose(fdiff, self.test_data['fdiff'], atol=atol))


    def test_oriented_filter_theta2(self):
        f1_2 = tg.getOrientedFilter(tg.getGaussDeriv, self.theta2)
        f2_2= tg.getOrientedFilter(tg.getGauss2Deriv, self.theta2)
        fdiff_2 = tg.getOrientedFilter(tg.getDoG, self.theta2)
        self.assertTrue(np.allclose(f1_2, self.test_data['f1_2'], atol=atol))
        self.assertTrue(np.allclose(f2_2, self.test_data['f2_2'], atol=atol))
        self.assertTrue(np.allclose(fdiff_2, self.test_data['fdiff_2'], atol=atol))

    def test_get_textons(self):
        labels, feats = tg.getTextons(self.img, vocab_size=4)
        self.assertTrue(np.allclose(labels, self.test_data['labels'], atol=atol))
        self.assertTrue(np.allclose(feats, self.test_data['feats'], atol=atol))
    #
    def test_get_masks_theta(self):
        theta3_mask1_even, theta3_mask2_even = tg.getMasks(self.theta3, 4)
        theta3_mask1_odd, theta3_mask2_odd = tg.getMasks(self.theta3, 5)
        self.assertTrue(np.allclose(theta3_mask1_even, self.test_data['theta3_mask1_even'], atol=atol))
        self.assertTrue(np.allclose(theta3_mask2_even, self.test_data['theta3_mask2_even'], atol=atol))

        self.assertTrue(np.allclose(theta3_mask1_odd, self.test_data['theta3_mask1_odd'], atol=atol))
        self.assertTrue(np.allclose(theta3_mask2_odd, self.test_data['theta3_mask2_odd'], atol=atol))


    def test_get_masks_theta2(self):
        theta2_mask1_even, theta2_mask2_even = tg.getMasks(self.theta2, 4)
        theta2_mask1_odd, theta2_mask2_odd = tg.getMasks(self.theta2, 5)

        self.assertTrue(np.allclose(theta2_mask1_even, self.test_data['theta2_mask1_even'], atol=atol))
        self.assertTrue(np.allclose(theta2_mask2_even, self.test_data['theta2_mask2_even'], atol=atol))

        self.assertTrue(np.allclose(theta2_mask1_odd, self.test_data['theta2_mask1_odd'], atol=atol))
        self.assertTrue(np.allclose(theta2_mask2_odd, self.test_data['theta2_mask2_odd'], atol=atol))

    def test_compute_texture_gradient(self):
        g = tg.computeTextureGradient(self.img, vocab_size=4, r=3)

        checka = np.allclose(g, self.test_data['g'], atol=atol)

        checkb = np.allclose(g, self.test_data['g_alt'], atol=atol)

        self.assertTrue(checka or checkb)


if __name__ == '__main__':
    unittest.main()
