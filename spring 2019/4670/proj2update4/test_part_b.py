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

import cv2
import numpy as np

import texture_gradient as tg

atol = 1e-06
class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.test_data = np.load('part_b_test_data.npz')
        self.small_img = self.test_data['small'] # 64 x 64 x 3 random data
        self.large_img = self.test_data['large'] # 128 x 128 x 3 random data
        self.X = self.test_data['X'] # 64 x 3 random feature matrix
        self.C = self.test_data['C'] # 128 x 3 random centroid matrix

    def test_compute_dis(self):
        self.assertTrue(np.allclose(tg.computeDistance(self.X, self.C), self.test_data['compute_dis_small'], atol=atol))

    def test_get_feats_small(self):
        print("our output")
        print(tg.getFeats(self.small_img))
        print("expected")
        print(self.test_data['get_feats_small'])
        self.assertTrue(np.allclose(tg.getFeats(self.small_img), self.test_data['get_feats_small'], atol=atol))

    def test_get_feats_large(self):
        print("our input")
        print(tg.getFeats(self.large_img))
        print("expected")
        print(self.test_data['get_feats_large'])
        self.assertTrue(np.allclose(tg.getFeats(self.large_img), self.test_data['get_feats_large'], atol=atol))

if __name__ == '__main__':
	unittest.main()
