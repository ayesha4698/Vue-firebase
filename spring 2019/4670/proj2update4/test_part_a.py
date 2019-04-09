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
class TestImageProcessing(unittest.TestCase):
	def setUp(self):
		self.test_data = np.load('part_a_test_data.npz')
		self.small_img = self.test_data['small_img'] # 64 x 64 random data
		self.big_img = self.test_data['big_img'] # 128 x 128 x 3 random data

	def test_normalize_image_grey(self):
		img_dup = tg.normalizeImage(self.small_img, 0,1,0,1)
		self.assertTrue(np.allclose(img_dup, self.test_data['small_img_normalized'], atol=atol))


	def test_normalize_image_color(self):

		img_dup = tg.normalizeImage(self.big_img, 0,1,0,1)
		self.assertTrue(np.allclose(img_dup, self.test_data['big_img_normalized'], atol=atol))


	def test_small_x_gradient(self):
		img_dup = tg.takeXGradient(self.small_img)
		self.assertTrue(np.allclose(img_dup, self.test_data['small_img_x_gradient'], atol=atol))

	def test_big_x_gradient(self):
		img_dup = tg.takeXGradient(self.big_img)
		self.assertTrue(np.allclose(img_dup, self.test_data['big_img_x_gradient'], atol=atol))

	def test_small_y_gradient(self):
		img_dup = tg.takeYGradient(self.small_img)
		self.assertTrue(np.allclose(img_dup, self.test_data['small_img_y_gradient'], atol=atol))

	def test_big_y_gradient(self):
		img_dup = tg.takeYGradient(self.big_img)
		self.assertTrue(np.allclose(img_dup, self.test_data['big_img_y_gradient'], atol=atol))

	def test_small_gradient_mag(self):
		img_dup = tg.takeGradientMag(self.small_img)
		self.assertTrue(np.allclose(img_dup, self.test_data['small_img_gradient_mag'], atol=atol))

	def test_big_gradient_mag(self):
		img_dup = tg.takeGradientMag(self.big_img)


		self.assertTrue(np.allclose(img_dup, self.test_data['big_img_gradient_mag'], atol=atol))

	def test_small_display_gradient(self):


		img_dup = tg.getDisplayGradient(tg.takeGradientMag(self.small_img))

		self.assertTrue(np.allclose(img_dup, self.test_data['small_img_gradient_display'], atol=atol))

	def test_big_display_gradient(self):
		img_dup = tg.getDisplayGradient(tg.takeGradientMag(self.big_img))
		self.assertTrue(np.allclose(img_dup, self.test_data['big_img_gradient_display'], atol=atol))


if __name__ == '__main__':
	unittest.main()
