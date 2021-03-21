#!/usr/bin/env python
"""Tests for `LinUCBAgent` package."""

import unittest

import numpy as np

from bandit_algorithms.linUCB import LinUCBAgent

class TestLinUCBAgent(unittest.TestCase):
    """Tests for `bandit_algorithms` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.n_arms = 3
        self.n_features = 2
        self.sigma = 0.1
        self.sigma_0 = 1
        self.model = LinUCBAgent(n_arms=self.n_arms, n_features=self.n_features, 
                                 sigma=self.sigma, sigma_0=self.sigma_0)

        np.random.seed(0)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_initialize(self):
        """Test initializing."""
        self.assertEqual(self.model.n_arms, self.n_arms)
        self.assertEqual(self.model.n_features, self.n_features)
        self.assertEqual(self.model.sigma, self.sigma)
        
        # initialize
        self.assertTrue((self.model.A_inv == (self.sigma_0**2 / self.sigma**2) * np.identity(self.n_features)).all())
        self.assertTrue((self.model.b == np.zeros(self.n_features)).all())
        
        # check size
        self.assertEqual(self.model.A_inv.shape[0], self.n_features)
        self.assertEqual(self.model.A_inv.shape[1], self.n_features)
        self.assertEqual(self.model.b.shape[0], self.n_features)
        
        # b はベクトルのはずなので2次元目が存在したらエラー
        try:
            self.model.b.shape[1]
        except:
            pass
        else:
            self.assertTrue(False, "LinUCB.theta has column!")

    def set_iter_num(self, iter_num):
        """反復回数をセットしたモデルを返す"""
        model = self.model
        model.set_iteration_number(iter_num)
        return model

    def test_calc_UCBScore(self):
        """Test calc_UCBScore"""
        # parameter settings        
        context = np.array([[1,2], [4,5], [7,8]])

        model = self.set_iter_num(2)
        ucb_score = model.calc_UCBScore(context)
        
        self.assertEqual(len(ucb_score), self.n_arms)
        
        # 反復回数が0以下だった時の例外処理
        try:
            model = self.set_iter_num(0)
            model.calc_UCBScore(context)
        except:
            pass
        else:
            self.assertTrue(False, "0 divide exception denied.")

    def test_get_arm(self):
        """Test get_arm"""
        # parameter settings
        model = self.set_iter_num(2)
        context = np.array([[1,2], [4,5], [7,8]])
        ucb_score = self.model.calc_UCBScore(context)
        selected_arm = self.model.get_arm(context)
        self.assertEqual(selected_arm, np.argmax(ucb_score))

    def test_update(self):
        """Test update"""
        # parameter settings
        model = self.set_iter_num(2)
        context = np.array([[1,2], [4,5], [7,8]])
        selected_arm = self.model.get_arm(context)
        
        reward_list = [0.1, 0.4, 0.3]
        reward = reward_list[selected_arm]
        
        model.update(context[selected_arm], reward)
        
        self.assertEqual(model.A_inv.shape[0], self.n_features)
        self.assertEqual(model.A_inv.shape[1], self.n_features)
        self.assertEqual(model.b.shape[0], self.n_features)
        
        # b はベクトルのはずなので2次元目が存在したらエラー
        try:
            model.b.shape[1]
        except:
            pass
        else:
            self.assertTrue(False, "LinUCB.b has column!")