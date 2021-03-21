#!/usr/bin/env python
"""Tests for `LogisticTSAgent` package."""

import unittest

import numpy as np

from bandit_algorithms.logistic_TS import LogisticTSAgent

class TestLogisticTSAgent(unittest.TestCase):
    """Tests for `bandit_algorithms` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.n_features = 2
        self.interval_update_theta = 100
        self.model = LogisticTSAgent(self.n_features, interval_update_theta=self.interval_update_theta)
        np.random.seed(0)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_initialize(self):
        """Test initializing."""
        n_features = 4
        sigma_0 = 0.2
        num_theta_max_iter=100
        epsilon=0.1
        interval_update_theta=10
        
        model = LogisticTSAgent(n_features, sigma_0, num_theta_max_iter, epsilon, interval_update_theta)
        
        # your settings
        self.assertEqual(model.n_features, n_features)
        self.assertEqual(model.sigma_0, sigma_0)
        self.assertEqual(model.num_theta_max_iter, num_theta_max_iter)
        self.assertEqual(model.epsilon, epsilon)
        self.assertEqual(model.interval_update_theta, interval_update_theta)
        
        # initialize
        self.assertTrue((model.theta == np.zeros(n_features)).all())
        self.assertTrue((model.H_inv == np.linalg.inv(np.identity(n_features)/sigma_0**2)).all())
        
        # check size
        self.assertEqual(model.theta.shape[0], n_features)
        
        # theta はベクトルのはずなので2次元目が存在したらエラー
        try:
            model.theta.shape[1]
        except:
            pass
        else:
            self.assertTrue(False, "LogisticTSAgent.theta has column!")

    def test_get_theta_by_normalDistribution(self):
        """Test get_theta_by_normalDistribution"""
        theta_tild = self.model.get_theta_by_normalDistribution()
        
        self.assertEqual(theta_tild.shape[0], self.n_features)

        # theta はベクトルのはずなので2次元目が存在したらエラー
        try:
            theta_tild.shape[1]
        except:
            pass
        else:
            self.assertTrue(False, "theta_tild has column!")

    def test_get_arm(self):
        """Test get_arm"""        
        context = np.array([[1,2], [4,5], [7,8]])
        selected_arm = self.model.get_arm(context)
    
    def test_sample(self):
        """Test update"""        
        model = self.model
        context = np.array([[1,2], [4,5], [7,8]])
        reward_list = [0,0,1]
        selected_arm = model.get_arm(context)
        # interval_update_theta 回反復していないと更新しないので、反復回数を疑似的に変更する
        model.set_iteration_number(self.interval_update_theta-1)
        model.sample(context[selected_arm], reward_list[selected_arm])
        self.assertFalse((model.theta == np.zeros(self.n_features)).all())