#!/usr/bin/env python
"""Tests for `contextual_bandit` module."""

import unittest

from bandit_algorithms import contextual_bandit

class TestBandit_algorithms(unittest.TestCase):
    """Tests for `ContextualBanditAlgorithm` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_set_iteration_number(self):
        """Test set_iteration_number."""
        model = contextual_bandit.ContextualBanditAlgorithm(n_features=2)
        model.set_iteration_number(2)
        self.assertEqual(model.get_iteration_number(),2)

        with self.assertRaises(AssertionError):
            model.set_iteration_number(0.33)
        
        with self.assertRaises(AssertionError):
            model.set_iteration_number(0)
