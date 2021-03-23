"""Super class of contextual bandit algorithm agent class"""
import numpy as np

class ContextualBanditAlgorithm(object):
    """
    Args :
        n_features : 特徴量の次元数

    Attributes:
        iter_num(int) : 現在の反復回数
    """
    def __init__(self, n_features:int):
        self.n_features = n_features

        self.iter_num = 0

    def get_iteration_number(self):
        return self.iter_num

    def set_iteration_number(self, t:int):
        """Setter of iteration 回数"""
        # t が0以下の値の場合、エラーを返す
        assert t > 0, "iteration number must be positive. t = {0}".format(t)
        self.iter_num = t

if __name__ == '__main__':
    pass