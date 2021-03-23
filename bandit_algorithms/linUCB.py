"""Linear Upper Confidence Bound policy agent class"""

import numpy as np
from bandit_algorithms.contextual_bandit import ContextualBanditAlgorithm

class LinUCBAgent(ContextualBanditAlgorithm):
    """
    Args :
        n_arms(int)     : 引ける腕の数
        sigma(float)    : 誤差項の分散
        alpha(float)    : UCBスコアにかかる係数
        sigma_0(float)  : theta の事前分布の分散

    Attributes:
        A_inv(np.array) : UCBスコアを計算するためのパラメータ. n_features*n_features
        b(np.array)     : UCBスコアを計算するためのパラメータ. n_features*1
    """
    def __init__(self, n_features, n_arms,
                 sigma=1, alpha=0.01, sigma_0=0.1):
        super().__init__(n_features)

        self.n_arms = n_arms

        self.sigma  = sigma
        self.alpha = alpha
        self.sigma_0 = sigma_0

        self.A_inv  = (self.sigma_0**2 / sigma**2) * np.identity(n_features)
        self.b      = np.zeros(n_features)
        
    def get_features(self, context):
        """入力されるcontext がどのような型でも計算できるように、numpy型に変換して返す
        
        Args:
            context : 入力する文脈. 属性情報とかそれまでの選択とか

        Returns:
            np.array : np.array型に変換した文脈
        """
        return np.array(context)

    def calc_posterior_parameters(self):
        """各特徴量に対する係数の事後分布のパラメータを算出

        Returns:
            1. np.array : 事後正規分布の平均. n_features*1
            2. np.array : 事後正規分布の分散. n_features*n_features
        """
        return self.A_inv.dot(self.b), self.A_inv

    def predict_normalDistribution(self, context):
        """文脈（特徴量）に対する正規分布の平均と分散を算出
        
        Args:
            context(list[float]) : 入力する文脈. 属性情報とかそれまでの選択とか. n_arms*n_features

        Returns:
            1. np.array : 報酬が従う正規分布の予測平均. n_arms*1
            2. np.array : 報酬が従う正規分布の予測分散. n_arms*n_arms
        """
        features = self.get_features(context)
        post_mean, post_var = self.calc_posterior_parameters()
        return features.dot(post_mean), features.dot(post_var).dot(features.T)
        
    def calc_UCBScore(self, context):
        """UCBスコアの計算
        
        Args:
            context(list[float]) : 入力する文脈. 属性情報とかそれまでの選択とか. n_arms*n_features
        """
        alpha_t = self.alpha * np.sqrt(np.log(self.get_iteration_number()))
        pred_mean, pred_var = self.predict_normalDistribution(context)
        ucb = pred_mean.T + alpha_t * np.sqrt(np.diag(pred_var))
        return ucb
    
    def get_arm(self, context):
        """腕を選択"""
        return np.argmax(self.calc_UCBScore(context))
    
    def update(self, selected_context, reward):
        """観測した報酬からパラメータ更新
        
        Args:
            selected_context(list[float]) : 選択した腕の特徴量. n_features*1
            reward(float)     : 得られた報酬. 報酬は正規分布に従うとしているため、binary でなくともよい
        """
        # 視認性向上のため、context 情報を変数に格納しておく
        a_it = self.get_features(selected_context)
        A_inv_a_it = self.A_inv.dot(a_it)
        A_inv_a_it_a_it_T = self.A_inv.dot(a_it).dot(a_it.T)
        
        # パラメータ更新
        self.A_inv = self.A_inv - np.dot(A_inv_a_it_a_it_T, self.A_inv) / self.sigma ** 2 +a_it.T.dot(A_inv_a_it)
        self.b = self.b + (self.sigma**2)*a_it*reward

    def sample(self, selected_context, reward):
        """選択した腕の文脈とその報酬を得て、更新を行う"""
        self.set_iteration_number(self.get_iteration_number()+1)
        self.update(selected_context, reward)