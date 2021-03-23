"""Logistic Thompson Sampling agent class"""

import numpy as np

from bandit_algorithms.contextual_bandit import ContextualBanditAlgorithm

class LogisticTSAgent(ContextualBanditAlgorithm):
    """
    Args :
        sigma_0(float)             : theta の事前分布の分散
        num_theta_max_iter(int)    : theta の探索を行う反復回数の上限
        epsilon(float)             : theta の探索を行う際の収束許容誤差
        interval_update_theta(int) : thetaの更新を行う間隔. この回数だけ反復したら theta, hessian を更新する

    Attributes:
        theta(np.array)                     : 各特徴量に対するパラメータ. n_features*1
        lst_context_history(list[np.array]) : 引いた腕のcontextリスト
        lst_reward_history(list[int])       : 引いた腕の報酬リスト. 0 or 1
        H_inv(np.array)                     : 乱数生成に必要なヘシアンの逆行列
    """
    def __init__(self, n_features,
                 sigma_0=0.1, num_theta_max_iter=1000, epsilon=0.01, interval_update_theta=100):
        super().__init__(n_features)

        self.sigma_0 = sigma_0
        self.num_theta_max_iter = num_theta_max_iter
        self.epsilon = epsilon
        self.interval_update_theta = interval_update_theta

        self.theta  = np.zeros(n_features)
        self.lst_context_history = []
        self.lst_reward_history = []
        self.H_inv = self.calc_hessian_inv(self.theta)
    
    def logistic(self, x):
        """logistic 関数
        
        xが小さすぎるとオーバーフローを起こすため、-500より小さければ0とする
        """
        if x < -500:
            return 0
        else:
            return 1 / (1+np.exp(-x))
        
    def calc_gradient(self, theta):
        """負の対数事後確率の勾配 g(θ) の計算"""
        g = (theta / self.sigma_0**2)

        for idx, context in enumerate(self.lst_context_history):
            g += context * (self.logistic(theta.dot(context)) - self.lst_reward_history[idx])
        
        return g
    
    def calc_hessian_inv(self, theta):
        """負の対数事後確率のヘシアン H(θ) の逆行列の計算"""
        H = (np.identity(self.n_features)/self.sigma_0 ** 2)

        for context in self.lst_context_history:
            val_logistic = self.logistic(theta.dot(context))
            H += context.dot(context.T) * val_logistic * (1 - val_logistic)

        return np.linalg.inv(H)
        
    def set_theta(self, theta):
        """theta の係数格納"""
        self.theta = theta
    
    def set_H_inv(self, H_inv):
        """乱数生成に必要なヘシアンの逆行列格納"""
        self.H_inv = H_inv
        
    def get_theta_by_normalDistribution(self):
        """多変量正規分布からパラメータをサンプリング"""
        return np.random.multivariate_normal(self.theta, self.H_inv)
    
    def get_arm(self, context):
        """腕を選択
        
        Args:
            context(np.array) : 特徴量. n_features*n_arms
        """
        np_context = np.array(context)
        theta_tild = self.get_theta_by_normalDistribution()
        
        score = np_context.dot(theta_tild)
        return np.argmax(score)
        
    def add_reward(self, reward):
        """得られた報酬のリストに追加"""
        self.lst_reward_history.append(reward)
        
    def add_context(self, context):
        """選択した腕のcontextをリストに追加"""
        self.lst_context_history.append(np.array(context))
    
    def update_theta(self):
        """観測した報酬からtheta更新"""
        theta_hat = self.theta
        
        # thetaが収束するか規定回数を反復するまで計算する
        for i in range(self.num_theta_max_iter):
            theta_before = theta_hat
            g = self.calc_gradient(theta_hat)
            H_inv = self.calc_hessian_inv(theta_hat)
            
            theta_hat = theta_hat - H_inv.dot(g)
            
            # 収束したら終了
            ## 収束しなかったらthetaを更新して再計算
            if np.linalg.norm(theta_hat - theta_before) < self.epsilon:
                break
                
        # 規定回数までに収束しなかったら表示
        if i == self.num_theta_max_iter-1:
            print("max iter over")
            
        # 収束したら更新
        self.set_theta(theta_hat)
        self.set_H_inv(self.calc_hessian_inv(theta_hat))
    
    def sample(self, selected_context, reward):
        """選択した腕の文脈とその報酬を得て、更新を行う"""
        self.set_iteration_number(self.get_iteration_number()+1)
        self.add_reward(reward)
        self.add_context(selected_context)
        # 設定した更新間隔に達していればthetaを更新する
        if self.get_iteration_number() % self.interval_update_theta == 0:
            self.update_theta()