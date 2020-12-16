# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:21:07 2020

@author: Jesper
"""
#imports
from stochastic_bandit import stochastic_bandit
from stochastic_bandit import simulation_environment as env
from stochastic_bandit import ucb_forecaster
from stochastic_bandit import bernoulli
from matplotlib import pyplot as plt

#Experiments

#Single run:
bandit = stochastic_bandit(arms=[bernoulli(0.2), bernoulli(0.7)],
                                 expected_values = [0.2, 0.7])
#env = env(bandit, random_forecaster)
env1 = env(bandit, ucb_forecaster())
env1.play_round(12)
res = env1.export_results("results_stochastic_bandit/Bernoulli_Experiment_12.csv")

print(env1.mu_hat)
print(env1.psi_s_inverse)        

#Many runs development of regret and pseudo_regret
env1.play_many_rounds(12, 100, True)        
        
env1.mean_pseudo_regret