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


#Hilfsfunktionen
def plot_list(l):
    for i in range(0,len(l)):
        plt.plot(l[i])

#Experiments
#number of reps
reps = 10

###Two arm bandit

#Single run:
bandit = stochastic_bandit(arms=[bernoulli(0.2), bernoulli(0.7)],
                                 expected_values = [0.2, 0.7])
env1 = env(bandit, ucb_forecaster())
#env = env(bandit, random_forecaster)

env1.play_round(rounds=12)
res = env1.export_results("results_stochastic_bandit/Bernoulli_Experiment_12.csv")


#Many runs development of regret and pseudo_regret
def increase_T(max_T, env):
    regret = np.zeros(max_T)
    pseudo_regret = np.zeros(max_T)
    for T in range(1, max_T+1):
        env.play_many_rounds(T, reps, True)
        pseudo_regret[T-1]=env.mean_pseudo_regret/T
        regret[T-1]=env.get_regret()/T
    return(regret, pseudo_regret)
    
regret, pseudo_regret = increase_T(100, env1)
        
plot_list(increase_T(100, env1))








