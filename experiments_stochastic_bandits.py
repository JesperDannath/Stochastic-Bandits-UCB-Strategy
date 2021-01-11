# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:21:07 2020

@author: Jesper
"""
#imports
from stochastic_bandit import stochastic_bandit
from stochastic_bandit import simulation_environment as env
from stochastic_bandit import ucb_forecaster
from stochastic_bandit import random_forecaster
from stochastic_bandit import bernoulli
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#Resultfolder
folder = "results_stochastic_bandit/"


#Hilfsfunktionen
def plot_list(l, title="", xlabel="", ylabel="", labels="",
              axis_data=[]):
    plt.close()
    if len(axis_data)==0:
        axis_data=np.arange(0, len(l[0]), step = int(len(l[0])/5))
    for i in range(0,len(l)):
        #plt.plot(np.stack((l[i], axis_data), axis=1))
        plt.plot(l[i])
    plt.title(title, fontdict={"fontweight": "bold"})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(labels=labels)
    #plt.xticks(np.arange(0,len(l[0]), step=int(len(l[0])/len(axis_data))),
    #           labels=axis_data)


#Experiments
#number of reps
reps = 200

###Two arm bandit

#Single run:
bandit = stochastic_bandit(arms=[bernoulli(0.2), bernoulli(0.7)],
                                 expected_values = [0.2, 0.7])
env1 = env(bandit, ucb_forecaster())
#env = env(bandit, random_forecaster)

env1.play_round(rounds=100)

env1.play_many_rounds(rounds=100, repetitions=100, log_pseudo_regret=True)


env1.forecaster.return_theoretic_bound([0.2,0.7], 12)

def export_result_sb(env, file):
    dictionary = {"Player rewards": env.forecaster_rewards,
                     "Player actions": env.actions}
    experiment_data = pd.DataFrame(data=dictionary)
    true_rewards = pd.DataFrame(env.true_rewards,
                                columns = [
                                "True Reward Arm "+str(i) for i in range(0,env.bandit.K)])
    mu_hat = pd.DataFrame(env.forecaster.mu_hat_list,
                          columns = [
                          "Mu Hat Arm "+str(i) for i in range(0,env.bandit.K)])
    psi_s_inverse = pd.DataFrame(env.forecaster.p_s_i_list,
                                 columns = [
                                 "P.S.I "+str(i) for i in range(0,env.bandit.K)])
    experiment_data = pd.concat([experiment_data,
                                 true_rewards,
                                 mu_hat,
                                 psi_s_inverse], axis=1)
    if file != None:
        experiment_data.to_csv(file, index=False)
    return(experiment_data)

res = export_result_sb(env1, file="results_stochastic_bandit/Bernoulli_Experiment_12.csv")


#Many runs development of regret and pseudo_regret
def increase_T(max_T, env, reps=reps, step=1, theory_bound=False):
    regret = []
    pseudo_regret = []
    if theory_bound: bounds = []
    for T in range(1, max_T+1, step):
        env.play_many_rounds(rounds=T, repetitions=reps, log_pseudo_regret=True)
        pseudo_regret.append(env.mean_pseudo_regret/T)
        regret.append(env.get_regret()/T)
        print(T)
        if theory_bound: bounds.append(
                env.forecaster.return_theoretic_bound(env.bandit.expected_values, T)/T)
    if theory_bound: return(regret, pseudo_regret, bounds)
    return(regret, pseudo_regret)
    
regret, pseudo_regret = increase_T(100, env1, step=1)
        
plot_list(increase_T(max_T = 100, env=env1, step=1), 
          title="(Pseudo-) Regret per Timestep for increasing \n Number of Rounds",
          ylabel="Values",
          xlabel="Timesteps",
          labels=["Regret", "Pseudo-Regret"],
          axis_data = np.arange(0, 100, step=20))

plt.savefig(folder+"Increase_t_2class.png")

#80 runs seem to be sufficient for convergence



#Many runs effect of alpha
def increase_alpha(min_alpha, max_alpha,
                   step_size, env, T=40, reps=reps):
    steps = int((max_alpha-min_alpha)/step_size)
    alpha = min_alpha
    log_alpha = np.zeros(steps)
    regret = np.zeros(steps)
    pseudo_regret = np.zeros(steps)
    for step in range(0, steps):
        env.forecaster.set_alpha(alpha)
        env.play_many_rounds(T, reps, True)
        pseudo_regret[step]=env.mean_pseudo_regret
        regret[step]=env.get_regret()
        log_alpha[step]=alpha
        alpha += step_size
    return(regret, pseudo_regret, log_alpha)
    
alpha_incr_2class = increase_alpha(min_alpha=0.1, max_alpha=4,
                                   step_size=0.1, env=env1)

plot_list(alpha_incr_2class[0:2],
          title="Effect of Increasing alpha",
          labels=["Regret", "Pseudo-Regret"],
          axis_data=np.arange(0.1,
                              4.0,
                              step=0.5))
plt.savefig(folder+"Increase_alpha_2class.png")
#0.5 seems to be a good alpha value

#Many runs effect of p-gap
def increase_p_gap(step_size, env, T=40, reps=reps):
    steps = int(1/step_size)
    regret = np.zeros(steps)
    pseudo_regret = np.zeros(steps)
    gap = np.zeros(steps)
    p_suboptimal=0.5
    p_optimal = 0.5
    for step in range(0, steps):
        env.bandit.set_arms(
                arms=[bernoulli(p_suboptimal),
                       bernoulli(p_optimal)],
                       expected_values=[p_suboptimal, p_optimal])
        env.play_many_rounds(T, reps, True)
        pseudo_regret[step]=env.mean_pseudo_regret
        regret[step]=env.get_regret()
        p_suboptimal -= step_size/2
        p_optimal += step_size/2
        gap[step] = p_optimal-p_suboptimal
    return(regret, pseudo_regret, gap)
    
incr_pgap_2class = increase_p_gap(0.01, env1)
    
plot_list(incr_pgap_2class[0:2],
          title="Effect of increasing p-gap",
          labels=["Regret", "Pseudo-Regret"],
          axis_data=np.arange(0.01, 1, step=0.5))
plt.savefig(folder+"Increase_p_gap_2class.png")



#Experiments Vaccine Example
arms_vaccines = [bernoulli(0.99), bernoulli(0.9945), bernoulli(0.995)]
bandit_vac = stochastic_bandit(arms=arms_vaccines, 
                               expected_values=[0.99, 0.9945, 0.995])
env_vac = env(bandit_vac, ucb_forecaster(alpha=2.1)) 

#env_vac.play_many_rounds(rounds=100000, repetitions=100, log_pseudo_regret=True)

#env_vac.play_round(5700000)
#Regret: 2629
#Pseudo-Regret: 2730.454

#Test Against random forecaster
env_vac_rand = env(bandit_vac, random_forecaster())
#env_vac_rand.play_round(5700000)
#Regret: 10260
#Pseudo_regret: 10449.534

#Increase T:
incr_T_vac = increase_T(100000, env_vac, reps=30, step=5000, theory_bound=True)


plot_list(incr_T_vac,
          title="Development of theoretical bound over time steps \n in Comparison with real (Pseudo-) Regret",
          labels=["Regret", "Pseudo-Regret", "Theoretical Bound"],
          axis_data=np.arange(0, 100000, step=20000),
          xlabel="Environment Steps",
          ylabel="(Pseudo-) Regret per time-step")

#plt.savefig(folder+"Increase_T_Vaccinations_bound.png")


