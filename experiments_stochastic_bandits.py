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
    plt.xticks(np.arange(0,len(l[0]), step=int(len(l[0])/len(axis_data))),
               labels=axis_data)


#Experiments
#number of reps
reps = 300

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
          title="(Pseudo-) Regret per Timestep for increasing \n Number of Rounds, p-gap=0.5, alpha="+str(env1.forecaster.alpha),
          ylabel="Values per Timestep",
          xlabel="Timesteps",
          labels=["Regret", "Pseudo-Regret"],
          axis_data = np.arange(0, 100, step=20))

plt.savefig(folder+"Increase_t_2class.png")

#100 runs seem to be sufficient for convergence
#Now include the theoretic bound
env1.forecaster.set_alpha(0.1)

increase_T_theory_bound = increase_T(max_T = 100, env=env1, step=1, theory_bound=True)

#[np.log(np.array(list)) for list in increase_T_theory_bound]
plot_list([list[0:len(list)] for list in increase_T_theory_bound], 
          title="Theoretic bound and (Pseudo-) Regret per Timestep \n for increasing T, p-gap=0.5, alpha="+str(env1.forecaster.alpha),
          ylabel="Values per Timestep",
          xlabel="Timesteps",
          labels=["Regret", "Pseudo-Regret", "Theoretical bound"],
          axis_data = np.arange(0, 100, step=20))

plt.savefig(folder+"Increase_t_2class_theory_bound_alpha_0.1.png")


#Many runs effect of alpha
def increase_alpha(min_alpha, max_alpha,
                   step_size, env, T=100, reps=reps):
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
          title="Effect of Increasing alpha \n p-gap=0.5, timesteps=100",
          labels=["Regret", "Pseudo-Regret"],
          axis_data=np.arange(0.1,
                              4.0,
                              step=0.5),
          xlabel="alpha")
plt.savefig(folder+"Increase_alpha_2class.png")
#0.5 seems to be a good alpha value

#Many runs effect of p-gap
def increase_p_gap(step_size, env, T=100, reps=reps):
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
    
env1.forecaster.set_alpha(alpha=1)
    
incr_pgap_2class = increase_p_gap(0.01, env1)
    
plot_list(incr_pgap_2class[0:2],
          title="Effect of increasing p-gap \n timesteps=100, alpha="+str(env1.forecaster.alpha) + ", repetetions="+str(reps),
          labels=["Regret", "Pseudo-Regret"],
          axis_data=np.arange(0.01, 1, step=0.5),
          xlabel="p-gap")
plt.savefig(folder+"Increase_p_gap_2class.png")



#Experiments Vaccine Example
arms_vaccines = [bernoulli(0.97), bernoulli(0.994), bernoulli(0.995)]
bandit_vac = stochastic_bandit(arms=arms_vaccines, 
                               expected_values=[0.97, 0.994, 0.995])
env_vac = env(bandit_vac, ucb_forecaster(alpha=2.1)) 

#env_vac.play_many_rounds(rounds=100000, repetitions=100, log_pseudo_regret=True)

tb = env_vac.forecaster.return_theoretic_bound(expected_values=[0.97, 0.994, 0.995],
                                               n=10000000)

env_vac.play_many_rounds(10000000, repetitions=3, log_pseudo_regret=True)
#Regret: 3286.666
#Pseudo-Regret: 3252.047
#Theoretic bound: 70445.841


#Alpha=0.1
env_vac.forecaster.set_alpha(0.1)
env_vac.play_many_rounds(10000000, repetitions=3, log_pseudo_regret=True)
#Regret: 495.666
#Pseudo-regret: 577.151


#Test Against random forecaster
env_vac_rand = env(bandit_vac, random_forecaster())
env_vac_rand.play_many_rounds(10000000, repetitions=3, log_pseudo_regret=True)
#Regret: 86521.666
#Pseudo_regret: 86634.758


#Increase T:
#Alpha = 2.1
incr_T_vac_21 = increase_T(100000, env_vac, reps=30, step=2500, theory_bound=True)


plot_list(incr_T_vac_21,
          title="Theoretical bound over time steps in Comparison \n with real (Pseudo-) Regret, repetitions=30, alpha=" + str(env_vac.forecaster.alpha),
          labels=["Regret", "Pseudo-Regret", "Theoretical Bound"],
          axis_data=np.arange(0, 100000, step=20000),
          xlabel="Environment steps",
          ylabel="(Pseudo-) Regret per time-step")

plt.savefig(folder+"Increase_T_Vaccinations_bound_alpha_2.1.png")


#alpha = 0.1
env_vac.forecaster.set_alpha(0.1)
incr_T_vac_01 = increase_T(100000, env_vac, reps=30, step=2500, theory_bound=True)

plot_list(incr_T_vac_01,
          title="Theoretical bound over time steps in Comparison \n with real (Pseudo-) Regret, repetitions=30, alpha=" + str(env_vac.forecaster.alpha),
          labels=["Regret", "Pseudo-Regret", "Theoretical Bound"],
          axis_data=np.arange(0, 100000, step=20000),
          xlabel="Environment steps",
          ylabel="(Pseudo-) Regret per time-step")

plt.savefig(folder+"Increase_T_Vaccinations_bound_alpha_0.1.png")

#alpha=1
env_vac.forecaster.set_alpha(1.0)
incr_T_vac_01 = increase_T(100000, env_vac, reps=30, step=2500, theory_bound=True)

plot_list(incr_T_vac_01,
          title="Theoretical bound over time steps in Comparison \n with real (Pseudo-) Regret, repetitions=30, alpha=" + str(env_vac.forecaster.alpha),
          labels=["Regret", "Pseudo-Regret", "Theoretical Bound"],
          axis_data=np.arange(0, 100000, step=20000),
          xlabel="Environment steps",
          ylabel="(Pseudo-) Regret per time-step")

plt.savefig(folder+"Increase_T_Vaccinations_bound_alpha_1.png")


#Increase_alpha Vaccines
alpha_incr_vac = increase_alpha(min_alpha=0.1, max_alpha=4,
                                   step_size=0.1, env=env_vac, T=10)


plot_list(alpha_incr_vac[0:2],
          title="Effect of Increasing alpha \n timesteps=10K, repetitions=30",
          labels=["Regret", "Pseudo-Regret"],
          axis_data=np.arange(0.1,
                              4.0,
                              step=0.5),
          xlabel="alpha")
          
plt.savefig(folder+"Increase_alpha_vac.png")











