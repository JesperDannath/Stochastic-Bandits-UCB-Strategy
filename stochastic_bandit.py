# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:29:42 2020

@author: Jesper Dannath
"""
#Import
import numpy as np
import random
from scipy.stats import bernoulli as ber

#Firstly, the Bandit and Forecaster Environment is created
class simulation_environment():
    
    def __init__(self, bandit, forecaster):
        self.bandit = bandit
        self.forecaster = forecaster
        self.forecaster_rewards = np.empty(1)
        self.true_rewards = np.empty(1)
        self.actions = np.empty(1)
        
    def reset_values(self, rounds):
        self.forecaster_rewards = np.zeros(rounds)
        self.mu_hat = np.zeros((rounds, self.bandit.K))
        self.psi_s_inverse = np.zeros((rounds, self.bandit.K))
        self.true_rewards = np.zeros((rounds, self.bandit.K))
            
    def play_round(self, rounds, factor=1):
        if factor==1: 
            self.reset_values(rounds)
            self.forecaster.reset_rounds(rounds, self.bandit.K, 1)
        true_rewards = self.bandit.generate_reward_table(rounds)
        forecaster_rewards = np.zeros(rounds)
        self.actions = np.zeros(rounds)
        #Iterate through time
        for t in range(0, rounds):
            #Forecaster
            #!!! Median-Action?
            action = self.forecaster.predict(
                    K=self.bandit.K,
                    rewards=forecaster_rewards[0:t],
                    actions=self.actions[0:t])
            self.actions[t] = action
            forecaster_rewards[t]+=true_rewards[t,action]
        self.true_rewards += np.multiply(true_rewards, factor)
        self.forecaster_rewards += np.multiply(forecaster_rewards, factor)
            #log calculations
            
    def play_many_rounds(self, rounds, repetitions, log_pseudo_regret=False):
        self.reset_values(rounds)
        self.forecaster.reset_rounds(rounds, self.bandit.K, repetitions)
        pseudo_regret = 0
        for i in range(0, repetitions):
            self.play_round(rounds, factor=1/repetitions)
            if log_pseudo_regret:
                pseudo_regret += self.get_pseudo_regret( #!!! muss noch überprüft werden
                            self.bandit.expected_values)*(1/repetitions)
        self.mean_pseudo_regret = pseudo_regret   
            
    def get_regret(self):
        forcaster_reward = np.sum(self.forecaster_rewards)
        max_arm_reward = np.max(np.sum(self.true_rewards, axis=0))
        regret = max_arm_reward - forcaster_reward
        return(regret)
        
    #Get the pseudo regret from the current sequence of actions
    #expected_values: list of expected Distribution values in correct order
    #of arms.
    def get_pseudo_regret(self, expected_values):
        pseudo_regret = len(self.actions)*np.max(np.asarray(expected_values))
        for action in self.actions:
            pseudo_regret -= expected_values[action.astype(int)]
        return(pseudo_regret)
    

#Now, we define the stochastic bandit
class stochastic_bandit():
#!!! my_hat  und psi hier loggen!    
    def __init__(self, arms, expected_values=None, shedule=False):
        #The list of arms for the bandit
        self.arms = arms
        #Number of arms
        self.K = len(arms)
        #Timestep (only important for sheduled arms)
        self.t = 1
        self.shedule=shedule
        self.expected_values = expected_values
        
    def set_arms(self, arms, expected_values=None):
        self.arms=arms
        self.expected_values=expected_values
    
    #Draws a value from the Distribution or shedule of an arm    
    def draw(self, arm, time=None):
        if self.shedule==False:
            return(self.arms[arm].random())
        else:
            return(self.arms[arm].shedule(time))
        return()

    #Genrate the table of true rewards per arm and round    
    def generate_reward_table(self,rounds):
        true_rewards = np.zeros((self.K, rounds))
        #Iterate through arms
        for k in range(0, self.K):
            true_rewards[k] = self.arms[k].random(size=rounds)
        return(np.transpose(true_rewards))
    
#arms/distributions
class bernoulli():  
    
    def __init__(self, p):
        #p-Parameter
        self.p = p
        #Scipy Distribution
        self.dist = ber(p=self.p)
    
    def random(self, size=1):
        return(self.dist.rvs(size=size))
        

#Forecasters will be Functions, that can view the past rewards
#and actions (choosen arms)

#Random forecaster
class random_forecaster():
    
    def reset_rounds(self, rounds, K, repetitions):
        None
    
    def predict(self, K, rewards, actions):
        return(random.randint(0,K-1))
    
    
#UCB-Forecaster which has sqrt(x/2) as default psi-star inverse
class ucb_forecaster():
    
    def __init__(self, psi_star_inverse=lambda x: np.sqrt(x/2), alpha=1):
        self.psi_star_inverse = psi_star_inverse
        self.alpha = alpha
        self.rounds=1
        
    def set_alpha(self, alpha):
        self.alpha=alpha
        
    def reset_rounds(self, rounds, K, reps):
        self.rounds=rounds
        self.reps=reps
        self.K = K
        self.mu_hat_list = np.zeros((self.rounds, self.K))
        self.p_s_i_list = np.zeros((self.rounds, self.K))
    
    def predict(self, K, rewards, actions):
        t = len(rewards)+1
        mu_hat_list = np.zeros(K, dtype=float)
        p_s_i_list = np.zeros(K, dtype=float)
        for i in range(0,K):
            #Calulate mu_hat
            x_indicator = np.where(actions==i)
            x = rewards[x_indicator]
            mu_hat_list[i] = np.mean(x) if not x.size==0 else 0.0   
            #Calculate psi-star-reversed of (alpha*ln(t)/n_i_selected)
            #n_i_selected - this can be zero!!!  
            #This means we go to infinity and explore those values fast!!
            if len(x_indicator[0])==0:
                p_s_i_list[i] = np.inf
            else:
                p_s_i_list[i] =  self.psi_star_inverse(self.alpha*np.log(t)/len(x_indicator[0]))
            #print(x_indicator)
        bounds = np.add(mu_hat_list, p_s_i_list)
        maximum = np.argwhere(bounds == np.amax(bounds)).flatten()
        #Treat case of multiple maximas
        if len(maximum)==1:
            maximum = maximum[0]
        else:
            maximum = maximum[random.randint(0,len(maximum)-1)]
        self.mu_hat_list[t-1]+=np.multiply(mu_hat_list, (1/self.reps))
        self.p_s_i_list[t-1]+=np.multiply(p_s_i_list, (1/self.reps))
        return(maximum)
        
    

        
