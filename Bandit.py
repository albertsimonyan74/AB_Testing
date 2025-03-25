"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BANDIT_REWARDS = [1, 2, 3, 4]
NUM_TRIALS = 20000
np.random.seed(42)

class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():

    def plot1(self):
        # Visualize the performance of each bandit: linear and log
        pass

    def plot2(self):
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        pass

#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, p):
        self.bandits = [{'true_mean': mu, 'mean': 0, 'N': 0} for mu in p]
        self.epsilon = 1.0
        self.rewards = []
        self.regrets = []
        self.name = "EpsilonGreedy"

    def __repr__(self):
        return f"EpsilonGreedy with {len(self.bandits)} bandits"

    def pull(self, bandit):
        return np.random.randn() + bandit['true_mean']

    def update(self, bandit, x):
        bandit['N'] += 1
        bandit['mean'] += (x - bandit['mean']) / bandit['N']

    def experiment(self):
        best_mean = max([b['true_mean'] for b in self.bandits])
        for t in range(1, NUM_TRIALS + 1):
            self.epsilon = 1 / t
            if np.random.random() < self.epsilon:
                i = np.random.randint(len(self.bandits))
            else:
                i = np.argmax([b['mean'] for b in self.bandits])

            x = self.pull(self.bandits[i])
            self.update(self.bandits[i], x)
            self.rewards.append((i, x, self.name))
            self.regrets.append(best_mean - self.bandits[i]['true_mean'])

    def report(self):
        df = pd.DataFrame(self.rewards, columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv("epsilon_greedy_rewards.csv", index=False)
        logger.info(f"[EpsilonGreedy] Cumulative Reward: {sum(df['Reward'])}")
        logger.info(f"[EpsilonGreedy] Cumulative Regret: {sum(self.regrets)}")

#--------------------------------------#

class ThompsonSampling(Bandit):
    pass




def comparison():
    # think of a way to compare the performances of the two algorithms VISUALLY and 
    pass

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
