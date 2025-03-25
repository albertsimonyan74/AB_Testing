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
    def __init__(self, p):
        self.bandits = [{'true_mean': mu, 'lambda': 1, 'tau': 1, 'm': 0, 'N': 0, 'sum_x': 0} for mu in p]
        self.rewards = []
        self.regrets = []
        self.name = "ThompsonSampling"

    def __repr__(self):
        return f"ThompsonSampling with {len(self.bandits)} bandits"

    def pull(self, bandit):
        return np.random.randn() + bandit['true_mean']

    def update(self, bandit, x):
        bandit['lambda'] += bandit['tau']
        bandit['sum_x'] += x
        bandit['m'] = bandit['tau'] * bandit['sum_x'] / bandit['lambda']
        bandit['N'] += 1

    def experiment(self):
        best_mean = max([b['true_mean'] for b in self.bandits])
        for _ in range(NUM_TRIALS):
            samples = [np.random.randn() / np.sqrt(b['lambda']) + b['m'] for b in self.bandits]
            i = np.argmax(samples)
            x = self.pull(self.bandits[i])
            self.update(self.bandits[i], x)
            self.rewards.append((i, x, self.name))
            self.regrets.append(best_mean - self.bandits[i]['true_mean'])

    def report(self):
        df = pd.DataFrame(self.rewards, columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv("thompson_sampling_rewards.csv", index=False)
        logger.info(f"[ThompsonSampling] Cumulative Reward: {sum(df['Reward'])}")
        logger.info(f"[ThompsonSampling] Cumulative Regret: {sum(self.regrets)}")

class Visualization():
    def __init__(self):
        self.eg_df = pd.read_csv("epsilon_greedy_rewards.csv")
        self.ts_df = pd.read_csv("thompson_sampling_rewards.csv")

    def plot1(self):
        for df, label in [(self.eg_df, 'Epsilon Greedy'), (self.ts_df, 'Thompson Sampling')]:
            bandit_counts = df.groupby('Bandit').cumcount()
            df['Cumulative Reward'] = df['Reward'].cumsum()
            plt.figure()
            plt.plot(df['Cumulative Reward'], label=f'{label} - Linear')
            plt.yscale('linear')
            plt.title(f"{label} - Linear Scale")
            plt.xlabel("Trials")
            plt.ylabel("Cumulative Reward")
            plt.legend()
            plt.grid()
            plt.show()

            plt.figure()
            plt.plot(df['Cumulative Reward'], label=f'{label} - Log')
            plt.yscale('log')
            plt.title(f"{label} - Log Scale")
            plt.xlabel("Trials")
            plt.ylabel("Cumulative Reward (log)")
            plt.legend()
            plt.grid()
            plt.show()

    def plot2(self):
        eg_rewards = self.eg_df['Reward'].cumsum()
        ts_rewards = self.ts_df['Reward'].cumsum()

        plt.plot(eg_rewards, label='Epsilon Greedy')
        plt.plot(ts_rewards, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.grid()
        plt.show()



#--------------------------------------#


def comparison():
    eg = EpsilonGreedy(BANDIT_REWARDS)
    ts = ThompsonSampling(BANDIT_REWARDS)

    eg.experiment()
    ts.experiment()

    eg.report()
    ts.report()

    vis = Visualization()
    vis.plot1()
    vis.plot2()

if __name__=='__main__':
    logger.info("Starting bandit simulations...")
    comparison()
    logger.info("Experiments complete.")

