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
    """
    Abstract base class for bandit strategies.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initialize the bandit strategy.

        Parameters
        ----------
        p : list
            List of true mean rewards for each bandit arm.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Return string representation of the strategy.
        """
        pass

    @abstractmethod
    def pull(self):
        """
        Pull a bandit arm and return a reward.
        """
        pass

    @abstractmethod
    def update(self, x):
        """
        Update the internal state with the received reward.

        Parameters
        ----------
        x : float
            Reward received.
        """
        pass

    @abstractmethod
    def experiment(self):
        """
        Run the full experiment for the bandit strategy.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Print logs and save reward/regret report.
        """
        pass

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy bandit strategy with decaying epsilon.
    """
    def __init__(self, p):
        """
        Initialize epsilon-greedy bandit.

        Parameters
        ----------
        p : list
            List of true mean rewards.
        """
        self.bandits = [{'true_mean': mu, 'mean': 0, 'N': 0} for mu in p]
        self.epsilon = 1.0
        self.rewards = []
        self.regrets = []
        self.name = "EpsilonGreedy"

    def __repr__(self):
        return f"EpsilonGreedy with {len(self.bandits)} bandits"

    def pull(self, bandit):
        """
        Pull a selected bandit arm.

        Parameters
        ----------
        bandit : dict
            The bandit to pull.

        Returns
        -------
        float
            Reward sampled from normal distribution.
        """
        return np.random.randn() + bandit['true_mean']

    def update(self, bandit, x):
        """
        Update the mean estimate for a bandit.

        Parameters
        ----------
        bandit : dict
            The bandit to update.
        x : float
            The observed reward.
        """
        bandit['N'] += 1
        bandit['mean'] += (x - bandit['mean']) / bandit['N']

    def experiment(self):
        """
        Run epsilon-greedy strategy for NUM_TRIALS.
        """
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
        """
        Generate CSV report and log cumulative reward/regret.
        """
        df = pd.DataFrame(self.rewards, columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv("epsilon_greedy_rewards.csv", index=False)
        logger.info(f"[EpsilonGreedy] Cumulative Reward: {sum(df['Reward'])}")
        logger.info(f"[EpsilonGreedy] Cumulative Regret: {sum(self.regrets)}")

#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    Thompson Sampling strategy with known precision.
    """
    def __init__(self, p):
        """
        Initialize Thompson Sampling bandit.

        Parameters
        ----------
        p : list
            List of true mean rewards.
        """
        self.bandits = [{'true_mean': mu, 'lambda': 1, 'tau': 1, 'm': 0, 'N': 0, 'sum_x': 0} for mu in p]
        self.rewards = []
        self.regrets = []
        self.name = "ThompsonSampling"

    def __repr__(self):
        return f"ThompsonSampling with {len(self.bandits)} bandits"

    def pull(self, bandit):
        """
        Pull a bandit arm and receive reward.

        Parameters
        ----------
        bandit : dict
            The bandit to pull.

        Returns
        -------
        float
            Reward.
        """
        return np.random.randn() + bandit['true_mean']

    def update(self, bandit, x):
        """
        Update posterior belief of the bandit.

        Parameters
        ----------
        bandit : dict
            Bandit to update.
        x : float
            Observed reward.
        """
        bandit['lambda'] += bandit['tau']
        bandit['sum_x'] += x
        bandit['m'] = bandit['tau'] * bandit['sum_x'] / bandit['lambda']
        bandit['N'] += 1

    def experiment(self):
        """
        Run Thompson Sampling strategy.
        """
        best_mean = max([b['true_mean'] for b in self.bandits])
        for _ in range(NUM_TRIALS):
            samples = [np.random.randn() / np.sqrt(b['lambda']) + b['m'] for b in self.bandits]
            i = np.argmax(samples)
            x = self.pull(self.bandits[i])
            self.update(self.bandits[i], x)
            self.rewards.append((i, x, self.name))
            self.regrets.append(best_mean - self.bandits[i]['true_mean'])

    def report(self):
        """
        Generate CSV report and log cumulative reward/regret.
        """
        df = pd.DataFrame(self.rewards, columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv("thompson_sampling_rewards.csv", index=False)
        logger.info(f"[ThompsonSampling] Cumulative Reward: {sum(df['Reward'])}")
        logger.info(f"[ThompsonSampling] Cumulative Regret: {sum(self.regrets)}")


#--------------------------------------#

class Visualization:
    """
    Handles the visualization of results for bandit experiments.
    """

    def __init__(self):
        """
        Initialize the Visualization class by loading reward data
        from CSVs generated by EpsilonGreedy and ThompsonSampling experiments.
        """
        self.eg_df = pd.read_csv("epsilon_greedy_rewards.csv")
        self.ts_df = pd.read_csv("thompson_sampling_rewards.csv")

    def plot1(self):
        """
        Plot the cumulative rewards for each algorithm on:
        - Linear scale
        - Logarithmic scale

        Each algorithm is visualized separately using its respective data.
        """
        for df, label in [(self.eg_df, 'Epsilon Greedy'), (self.ts_df, 'Thompson Sampling')]:
            bandit_counts = df.groupby('Bandit').cumcount()
            df['Cumulative Reward'] = df['Reward'].cumsum()

            # Linear scale plot
            plt.figure()
            plt.plot(df['Cumulative Reward'], label=f'{label} - Linear')
            plt.yscale('linear')
            plt.title(f"{label} - Linear Scale")
            plt.xlabel("Trials")
            plt.ylabel("Cumulative Reward")
            plt.legend()
            plt.grid()
            plt.show()

            # Log scale plot
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
        """
        Plot a side-by-side comparison of cumulative rewards between
        EpsilonGreedy and ThompsonSampling algorithms on a single chart.
        """
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
    """
    Run the full comparison experiment between EpsilonGreedy and ThompsonSampling.

    This function:
    - Runs each algorithm
    - Generates reward/regret reports
    - Visualizes results using `Visualization` class
    """
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
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

