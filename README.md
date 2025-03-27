# A/B Testing with Multi-Armed Bandits

This project implements and compares two bandit algorithms — **Epsilon-Greedy** and **Thompson Sampling** — for A/B testing over 20,000 trials with four advertisement options (bandits). Results are visualized and saved for evaluation.

---

## 📁 Folder Structure

```
AB_Testing/
├── venv/                            # Python virtual environment
├── .gitignore
├── .python-version                 # Python version manager file
├── Bandit.py                       # Main experiment code
├── bonus.txt                       # Bonus question/implementation
├── epsilon_greedy_rewards.csv      # Epsilon-Greedy results
├── thompson_sampling_rewards.csv   # Thompson Sampling results
├── requirements.txt                # Project dependencies
└── README.md                       # Project overview (this file)
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/albertsimonyan74/AB_Testing.git
cd AB_Testing
```

### 2. Install `pyenv` and Python 3.10 (if not already installed)

If you don't have `pyenv`:
```bash
brew install pyenv
```

Install Python 3.10:
```bash
pyenv install 3.10.13
pyenv local 3.10.13
```

---

## 🐍 Setup the Virtual Environment

### 1. Create and activate the virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Experiment

To run the bandit experiment and generate the results:
```bash
python Bandit.py
```

This will:
- Run **Epsilon-Greedy** and **Thompson Sampling** algorithms
- Generate and save rewards to `epsilon_greedy_rewards.csv` and `thompson_sampling_rewards.csv`
- Display visualizations of learning progress and performance

---

## 📊 Outputs

- `epsilon_greedy_rewards.csv`: Stores each trial’s reward and bandit for Epsilon-Greedy
- `thompson_sampling_rewards.csv`: Stores each trial’s reward and bandit for Thompson Sampling
- Visual plots (linear/log scale and cumulative comparisons) are shown in pop-up windows

---

## 🔚 Deactivate the Environment

When you're done:
```bash
deactivate
```

---

## 🧠 Bonus

See `bonus.txt` for insights on why UCB1 could be a better implementation plan.
