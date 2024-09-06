# analysis.py

import numpy as np
import matplotlib.pyplot as plt

def plot_episode_rewards():
    rewards = np.load("rewards.npy")
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.title("Cumulative Rewards Over Episodes")
    plt.show()

def plot_epsilon_decay(epsilon, epsilon_decay, epsilon_min, no_episodes):
    epsilon_values = [max(epsilon_min, epsilon_decay**episode * epsilon) for episode in range(no_episodes)]
    plt.plot(epsilon_values)
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon Value")
    plt.title("Epsilon Decay Over Episodes")
    plt.show()

# def plot_q_value_heatmaps():
#     Q = np.load("q_table.npy")
#     actions = ['Up', 'Down', 'Left', 'Right']
    
#     fig, ax = plt.subplots(2, 2, figsize=(9, 9))
#     for i in range(4):
#         row = i // 2
#         col = i % 2
#         cax = ax[row, col].matshow(Q[:, :, i], cmap='viridis')
#         for x in range(Q.shape[0]):
#             for y in range(Q.shape[1]):
#                 ax[row, col].text(y, x, f'{Q[x, y, i]:.2f}', va='center', ha='center', color='white', fontsize=8)
#         fig.colorbar(cax, ax=ax[row, col])
#         ax[row, col].set_title(f'Q-value for action: {actions[i]}')
#     plt.tight_layout()
#     plt.show()

def plot_optimal_policy():
    Q = np.load("q_table.npy")
    optimal_policy = np.argmax(Q, axis=2)
    
    fig, ax = plt.subplots()
    ax.matshow(np.zeros((7, 7)), cmap='gray_r')  # Empty grid for visualizing policy
    for i in range(7):
        for j in range(7):
            action = optimal_policy[i, j]
            if action == 0:  # Up
                ax.arrow(j, i, 0, -0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            elif action == 1:  # Down
                ax.arrow(j, i, 0, 0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            elif action == 2:  # Left
                ax.arrow(j, i, -0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            elif action == 3:  # Right
                ax.arrow(j, i, 0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    plt.title('Optimal Policy Visualization')
    plt.show()

def plot_q_value_convergence():
    Q_values_over_time = np.load("Q_values_over_time.npy")  # Assuming you save Q-values at intervals
    q_diff = [np.sum(np.abs(Q_values_over_time[i+1] - Q_values_over_time[i])) for i in range(len(Q_values_over_time)-1)]
    
    plt.plot(q_diff)
    plt.xlabel("Intervals")
    plt.ylabel("Sum of Absolute Q-value Differences")
    plt.title("Q-value Convergence Over Time")
    plt.show()

def plot_action_distribution():
    actions_taken = np.load("actions_taken.npy")  # Assuming you save actions taken
    action_counts = np.bincount(actions_taken, minlength=4)
    
    plt.bar(['Up', 'Down', 'Left', 'Right'], action_counts)
    plt.xlabel("Actions")
    plt.ylabel("Frequency")
    plt.title("Frequency of Actions Taken")
    plt.show()

if __name__ == "__main__":
    epsilon = 1.0  # Initial exploration rate
    epsilon_decay = 0.995  # Decay rate for exploration
    epsilon_min = 0.1  # Minimum exploration rate
    no_episodes = 1000  # Number of episodes

    plot_episode_rewards()
    plot_epsilon_decay(epsilon, epsilon_decay, epsilon_min, no_episodes)
    #plot_q_value_heatmaps()
    plot_optimal_policy()
    # Uncomment if you have Q_values_over_time.npy
    plot_q_value_convergence()
    # Uncomment if you have actions_taken.npy
    plot_action_distribution()
