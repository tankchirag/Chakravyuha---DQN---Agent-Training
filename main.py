# Imports:
import torch
import gymnasium as gym
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from env_cht8982 import ChakravyuhEnv

# User definitions:
train_dqn = True  # Set to True to train
test_dqn = False  # Set to True to test
render = False

# Define env attributes (environment specific)
env = ChakravyuhEnv()
no_actions = 4 #env.action_space.n
no_states = 2 #env.grid_size[0] * env.grid_size[1]

# Hyperparameters:
learning_rate = 0.005
gamma = 0.98
buffer_limit = 50_000
batch_size = 32
num_episodes = 10_000
max_steps = 100  # Use the max steps defined in your env
initial_epsilon = 0.08
final_epsilon = 0.01
decay_rate = 0.995
exponential_decay = False # If true Epsilon decay using exponential function ..else linear function

# Also tune--- random_position=True flag ffor reset....
# Also tune--- self.use_motivation-True flag....tune self.motivate_rate = 0.2 and self.motivate_decay = 0.2


# Main:
if train_dqn:

    # Initialize the Q Net and the Q Target Net
    q_net = Qnet(no_actions=no_actions, no_states=no_states)
    q_target = Qnet(no_actions=no_actions, no_states=no_states)
    q_target.load_state_dict(q_net.state_dict())

    # Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 5
    episode_reward = 0.0
    print(f'episode_reward: {episode_reward}')
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    rewards = []

    for n_epi in range(num_episodes):
        if exponential_decay == False:
            print('Epsilon decay using linear function')
            epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        else:
            print('Epsilon decay using exponential function')
            epsilon = final_epsilon + (initial_epsilon - final_epsilon) * (decay_rate ** n_epi)


        s = env.reset()
        #s = s.flatten()  # Flatten the state
        done = False

        for _ in range(max_steps):
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, _ = env.step(a)
            #s_prime = s_prime.flatten()  # Flatten the state

            done_mask = 0.0 if done else 1.0

            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            print(f"Step reward: {r}")  # Print reward for each step
            episode_reward += r #env.cumulative_reward
            print(f'episode reward : {episode_reward}')

            if done:
                break


        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(
                f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}")

        rewards.append(episode_reward)
        episode_reward = 0.0
        print(f'episode reward : {episode_reward}')

        if rewards[-10:] == [max_steps]*10:
            break

    env.close()

    torch.save(q_net.state_dict(), "dqn.pth")

    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

if test_dqn:
    print("Testing the trained DQN: ")
    dqn = Qnet(no_actions=no_actions, no_states=no_states)
    dqn.load_state_dict(torch.load("dqn.pth"))

    for _ in range(10):
        s = env.reset()
        #s = s.flatten()  # Flatten the state
        episode_reward = env.cumulative_reward #

        for _ in range(max_steps):
            action = dqn(torch.from_numpy(s).float())
            s_prime, reward, done, _ = env.step(action.argmax().item())
            #s_prime = s_prime.flatten()  # Flatten the state

            s = s_prime
            episode_reward += reward #= env.cumulative_reward #

            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()
