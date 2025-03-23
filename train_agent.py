import numpy as np

from tqdm import tqdm
import pickle

from model import Model
from state import *
from taxi_env import TaxiEnv
        
def train(episodes=5000, alpha=0.05, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.9995):
    # The default parameters should allow learning, but you can still adjust them to achieve better training performance.
    """
    ✅ Train a Q-learning agent using **PyTorch tensors**.
    """
    env = TaxiEnv(grid_size=BOARD_SIZE)
    
    model = Model(STATE_SIZE, ACTION_SIZE, lr=alpha)
    epsilon = epsilon_start
    shaped_rewards_per_episode = []
    actual_returns_per_episode = []
    success_per_episode = []

    for episode in tqdm(range(episodes)):
        obs, _ = env.reset()
        reset_state()

        # Retrieve the agent's state directly from the environment.
        state = get_agent_state(obs)
        step_count = 0

        done = False
        shaped_return = 0
        actual_return = 0
        
        while not done:
            action = model.get_action(state, epsilon)
            resolve_state(obs, action)
            obs, reward, done, _ = env.step(action)
            next_state = get_agent_state(obs)
            
            actual_return += reward
            reward = reward_shaping(state, action, reward)
            shaped_return += reward
            
            target = reward + gamma * np.max(model.q_table[next_state])
            model.update(state, action, target)
            
            state = next_state
            step_count += 1
            
        # Decay epsilon over time to reduce exploration.
        epsilon = max(epsilon_end, epsilon * decay_rate)
        shaped_rewards_per_episode.append(shaped_return)
        actual_returns_per_episode.append(actual_return)
        success_per_episode.append(1 if env.current_fuel > 0 else 0)

        # Print progress every 100 episodes.
        if (episode + 1) % 100 == 0:
            shaped_mean = np.mean(shaped_rewards_per_episode[-100:])
            actual_mean = np.mean(actual_returns_per_episode[-100:])
            success_rate = np.mean(success_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Total Return: {shaped_mean:.8f}, Actual Return: {actual_mean:.8f}, Epsilon: {epsilon:.3f}, Success Rate: {success_rate}")

    return model

model = train(episodes=3000)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)