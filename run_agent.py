import importlib.util

from taxi_env import TaxiEnv
from state import *

def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = TaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    
    taxi_row, taxi_col = obs[0], obs[1]

    if render:
        env.render_env((taxi_row, taxi_col), action=None, step=step_count, fuel=env.current_fuel)
    
    while not done:
        action = student_agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col = obs[0], obs[1]
        if render:
            env.render_env((taxi_row, taxi_col), action=action, step=step_count, fuel=env.current_fuel)

    # print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    import random
    from tqdm import tqdm
    agent_scores = []
    for t in tqdm(range(1000)):
        env_config = {
            "grid_size": random.randint(5, 10),
            "fuel_limit": 5000,
            "obstacles_rate": 0.1
        }
        agent_score = run_agent("student_agent.py", env_config)
        agent_scores.append(agent_score)
    # agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {np.mean(agent_scores)}")