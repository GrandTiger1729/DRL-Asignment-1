# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle

from taxi_env import *
from state import *
from model import Model

with open("model.pkl", "rb") as f:
    model: Model = pickle.load(f)
last_state = None
last_action = None

def get_action(obs):
    global last_state, last_action
    state = get_agent_state(obs)

    if last_state is not None:
        target = reward_shaping(last_state, last_action) + 0.99 * np.max(model.q_table[state])
        model.update(last_state, last_action, target)

    action = model.get_action(state, 0)
    if action == 5 and (obs[15] and carrying and (obs[0], obs[1]) == (target_y, target_x)):
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
    resolve_state(obs, action)

    last_state = state
    last_action = action
    
    return action

