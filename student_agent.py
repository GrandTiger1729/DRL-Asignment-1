# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle

from taxi_env import *
from model import Model

with open("model.pkl", "rb") as f:
    model: Model = pickle.load(f)

def get_action(obs):
    state = get_agent_state(obs)
    action = model.get_action(state, 0)
    return action

