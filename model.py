import numpy as np

class Model:
    def __init__(self, state_size, action_size, lr=0.1):
        self.state_size = state_size
        self.action_size = action_size

        self.q_table = np.zeros((*state_size, action_size), dtype=np.float32)
        self.alpha = lr

    def update(self, state, action, target):
        self.q_table[*state, action] += self.alpha * (target - self.q_table[*state, action])

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)  # Explore.
        else:
            return np.argmax(self.q_table[*state]).item()  # Exploit.
