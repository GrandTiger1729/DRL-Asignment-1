import numpy as np

BOARD_SIZE = 5

# Direction for each station: (-1, 0, 1) for each axis, obstacle test for adjacent cells (moved cell will be seen as obstacle), passenger and destination test and carrying
STATE_SIZE = tuple([3] * 2 + [2] * 4 + [2] * 3)
ACTION_SIZE = 6
visited_actions = np.zeros((BOARD_SIZE, BOARD_SIZE, 4), dtype=bool)
target_id = 0
target_y, target_x = None, None
dropped_off = False
carrying = False

_state = None
def reset_state():
    global _state, visited_actions, target_y, target_x, target_id, dropped_off, carrying
    _state = None
    visited_actions.fill(False)
    target_id = 0
    target_y, target_x = None, None
    dropped_off = False
    carrying = False

def resolve_state(obs, action):
    global _state, visited_actions, target_y, target_x, target_id, dropped_off, carrying

    taxi_y, taxi_x = obs[0], obs[1]

    if action == 4: # PICKUP
        if target_y == taxi_y and target_x == taxi_x and not carrying:
            visited_actions.fill(False)
            if not dropped_off:
                target_id = (target_id + 1) % 4
            carrying = True
            dropped_off = False
    elif action == 5: # DROPOFF
        if carrying:
            dropped_off = True
            carrying = False
            target_y, target_x = taxi_y, taxi_x
    else: # MOVE
        visited_actions[obs[0], obs[1], action] = True

def get_agent_state(obs):
    global _state, visited_actions, target_y, target_x, target_id, dropped_off, carrying

    taxi_y, taxi_x = obs[0], obs[1]
    station_y = [obs[2], obs[4], obs[6], obs[8]]
    station_x = [obs[3], obs[5], obs[7], obs[9]]
    is_obstacle = [obs[11], obs[10], obs[12], obs[13]] # obs in north(1), south(0), east(2), west(3) order
    passenger_look = obs[14]
    destination_look = obs[15]

    if not dropped_off:
        target_y, target_x = station_y[target_id], station_x[target_id]
    
    passenger_test = int(passenger_look and not carrying and (taxi_y, taxi_x) == (target_y, target_x))
    destination_test = int(destination_look and carrying and (taxi_y, taxi_x) == (target_y, target_x))
    
    next_state = [np.sign(target_y - taxi_y), np.sign(target_x - taxi_x)] + list(np.bitwise_or(is_obstacle, visited_actions[taxi_y, taxi_x], dtype=int)) + [passenger_test, destination_test, int(carrying)]
    
    _state = next_state
    return tuple(_state)
