import numpy as np

BOARD_SIZE = 5

# Direction for each station: (-1, 0, 1) for each axis, obstacle test for adjacent cells (moved cell will be seen as obstacle),targeting station, passenger and destination test and carrying
STATE_SIZE = tuple([3] * 2 + [2] * 4 + [4] + [2] * 3)
ACTION_SIZE = 6
visited_actions = np.zeros((BOARD_SIZE, BOARD_SIZE, 4), dtype=bool)

_state = None
def reset_state():
    global _state, visited_actions
    _state = None
    visited_actions.fill(False)

def update_state(obs, action):
    global _state, visited_actions

    if action == 4: # PICKUP
        _state[8] = 1
    elif action == 5: # DROPOFF
        _state[8] = 0
    else: # MOVE
        visited_actions[obs[0], obs[1], action] = True

def get_agent_state(obs):
    global _state, visited_actions

    taxi_y, taxi_x = obs[0], obs[1]
    station_y = [obs[2], obs[4], obs[6], obs[8]]
    station_x = [obs[3], obs[5], obs[7], obs[9]]
    is_obstacle = [obs[11], obs[10], obs[12], obs[13]] # obs in north(1), south(0), east(2), west(3) order
    passenger_look = obs[14]
    destination_look = obs[15]
    
    carrying_passenger = _state[9] if _state is not None else 0
    target_station = _state[6] if _state is not None else 0
    passenger_test = int(passenger_look and not carrying_passenger and (taxi_y, taxi_x) in zip(station_y, station_x))
    destination_test = int(destination_look and carrying_passenger and (taxi_y, taxi_x) in zip(station_y, station_x))
    
    if station_y[target_station] == taxi_y and station_x[target_station] == taxi_x:
        visited_actions.fill(False)
        target_station = (target_station + 1) % 4
    
    next_state = [np.sign(station_y[target_station] - taxi_y), np.sign(station_x[target_station] - taxi_x)] + list(np.bitwise_or(is_obstacle, visited_actions[taxi_y, taxi_x], dtype=int)) + [target_station, passenger_test, destination_test, carrying_passenger]
    
    _state = next_state
    return tuple(_state)
