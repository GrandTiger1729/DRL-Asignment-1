import numpy as np

BOARD_SIZE = 10

# Direction for each station: (-1, 0, 1) for each axis, currently targeting station, obstacle test for adjacent cells (moved cell will be seen as obstacle), passenger and destination test and carrying
STATE_SIZE = tuple([3] * 2 + [4] + [2] * 4 + [2] * 3)
ACTION_SIZE = 6
visited_actions = np.zeros((BOARD_SIZE, BOARD_SIZE, ACTION_SIZE), dtype=bool)


_state = None
def reset_state():
    global _state
    _state = None

def update_state(obs, action):
    global _state, visited_actions

    visited_actions[obs[0], obs[1], action] = True
    if action == 4: # PICKUP
        _state[9] = 1
    elif action == 5: # DROPOFF
        _state[9] = 0

def get_agent_state(obs):
    global _state

    taxi_y, taxi_x = obs[0], obs[1]
    station_y = [obs[2], obs[4], obs[6], obs[8]]
    station_x = [obs[3], obs[5], obs[7], obs[9]]
    is_obstacle = [obs[11], obs[10], obs[12], obs[13]] # obs in north(1), south(0), east(2), west(3) order
    passenger_look = obs[14]
    destination_look = obs[15]
    
    carrying_passenger = _state[9] if _state is not None else 0
    target_station = _state[4] if _state is not None else 0
    passenger_test = int(passenger_look and not carrying_passenger and (taxi_y, taxi_x) in zip(station_y, station_x))
    destination_test = int(destination_look and carrying_passenger and (taxi_y, taxi_x) in zip(station_y, station_x))
    
    if station_y[target_station] == taxi_y and station_x[target_station] == taxi_x:
        visited_actions.fill(False)
        target_station = (target_station + 1) % 4
    
    next_state = [np.sign(station_y[target_station] - taxi_y), np.sign(station_x[target_station] - taxi_x)] + [target_station] + list(np.bitwise_or(is_obstacle, visited_actions[taxi_y, taxi_x, :4], dtype=int)) + [passenger_test, destination_test, carrying_passenger]
    
    _state = next_state
    return tuple(_state)
