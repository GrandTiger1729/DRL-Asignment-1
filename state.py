# Direction for each station: (-1, 0, 1) for each axis, whether the square adjacent to the taxi is an obstacle, whether it is carrying a passenger, possible state for station (unknown, empty, passenger, destination)
# _state_size = tuple([3] * 8 + [2] * 4 + [2] + [4] * 4)
_state_size = tuple([2] * 4)
def get_state_size():
    return _state_size

_action_size = 6
def get_action_size():
    return _action_size

_state = None
def reset_state():
    global _state
    _state = None

def get_agent_state(obs):
    global _state
    _state = tuple(obs[10:14])
    return _state
    # taxi_y, taxi_x = obs[0], obs[1]
    # station_y = [obs[2], obs[4], obs[6], obs[8]]
    # station_x = [obs[3], obs[5], obs[7], obs[9]]
    # is_obstacle = [obs[10], obs[11], obs[12], obs[13]]
    # passenger_look = obs[14]
    # destination_look = obs[15]
    
    # carrying_passenger = _state[12] if _state is not None else 0
    # station_observation = list(_state[13:]) if _state is not None else [0] * 4
    
    # if passenger_look or destination_look:
    #     for i in range(4):
    #         # if adjecent to the taxi
    #         if station_observation[i] == 0 and abs(station_y[i] - taxi_y) + abs(station_x[i] - taxi_x) <= 1:
    #             if not carrying_passenger and passenger_look:
    #                 station_observation[i] = 2
    #             elif destination_look:
    #                 station_observation[i] = 3
    #             else:
    #                 station_observation[i] = 1
    
    # next_state = [np.sign(y - taxi_y) for y in station_y] + [np.sign(x - taxi_x) for x in station_x] + [carrying_passenger] + is_obstacle + station_observation
    # _state = tuple(next_state)
    # return _state
