from IPython.display import clear_output
import random
import numpy as np

class TaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=5000, obstacles_rate=0.3):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        
        self.stations = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None
        self.obstacles = set()  # No obstacles in simple version
        self.destination = None
        self.obstacles_rate = obstacles_rate
        
    def _get_neighbours(self, pos):
        """Return the neighbours of a position."""
        row, col = pos
        neighbours = []
        if row > 0:
            neighbours.append((row - 1, col))
        if row < self.grid_size - 1:
            neighbours.append((row + 1, col))
        if col > 0:
            neighbours.append((row, col - 1))
        if col < self.grid_size - 1:
            neighbours.append((row, col + 1))
        return neighbours

    def _has_path(self, start, end):
        """Find the shortest path between two points using BFS."""
        visited = set()
        queue = [start]
        while queue:
            current = queue.pop(0)
            if current == end:
                return True
            visited.add(current)
            for neighbour in self._get_neighbours(current):
                if neighbour not in visited and neighbour not in self.obstacles:
                    queue.append(neighbour)
        return False

    def reset(self):
        """Reset the environment, ensuring Taxi, passenger, and destination are not overlapping obstacles"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        
        
        success = False
        while not success:
            success = True
            
            self.obstacles = set(random.sample([(x, y) for x in range(self.grid_size) for y in range(self.grid_size)], k=int(self.grid_size * self.grid_size * self.obstacles_rate)))

            self.stations = random.sample([(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in self.obstacles], k=4)
            
            for i in range(4):
                for j in range(i + 1, 4):
                    if abs(self.stations[i][0] - self.stations[j][0]) + abs(self.stations[i][1] - self.stations[j][1]) == 1:
                        success = False
                        break
                if not success:
                    break
            if not success:
                continue
            
            available_positions = [
                (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                if (x, y) not in self.stations and (x, y) not in self.obstacles
            ]

            self.taxi_pos = random.choice(available_positions)
            self.passenger_loc = random.choice([pos for pos in self.stations])
            possible_destinations = [s for s in self.stations if s != self.passenger_loc]
            self.destination = random.choice(possible_destinations)
            
            if not (self._has_path(self.taxi_pos, self.passenger_loc) and self._has_path(self.passenger_loc, self.destination)):
                success = False
            
        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0:  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        
        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos  
                else:
                    reward = -10  
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward - 0.1, True, {}
                    else:
                        reward -= 10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -= 10
                    
        reward -= 0.1  
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward - 10, True, {}

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        destination_loc_north = int((taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int((taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int((taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int((taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int((taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        state = (taxi_row, taxi_col, self.stations[0][0], self.stations[0][1], self.stations[1][0], self.stations[1][1], self.stations[2][0], self.stations[2][1], self.stations[3][0], self.stations[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state

    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        clear_output(wait=True)
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        

        for pos, color in zip(self.stations, ['R', 'G', 'Y', 'B']):
            grid[pos[0]][pos[1]] = color
            
        for pos in self.obstacles:
            grid[pos[0]][pos[1]] = 'X'
        
        if not self.passenger_picked_up:
            py, px = self.passenger_loc
            grid[py][px] = 'P'
        
        if self.destination is not None:
            dy, dx = self.destination
            grid[dy][dx] = 'D'
        
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"

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
