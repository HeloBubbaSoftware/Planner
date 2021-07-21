from enum import Enum
from planning_utils import h_3D
from queue import PriorityQueue
import numpy as np
import operator

# Quadroter assume all actions cost the same.
class Action_3D(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 3 values are the delta of the action relative
    to the current grid position. The fourth and final value
    is the cost of performing the action.
    """
    UP = (0, 0,  1, 1)
    DN = (0, 0, -1, 1)
    WEST = (0, -1, 0, 1)
    WEST_UP = (0, -1, 1, np.sqrt(2))
    WEST_DN = (0, -1, -1, np.sqrt(2))
    EAST = (0, 1, 0, 1)
    EAST_UP = (0, 1, 1, np.sqrt(2))
    EAST_DN = (0, 1,-1, np.sqrt(2))
    NORTH = (-1, 0, 0, 1)
    NORTH_UP = (-1, 0, 1, np.sqrt(2))
    NORTH_DN = (-1, 0,-1, np.sqrt(2))
    SOUTH = (1, 0, 0, 1)
    SOUTH_UP = (1, 0, 1, np.sqrt(2))
    SOUTH_DN = (1, 0,-1, np.sqrt(2))
    NORTH_WEST = (-1, -1, 0, np.sqrt(2))
    NORTH_WEST_UP = (-1, -1, 1, np.sqrt(2))
    NORTH_WEST_DN = (-1, -1,-1, np.sqrt(2))
    NORTH_EAST = (-1, 1, 0, np.sqrt(2))
    NORTH_EAST_UP = (-1, 1, 1, np.sqrt(2))
    NORTH_EAST_DN = (-1, 1,-1, np.sqrt(2))
    SOUTH_WEST = (1, -1, 0, np.sqrt(2))
    SOUTH_WEST_UP = (1, -1, 1, np.sqrt(2))
    SOUTH_WEST_DN = (1, -1,-1, np.sqrt(2))
    SOUTH_EAST = (1, 1, 0, np.sqrt(2))
    SOUTH_EAST_UP = (1, 1, 1, np.sqrt(2))
    SOUTH_EAST_DN = (1, 1, -1, np.sqrt(2))

    @property
    def cost(self):
        return self.value[3]

    @property
    def delta(self):
        return (self.value[0], self.value[1], self.value[2])


def valid_actions(vol, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action_3D)
    n, m, l = vol.shape[0]//2, vol.shape[1]//2, vol.shape[2]//2
    x, y, z = current_node
    x_prime, y_prime, z_prime = x + n, y + m, z + l
    #print("x:", x, "y:",y,"z:", z)
    # check if the node is off the grid or
    # it's an obstacle
    if x - 1 < -n or vol[x_prime - 1, y_prime, z_prime] == 1:
        valid_actions.remove(Action_3D.NORTH)
    if x + 1 > n or vol[x_prime + 1, y_prime, z_prime] == 1:
        valid_actions.remove(Action_3D.SOUTH)
    if y - 1 < -m or vol[x_prime, y_prime - 1, z_prime] == 1:
        valid_actions.remove(Action_3D.WEST)
    if y + 1 > m or vol[x_prime, y_prime + 1, z_prime] == 1:
        valid_actions.remove(Action_3D.EAST)
    
    if (x - 1 < -n or y - 1 < -m) or vol[x_prime - 1, y_prime - 1, z_prime] == 1:
        valid_actions.remove(Action_3D.NORTH_WEST)
    if (x - 1 < -n or y + 1 > m) or vol[x_prime - 1, y_prime + 1, z_prime] == 1:
        valid_actions.remove(Action_3D.NORTH_EAST)
    if (x + 1 > n or y - 1 < -m) or vol[x_prime + 1, y_prime - 1, z_prime] == 1:
        valid_actions.remove(Action_3D.SOUTH_WEST)
    if (x + 1 > n or y + 1 > m) or vol[x_prime + 1, y_prime + 1, z_prime] == 1:
        valid_actions.remove(Action_3D.SOUTH_EAST)
    
    if (z - 1 < -l ) or vol[x_prime, y_prime, z_prime-1] == 1:
        valid_actions.remove(Action_3D.DN)
    if (z + 1 > l )  or vol[x_prime, y_prime, z_prime+1]== 1:
        valid_actions.remove(Action_3D.UP)

    if (x - 1 < -n) or (z - 1 < -l) or vol[x_prime - 1, y_prime, z_prime - 1] == 1:
        valid_actions.remove(Action_3D.NORTH_DN) 
    if (x + 1 > n) or (z - 1 < -l) or vol[x_prime + 1, y_prime, z_prime - 1] == 1:
        valid_actions.remove(Action_3D.SOUTH_DN)
    if (y - 1 < -m) or (z - 1 < -l) or vol[x_prime, y_prime - 1, z_prime - 1] == 1:
        valid_actions.remove(Action_3D.WEST_DN)
    if (y + 1 > m) or (z - 1 < -l) or vol[x_prime, y_prime + 1, z_prime - 1] == 1:
        valid_actions.remove(Action_3D.EAST_DN)

    if (x - 1 < -n) or (z + 1 > l) or vol[x_prime - 1, y_prime, z_prime + 1] == 1:
        valid_actions.remove(Action_3D.NORTH_UP) 
    if (x + 1 > n) or (z + 1 > l) or vol[x_prime + 1, y_prime, z_prime + 1] == 1:
        valid_actions.remove(Action_3D.SOUTH_UP)
    if (y - 1 < -m) or (z + 1 > l) or vol[x_prime, y_prime - 1, z_prime + 1] == 1:
        valid_actions.remove(Action_3D.WEST_UP)
    if (y + 1 > m) or (z + 1 > l) or vol[x_prime, y_prime + 1, z_prime + 1] == 1:
        valid_actions.remove(Action_3D.EAST_UP)  

    if (x - 1 < -n or y - 1 < -m or z - 1 < -l) or vol[x_prime - 1, y_prime - 1, z_prime - 1] == 1:
        valid_actions.remove(Action_3D.NORTH_WEST_DN)
    if (x - 1 < -n or y + 1 > m or z - 1 < -l) or vol[x_prime - 1, y_prime + 1, z_prime - 1] == 1:
        valid_actions.remove(Action_3D.NORTH_EAST_DN)
    if (x + 1 > n or y - 1 < -m or z - 1 < -l) or vol[x_prime + 1, y_prime - 1, z_prime - 1] == 1:
        valid_actions.remove(Action_3D.SOUTH_WEST_DN)
    if (x + 1 > n or y + 1 > m or z - 1 < -l) or vol[x_prime + 1, y_prime + 1, z_prime - 1] == 1:
        valid_actions.remove(Action_3D.SOUTH_EAST_DN)
  
    if (x - 1 < -n or y - 1 < -m or z + 1 > l) or vol[x_prime - 1, y_prime - 1, z_prime + 1] == 1:
        valid_actions.remove(Action_3D.NORTH_WEST_UP)
    if (x - 1 < -n or y + 1 > m or z + 1 > l) or vol[x_prime - 1, y_prime + 1, z_prime + 1] == 1:
        valid_actions.remove(Action_3D.NORTH_EAST_UP)
    if (x + 1 > n or y - 1 < -m or z + 1 > l) or vol[x_prime + 1, y_prime - 1, z_prime + 1] == 1:
        valid_actions.remove(Action_3D.SOUTH_WEST_UP)
    if (x + 1 > n or y + 1 > m or z + 1 > l) or vol[x_prime + 1, y_prime + 1, z_prime + 1] == 1:
        valid_actions.remove(Action_3D.SOUTH_EAST_UP)

    #print(valid_actions)

    return valid_actions


def a_star_3D(vol, h_3D, start, goal):
    goal = tuple(map(operator.sub, goal, start))
    goal = int(goal[0]), int(goal[1]), int(goal[2])
    start = (0, 0, 0)
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)
    branch = {}
    found = False
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(vol, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1], current_node[2] + da[2])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h_3D(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
        print("Local Path Found")
    else:
        print('**********************')
        print('Failed to find a LOCAL path!')
        print('**********************')
    return path[::-1], path_cost