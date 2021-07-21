from enum import Enum
from queue import PriorityQueue
import numpy as np
from numpy.lib.histograms import _ravel_and_check_weights
from skimage import draw
import numpy.linalg as LA
from shapely.geometry import Polygon, Point
from bresenham import bresenham
import math
from skimage.data import cell
from udacidrone.frame_utils import local_to_global
import scipy.ndimage.filters as ndif

def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        #if alt + d_alt > drone_altitude:
        obstacle = [
            int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
            int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
            int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
            int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
        ]
        #the "dimension" variable is to add height data to grid vs. simple 1/0
        grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = alt+d_alt

    return grid, int(north_min), int(east_min)

def create_height_map(data):
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    height_map = np.zeros((north_size, east_size))
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        obstacle = [
            int(np.clip(north - d_north -  north_min, 0, north_size-1)),
            int(np.clip(north + d_north -  north_min, 0, north_size-1)),
            int(np.clip(east - d_east -  east_min, 0, east_size-1)),
            int(np.clip(east + d_east -  east_min, 0, east_size-1)),
        ]
        #the "dimension" variable is to add height data to grid vs. simple 1/0
        height_map[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = alt+d_alt
        #print("Height:", height_map[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1])
    return height_map


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relativeR
    to the current lateral grid position. Third, Z Axis move and the fourth  and final value
    is the cost of performing the action.
    """
    #for no altitude change
    WEST = (0, -1, 0, 1)
    EAST = (0, 1, 0, 1)
    NORTH = (-1, 0, 0, 1)
    SOUTH = (1, 0, 0, 1)
    NORTH_WEST = (-1, -1, 0, np.sqrt(2))
    NORTH_EAST = (-1, 1, 0, np.sqrt(2))
    SOUTH_WEST = (1, -1, 0, np.sqrt(2))
    SOUTH_EAST = (1, 1, 0, np.sqrt(2))

    @property
    def cost(self):
        return self.value[3]

    @property
    def delta(self):
        return (self.value[0], self.value[1], self.value[2])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m    = grid.shape[0] - 1, grid.shape[1] - 1
    x, y    = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 0:
        valid_actions.remove(Action.NORTH)

    if x + 1 > n or grid[x + 1, y] == 0:
        valid_actions.remove(Action.SOUTH)

    if y - 1 < 0 or grid[x, y - 1] == 0:
        valid_actions.remove(Action.WEST)

    if y + 1 > m or grid[x, y + 1] == 0:
        valid_actions.remove(Action.EAST)

    if (x - 1 < 0 or y - 1 < 0) or grid[x - 1, y - 1] == 0:
        valid_actions.remove(Action.NORTH_WEST)
 
    if (x - 1 < 0 or y + 1 > m) or grid[x - 1, y + 1] == 0:
        valid_actions.remove(Action.NORTH_EAST)

    if (x + 1 > n or y - 1 < 0) or grid[x + 1, y - 1] == 0:
        valid_actions.remove(Action.SOUTH_WEST)

    if (x + 1 > n or y + 1 > m) or grid[x + 1, y + 1] == 0:
        valid_actions.remove(Action.SOUTH_EAST)

    return valid_actions


def a_star(path, grid, h, start, goal):
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():  
        item = queue.get()
        current_node = item[1]
       # print("Queue:", item, "\n")
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                #print("Action:", action)
                
                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path = np.resize(path,(1000,4))
        path[0] = np.append(np.array(goal).astype(int),np.array([0,0.0]))
        i=1
        # while branch[n][1] != start:
        #     np.append(path,(branch[n][1]), axis=0)
        #     n = branch[n][1]
        # np.append(path,(branch[n][1]))
        while branch[n][1] != start:
            A = np.array(branch[n][1])
            B = np.array([0,0.0])
            #print("path len:",len(path)," i:", i)
            path[i] = np.append(A, B, axis=0)
            i +=  1
            n = branch[n][1]
        path[i] = np.append(np.array(branch[n][1]),np.array([0,0.0]), axis=0)
        path = np.resize(path,(i+1,4))
        path[:,0:3] = path[:,0:3].astype(int)
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return np.array(path[::-1]), path_cost

def add_heading_to_path(path, north, east, home):
    # This routine takes the reduced path and adds the headings for each edge to the list
    pointA = []
    pointB = []
    for i in range(len(path)-2):
        pointA.append(path[i][0]+north)
        pointA.append(path[i][1]+east)
        pointA.append(path[i][2])
        pA = local_to_global(tuple(pointA), home)
        #reverese the long/lat to lat/long for the compass bearing routine
        pA_=tuple([pA[1],pA[0],pA[2]])
        #repeat the process for pointB
        pointB.append(path[i+1][0]+north)
        pointB.append(path[i+1][1]+east)
        pointB.append(path[i+1][2])
        pB = local_to_global(tuple(pointB), home)
        pB_ = tuple([pB[1],pB[0],pB[2]])
        #heading = calculate_initial_compass_bearing(tuple(pA_), tuple(pB_))
        heading = calculate_initial_compass_bearing(pA_,pB_)
        #print("pointA:", pA, "pointB:", pB, "heading:", heading)
        #clear the lists for next itereation if needed
        pointA = []
        pointB = []
       # assign the current edge bearing (A to B) to B's postion in the list.
       # this is necessary due to self.cmd_position structure
        path[i+1][3] = np.deg2rad(heading)
    #once complete with the path, assign the last and first points some value 
    #path[-1][3] = np.deg2rad(heading)
    path[0][3] =  2*np.pi
    return path


def add_altitude_to_path(path,alt):
    for i in range(len(path)):
        path[i][2] = alt
    return path

def check_path(path,grid,cube_size):
    # This routine will check a given path segment for collisions based on a cube boundary
    # centered around the discrete point along the segment, as found by breshenham
    did_collide = False
    i = 0
    while i < (len(path)-1):
        #Bresenhan module imported worked fine for 2D, on 3D, went with skimage.line_nd
        cells = list(bresenham(path[i][0],path[i][1],path[i+1][0],path[i+1][1]))
        #cells = list(draw.line_nd((path[i][0:3]),(path[i+1][0:3])))
        #print("path:",path[i][0],path[i][1],path[i+1][0],path[i+1][1])
        for cell in cells:
            if not collision_check(cell,grid,cube_size, cube_size):
                did_collide = True
                #print("Collision at LOCAL ADDRESS:",cell, ", point removal skipped")
                break 
            else:
                did_collide = False
        if did_collide:
            break
        else:
            i += 1  
    return did_collide

def collision_check(cell, grid, north, east):
    # the collision check uing cube boundary on the specicic point
    cube= np.array(grid[cell[0]-north:cell[0]+north,cell[1]-east:cell[1]+east])
    return np.all((cube==0)) 

def over_obstacle_clearance(path, grid, clearance, goal):
    # This routine will iterate through the path to determine of removal of B is allowed from
    # a segment ABC.  Segment AC is tested for collision, if none found it will keep the removal 
    # of B, otherwise it will replace B.  It will then move to the next point.  Once complete it will
    # repeat the process until no changes are made to the path.  The number of repetitions through the path 
    # has shown to be a function of halving the original path length.
    for i in range(len(path)-2):
        line = list(bresenham(path[i][0],path[i][1],path[i+1][0],path[i+1][1]))
        alt = [path[i][2]]*len(line)
        line_3D = list(zip(line,[path[i][2]]*len(line)))
        for j in range(len(line_3D)):
            point = line_3D[j][0]
            z = line_3D[j][1]
            #print("Drone Clearance:", path[i][2]- grid[point[0],point[1]], "at location:", point[0], point[1])
    print("Complete with Alt Clear")
    return path

def find_start_goal(skel, start, goal):
    #this routine finds the on-ramp and the off ramp on the graph to the start and goal points respectively
    skel_cells = np.transpose(skel.nonzero())
    start_min_dist = np.linalg.norm(np.array(start).astype(int) - np.array(skel_cells), axis=1).argmin()
    near_start = skel_cells[start_min_dist]
    goal_min_dist = np.linalg.norm(np.array(goal).astype(int) - np.array(skel_cells), axis=1).argmin()
    near_goal = skel_cells[goal_min_dist]
    return np.array(near_start), np.array(near_goal)

def h(position, goal_position):
    x1 = position[0]
    x2 = goal_position[0]
    y1 = position[1]
    y2 = goal_position[1]
    h = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return h

def h_3D(position, goal_position):
    # for 3D points
    x1 = position[0]
    x2 = goal_position[0]
    y1 = position[1]
    y2 = goal_position[1]
    z1 = position[2]
    z2 = goal_position[2]
    h_3D = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return h_3D

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Credit: https://gist.github.com/jeromer

    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """

    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def get_sector(heading, width):
    first = []
    second = []
    if not math.isnan(heading):
        compass = np.linspace(0,359,360).astype(int)
        left  = int(heading - width)
        right = int(heading + width)
        l= left
        r = right
        if left < 0:
            l = (360 - (width-heading))
        elif right > 359:
            r= width - (360 - heading)
        if left < 0:
            first = [x for x in compass if l <= x <= 360]
            second = [x for x in compass if 0 <= x <= r]
        elif right > 359:
            first = [x for x in compass if l <= x <= 360]
            second = [x for x in compass if 0 <= x <= r]
        else:
            first = []
            second = [x for x in compass if l <= x <= r]
    return np.concatenate((first, second), axis=0).astype(int)

def findMiddle(input_list):
    middle = int(len(input_list)//2)
    return input_list[middle]

def pop(my_array,pr):
    """ row popping in numpy arrays
    Input: my_array - NumPy array, pr: row index to pop out
    Output: [new_array,popped_row] """
    i = pr
    pop = my_array[i]
    new_array = np.vstack((my_array[:i],my_array[i+1:]))
    return [new_array,pop]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

class StreamingMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)