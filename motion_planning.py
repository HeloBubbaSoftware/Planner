import argparse
import time
import msgpack
import csv
import re
from enum import Enum, auto
import matplotlib.pyplot as plt

import numpy as np
import visdom
from skimage.morphology import medial_axis
from skimage.util import invert
from planning_utils import a_star, check_path, add_heading_to_path, create_grid, create_height_map, find_start_goal, h,h_3D, add_altitude_to_path, over_obstacle_clearance
from voxmap import create_voxmap
from My_Data import MyDrone
#from planning_3D import a_star_3D
from planning_2_5D import a_star_2_5D
from planning_3D import a_star_3D
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local
from sympy import Point, Polygon, Segment
from shield import shield

from time import sleep

plt.rcParams['figure.figsize'] = 12, 12

class States(Enum)  :
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()

class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}
        self.global_voxmap = []
        self.global_path = []
        self.path_3D = []
        self.grid = []
        self.local_path = []
        self.north_offset = 0.0
        self.east_offset = 0.0
        self.heartbeat = 0

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        # print("In mission? - In State Callback,", self.in_mission, self.flight_state )
        # print("Target Posion:", self.target_position)
        # print("Local Position:", self.local_position)
        # print("Length of waypoint array:", len(self.waypoints))
        # print("Distance to Next Goal:",np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]))
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.90 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if len(self.waypoints) > 0: 
                self.heartbeat += 1
                if self.heartbeat%10 == 0:
                    self.local_planner()
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 5.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self): 

        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()
        
    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if len(self.waypoints) == 0:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                if len(self.waypoints) > 0:
                     self.arm()
                     self.take_control()
                     self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        if len(self.waypoints) > 0:
            self.arm()
            self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        sink = self.global_path.pop(0)
        self.target_position = self.waypoints.pop(0)
        #print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        pack = np.array(self.waypoints)[:,0:3].astype(int).tolist()
        data = msgpack.dumps(pack)
        #clear = msgpack.dumps([0,0,0,0])
        print("after dumps")
        #self.connection._master.write(clear)
        #sleep(0.01)
        self.connection._master.write(data)
        print("after write")

    def local_planner(self):
        shield_grid = shield(self.local_position,self.north_offset, self.east_offset, self.grid,10,6)
        self.local_path, goal = self.create_local_path_2_5D(20, 1)
        #self.local_path = self.simplify_path("local", 10, goal)
        #print("Start of local path:" ,self.local_path[0])

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 35
        SAFETY_DISTANCE = 0
        TARGET_CLEARANCE = 3

        self.target_position[2] = TARGET_ALTITUDE

        # Open colliders to get the predefiined start lat/long
        # f = open('colliders.csv', newline='')
        # reader = csv.reader(f)
        # first_line = next(reader)
        # lat0 = float(re.findall(r"[-+]?\dquit*\.\d+|\d+",first_line[0])[1])
        # lon0 = float(re.findall(r"[-+]?\d*\.\d+|\d+",first_line[1])[1])

        # Set home postion.  This is needed for multiple reasons
        # to include the calculation of the heading between points
        self.set_home_as_current_position()
        # Set global postion
        #global_position = self.global_position
        #self.set_home_position(global_position[0], global_position[1], global_position[2])
        
        #print("global home:", self.global_home)

        # This gets us to the local graph postion locations
        self.set_local_position = global_to_local((self.global_position[0], self.global_position[1], self.global_position[2]),self.global_home)
        #print("self.local_position:", self.local_position)
        #print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
        #                                                                 self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles, made for creating the graph.  
        print("Creating Grid")
        self.grid, self.north_offset, self.east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("Finishing Grid")
        
        # #populate global_voxmap
        # print("Creating VOXMAP")
        # #self.global_voxmap = create_voxmap(data, 1)
        # print("Finished VOXMAP")

        # Grid start and goal setting (using local coordinates - trivial to convert to lat/long)
        grid_start = (int(self.local_position[0])-self.north_offset, int(self.local_position[1])-self.east_offset)
        #print("Grid Start:", grid_start)
        
        grid_goal = (600,10)

        # Medial Axis selected because is it reasonably predicable (vs. random point) and provides inherent obsticle avoidance
        # also - it is fast!
        print("Starting Medial Axis")
        skeleton = medial_axis(invert(np.array(self.grid).astype(bool)))
        print("Finished Medial Axis")
        # find the entry/exit points to the graph

        print("Sarting Start Goal)")
        start_ne, goal_ne = find_start_goal(skeleton, grid_start, grid_goal)
        print("finishing Start Goal")
        
        # A* - fast, reliable and complete.
        print("Starting AStar")
        self.global_path, cost = a_star(invert(skeleton).astype(int), h, tuple(start_ne), tuple(goal_ne))
        print("Finished AStar")
        # Add dimensions of altitude and heading placeholder to the path
        self.global_path = add_altitude_to_path(self.global_path, TARGET_ALTITUDE)
        
        # save for display
        original_path = self.global_path.copy()
        print("Original Path Length:", len(self.global_path))
        # Define the edge of the cube buffer to determine collisions.  I attempted multiple sizes, 2 was the 
        # speed/safety trade-off.  Collisons checks on the path planning are computationally intensive.
        # This also played into the seletion of the SAFETY_DISTANCE estimation.
        cube_size = 2
        self.global_path = self.simplify_path("global", cube_size, grid_goal)
        print("After Simplification Path Length:", len(self.global_path))
        #Now using 2.5 degree grid, adjust the path for over-obsticle clearence.
        #height_map = create_height_map(data)
        #path = over_obstacle_clearance(path, height_map, TARGET_CLEARANCE, grid_goal)
        # Adds the heading values to each edge of the path.
        self.global_path = add_heading_to_path(self.global_path, self.north_offset, self.east_offset, self.global_home)
        # Convert path to waypoints
        
        self.waypoints = [[int(p[0] + self.north_offset), int(p[1] + self.east_offset), int(p[2]), p[3]] for p in self.global_path]
        # Used to display the finished path in the simulator
        self.send_waypoints()

        # plt.imshow(self.grid, cmap='Greys', origin='lower')
        # plt.imshow(skeleton, cmap='Greys', origin='lower', alpha=0.7)
        # plt.plot(start_ne[1], start_ne[0], 'bx')
        # plt.plot(goal_ne[1], goal_ne[0], 'bx')
        # pp2 = np.array(original_path)
        # plt.plot(pp2[:, 1], pp2[:, 0], 'r')
        # pp = np.array(self.global_path)
        # plt.plot(pp[:, 1], pp[:, 0], 'g')
        # pp3 = np.array(self.waypoints)
        # plt.plot(pp3[:, 1], pp[:, 0], 'b')
        # plt.xlabel('EAST')
        # plt.ylabel('NORTH')
        # plt.show()

        
    def create_local_path_2_5D(self, size=40, resolution = 1):
        pt = []
        pt.append(self.local_position[1] + self.north_offset)
        pt.append(self.local_position[0] + self.east_offset)
        pt.append(abs(self.local_position[2]))
        n_low = int(pt[0])-int(size)-1
        n_hi  = int(pt[0])+int(size)
        e_low = int(pt[1])-int(size)-1
        e_hi  = int(pt[1])+int(size)
        #print("Current Point:",pt)
        if n_low < 0: n_low = 0
        if n_hi >np.array(self.grid).shape[0]-1: n_hi = np.array(self.grid).shape[0]-1
        if e_low < 0: e_low = 0
        if e_hi > np.array(self.grid).shape[1]-1: e_hi = np.array(self.grid).shape[1]-1
        local_vox = self.grid[n_low:n_hi, e_low:e_hi]
        if np.all((local_vox != 0)):
            print("OBSTICALE DETECTED IN LOCAL VOX MAP")
            #print(n_low, n_hi, e_low, e_hi)
            horizon = Polygon((n_hi-1, e_hi-1), (n_low+1, e_hi-1), (n_low+1, e_low+1), (n_hi-1, e_low+1))
            #search out on path for a point outside of the local voxmap
            for i in range(len(self.waypoints)):
                pt2 = Point(self.waypoints[i])
                if horizon.contains(pt2[0:2]):
                    continue
                else:
                    break
            #then assign it as a SEGMENT (starts from current postion looking outward
            path_segment = Segment(Point(pt[0:2]), pt2)
            #find the point that it penetrates the local voxmap and use it as the new goal
            local_goal = horizon.intersection(path_segment)
            local_goal = (int(local_goal[0][0]),int(local_goal[0][1]))+(abs(pt[2]),)
            #search the local_vox
            self.local_path = list(a_star_2_5D(local_vox,h, pt, local_goal))
            new_path = []
            new_new_path = []
            for i in range(len(self.local_path[0])):
                new_path.append(list(self.local_path[0][i]))
                new_path[i][0] = new_path[i][0] - self.north_offset
                new_path[i][1] = new_path[i][1] - self.east_offset
                new_path[i][2] = new_path[i][2] - self.local_position[2]
            for i in range (len(new_path)):
                new_new_path.append(tuple(new_path[i]))
            self.local_path = new_new_path
        else:
            self.local_path.append([(-99,-99,-99)])
            local_goal = [(0,0,0)]
        return self.local_path, local_goal

    def create_local_path(self, size=40, height = 10, resolution = 1):
        # pt = global_to_local(self.global_position, self.global_home)
        # pt[0] = pt[0]-self.north_offset
        # pt[1] = pt[1]-self.east_offset
        # pt[2] = abs(pt[2])
        pt = []
        pt.append(self.local_position[1] - self.north_offset)
        pt.append(self.local_position[0] - self.east_offset)
        pt.append(abs(self.local_position[2]))
        n_low = int(pt[0])-int(size)-1
        n_hi  = int(pt[0])+int(size)
        e_low = int(pt[1])-int(size)-1
        e_hi  = int(pt[1])+int(size)
        alt_low = int(abs(pt[2]))-int(height)-1
        alt_hi  = int(abs(pt[2]))+int(height)
        #print("Current Point:",pt)
        if n_low < 0: n_low = 0
        if n_hi >np.array(self.global_voxmap).shape[0]-1: n_hi = np.array(self.global_voxmap).shape[0]-1
        if e_low < 0: e_low = 0
        if e_hi > np.array(self.global_voxmap).shape[1]-1: e_hi = np.array(self.global_voxmap).shape[1]-1
        if alt_low < 0: alt_low = 0
        if alt_hi > np.array(self.global_voxmap).shape[2]-1: alt_hi = np.array(self.global_voxmap).shape[1]-1
        local_vox = self.global_voxmap[n_low:n_hi, e_low:e_hi, alt_low:alt_hi]
        if np.all((local_vox != 0)):
            print("OBSTICALE DETECTED IN LOCAL VOX MAP")
            #print(n_low, n_hi, e_low, e_hi)
            horizon = Polygon((n_hi-1, e_hi-1), (n_low+1, e_hi-1), (n_low+1, e_low+1), (n_hi-1, e_low+1))
            #search out on path for a point outside of the local voxmap
            for i in range(len(self.waypoints)):
                pt = Point(self.waypoints[i])
                if horizon.contains(pt[0:2]):
                    continue
                else:
                    break
            #then assign it as a SEGMENT (starts from current postion looking outward
            path_segment = Segment(Point(pt[0:2]), pt)
            #find the point that it penetrates the local voxmap and use it as the new goal
            local_goal = horizon.intersection(path_segment)
            local_goal = (int(local_goal[0][0]),int(local_goal[0][1]))+(abs(pt[2]),)
            #search the local_vox
            self.local_path = list(a_star_3D(local_vox,h_3D, pt, local_goal))
            new_path = []
            new_new_path = []
            for i in range(len(self.local_path[0])):
                new_path.append(list(self.local_path[0][i]))
                new_path[i][0] = new_path[i][0] - self.north_offset
                new_path[i][1] = new_path[i][1] - self.east_offset
                new_path[i][2] = new_path[i][2] - self.local_position[2]
            for i in range (len(new_path)):
                new_new_path.append(tuple(new_path[i]))
            self.local_path = new_new_path
        else:
            self.local_path.append([(-99,-99,-99)])
            local_goal = [(0,0,0)]
        return self.local_path, local_goal
    
    def simplify_path(self, path_type, cube_size, goal):
        # This routine will iterate through the path to determine of removal of B is allowed from
        # a segment ABC.  Segment AC is tested for collision, if none found it will keep the removal 
        # of B, otherwise it will replace B.  It will then move to the next point.  Once complete it will
        # repeat the process until no changes are made to the path.  The number of repetitions through the path 
        # has shown to be a function of halving the original path length.
        if (path_type == "local"):
            path = self.local_path
        elif (path_type == "global"):
            path = self.global_path
        grid = self.grid
        path_changing = True
        old_path_len = len(path)
        while path_changing:
            i = 0
            while i in range(len(path)-2):
                save_pt = path.pop(i+1)
                if check_path(path[i:i+2], grid, cube_size):
                    path.insert(i+1,save_pt)
                i += 1      
            new_path_len = len(path)
            if new_path_len == old_path_len:
                path_changing = False
            else:
                path_changing = True
                old_path_len = new_path_len
            #print("Path Length:", len(path))
        # The ultiamte path goal may get deleted, add it back
        path.append(goal[0:2])
        path[-1] = path[-1]+(self.target_position[2],)
        return path

    def start(self):
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        self.connection.start()

        # # Only required if they do threaded
        # while self.in_mission:
        #     pass

        self.stop_log()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    #parser.add_argument('--host', type=str, default='10.147.1.222', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()
    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    #my_data = MyDrone(conn)

    time.sleep(1)

    drone.start()