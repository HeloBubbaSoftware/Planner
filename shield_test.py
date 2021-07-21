import argparse
import time
from bresenham import bresenham
import msgpack
import csv
import re
from enum import Enum, auto

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import numpy as np
import visdom
from skimage.morphology import medial_axis
from skimage.util import invert
from planning_utils import a_star, find_nearest, StreamingMovingAverage, pop, check_path, add_heading_to_path, create_grid, create_height_map, find_start_goal, h,h_3D, add_altitude_to_path, over_obstacle_clearance, get_sector
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
import os


from time import sleep
import math

plt.rcParams['figure.figsize'] = 12, 12

class States(Enum)  :
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    AVOID = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING_GLOBAL = auto()
    PLANNING_LOCAL = auto()

class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0, 0.0])
        self.position_snapshot = np.array([0.0,0.0,0.0,0.0])
        self.nearest = np.array([0,0])
        self.data = np.zeros((5000,10))
        self.waypoints = np.zeros((1000,3), dtype="int")
        self.in_mission = True
        self.check_state = {}
        self.global_path = np.zeros((1000,4), dtype="object")
        self.global_path[:,0:3] = self.global_path[:,0:3].astype(int)
        self.global_path[:,3] = self.global_path[:,3].astype(float)
        self.discrete_global_path = np.zeros((10000,2), dtype="uint8")
        self.local_path = self.global_path.copy()
        self.grid = np.zeros((1000,1000),dtype='float')
        self.skeleton = self.grid.copy()
        self.sub_grid = self.grid.copy()
        self.north_offset = 0.0
        self.east_offset = 0.0
        self.heartbeat = 0
        self.quadrant = np.zeros((101), dtype='uint8')
        self.bearing = np.zeros((101), dtype='uint8')
        self.available_headings = np.empty((1000), dtype='uint8')
        self.speed = 0
        self.course = 0
        self.halo_restore = 10
        self.halo_radius = 10
        self.halo_height = 10
        self.halo_buffer = StreamingMovingAverage(100)
        self.halo = np.array((self.halo_height,100), dtype='uint8')
        self.elevation = np.zeros((self.halo_height), dtype='uint8')
        self.target_altitude = 5
        self.safety_distance = 2
        self.target_clearance = 2
        self._local_position_time = 0.9
        
        
        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks hereR
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def _halo(self):
        self.halo[:] = 0
        loc = self.local_position[0:3]
        angle = np.linspace(0, 2*math.pi, 100)
        height = np.linspace(-self.halo_height//2, self.halo_height//2, self.halo_height+1).astype(int)
        compass = np.linspace(0, 360, 100).astype(int)
        loc_north = int(loc[0]-self.north_offset)
        loc_east = int(loc[1]-self.east_offset)
        x = np.array([self.halo_radius*math.cos(theta) for theta in angle]).astype(int)
        y = np.array([self.halo_radius*math.sin(theta) for theta in angle]).astype(int)
        coords = np.stack((x,y), axis=1)
        self.halo = np.zeros((self.halo_height+1,101))
        i, j = 0, 0
        for h in height:
            for c in coords:
                if self.grid[np.clip(int(c[0]+loc_north),0, self.grid.shape[0]-1),np.clip(int(c[1]+loc_east),0,self.grid.shape[1]-1)] > np.clip(int(abs(loc[2])+h),0,10000):
                    self.halo[i, j] = 1
                j +=1
            i += 1
            j = 0
        self.bearing = self._bearing()
        self.elevation = self._elevation()
        self.availalbe_headings = []
        if len(self.bearing) > 0:
            self.available_headings = list(set(compass)-set(self.bearing))
        return self.halo, self.elevation, self.available_headings, loc

    def _quadrant(self):  
        
        return 0

    def _bearing(self):
        self.bearing[:]=0
        bearing = self.bearing
        s = self.halo
        if not np.all((s==0)):
            for i in range(len(s[1])):
                if not np.all((s[:,i]==0)):
                    bearing[i] = int(i*(360/len(s[1])))
        return self.bearing

    def _elevation(self):
        self.elevation[:] = 0
        s = self.halo
        #elevation = np.zeros(len(s)).astype(int)
        elevation = self.elevation
        if not np.all((s==0)): 
            for i in range(len(s)-1):
                if not np.all((s[i,:]==0)):
                    elevation[i] = int(i)
        return self.elevation 

    def _course(self):
        s = self.speed
        n = self._velocity_north
        e = self._velocity_east
        if n > 0 and e > 0:
            self.course =  math.degrees(math.asin(abs(e)/abs(s)))
        elif n >  0 and e <=0:
            self.course =   math.degrees(2*math.pi - math.asin(abs(e)/abs(s)))   
        elif n <= 0 and e > 0:
            self.course =   math.degrees(math.pi - math.asin(abs(e)/abs(s)))  
        elif n <= 0 and e <= 0:
            self.course =   math.degrees(math.pi + math.asin(abs(e)/abs(s)))
        return self.course
                
    def _speed(self):
        n = self._velocity_north
        e = self._velocity_east
        self.speed = np.linalg.norm([abs(n),abs(e)])
        return self.speed
    
    def fly_course_and_speed_until_clear(self,crs, speed):
        north = speed * math.cos(math.radians(crs))
        east  = speed * math.sin(math.radians(crs))
        heading = math.radians(crs)
        self.take_control()
        delta_point  = 10*np.array([north, east]) / abs(np.linalg.norm([north,east]))
        short_target = self.local_position[0:2] + delta_point
        self.cmd_velocity(north, east, 0, heading)
        self.cmd_position(short_target[0], short_target[1], self.local_position[2], heading)

    def local_position_callback(self):
        bearing, elevation, suggested_fly_to, self.position_snapshot = self._halo()
        #print(self.halo_buffer.process(self.halo_radius))
        #self.course = self._course()
        #self.speed = self._speed()
        #print("Frequecy:",self._local_position_frequency)
        #print("L:",self.local_position[0:2].astype(int), "T:",self.target_position[0:2].astype(int))
        if np.any(bearing) and self.flight_state==States.WAYPOINT:
            self.flight_state = States.AVOID

        if np.any(bearing) and self.flight_state==States.AVOID:
            index = list(np.array(self.discrete_global_path)[:,0]).index(min(list(np.array(self.discrete_global_path)[:,0]), key=lambda x: h(x,self.local_position[0:2])))
            index = index +10
            look_ahead = np.clip(index, a_min=0, a_max=len(self.discrete_global_path)-1)
            self.nearest = self.discrete_global_path[look_ahead]
            way_points_in_look_ahead = list(np.array(self.discrete_global_path)[:,index:look_ahead]).count('WP')
            for i in range(way_points_in_look_ahead):
                [self.global_path,sink] = pop(self.global_path, 0)
                [self.waypoints,self.target_position] = pop(self.waypoints,0)
            self.cmd_position(self.nearest[0][0],self.nearest[0][1], self.target_position[2], self.target_position[3])
            self.halo_radius -= 1
            self.halo_height -= 1
            self.halo_radius = np.clip(self.halo_radius, 1,10)
            self.halo_height = np.clip(self.halo_height, 1,10)
            print("HALO DECREMENTED:", self.halo_radius, self.halo_height)
            
            if self.halo_radius <= 1:
                print("ALL STOP")
                self.cmd_velocity(0,0,0,0)

        if not np.any(bearing) and self.flight_state==States.AVOID:
            self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

            self.flight_state = States.WAYPOINT

        if not np.any(bearing) and self.flight_state==States.WAYPOINT:
            
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 0.5:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()
            else:
                self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])
        
        # if self.halo_buffer.process(self.halo_radius) > 9.0 and self.flight_state == States.AVOID:
        #     self.halo_radius = self.halo_restore
        #     self.halo_height = self.halo_restore
        #     #self.halo_radius = np.clip(self.halo_radius,1,10)
        #     print("==================>Halo Restored to:", self.halo_radius, self.halo_height)
        #     self.flight_state = States.WAYPOINT

        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.90 * self.target_position[2]:
                self.waypoint_transition()
            

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
                self.plan_path("global")
            elif self.flight_state == States.AVOID:
                self.plan_path("local")
            elif self.flight_state == States.PLANNING_GLOBAL:
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
        self.arm()
        #self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        [self.global_path,sink] = pop(self.global_path, 0)
        [self.waypoints,self.target_position] = pop(self.waypoints,0)
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
        #print("Sending waypoints to simulator ...")
        pack = [[int(p[0]) , int(p[1]) , int(p[2])] for p in self.waypoints]
        self.waypoints = np.array(self.waypoints)
        data = msgpack.dumps(pack)
        #clear = msgpack.dumps([[0,0,0]])
        #print("after dumps")
        #self.connection._master.write(clear)
        #sleep(0.01)
        self.connection._master.write(data)
        #print("after write")
        return 0

    def discretize_global_path(self):
        path = self.global_path
        d_path = []
        for i in range(len(path)-1):
            bres = list(bresenham(path[i][0],path[i][1],path[i+1][0],path[i+1][1]))
            for j in range(len(bres)-1):
                a = [bres[j][0]+self.north_offset,bres[j][1]+self.east_offset]
                for k in range(len(self.global_path)):
                    b = 0
                    if bres[j] == tuple(self.global_path[k][0:2]):
                        b = "WP"
                        break
                    else:
                        b = 0
                d_path.append([a,b])
        return d_path
            
        self.cmd_position(self.position_snapshot[0], self.position_snapshot[1], abs(self.position_snapshot[2]), self.course)

    def plan_path(self, local_or_global):
        if local_or_global == "global":
            self.flight_state = States.PLANNING_GLOBAL
            self.set_home_as_current_position()
            self.set_local_position = global_to_local((self.global_position[0], self.global_position[1], self.global_position[2]),self.global_home)
            self.data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
            self.grid, self.north_offset, self.east_offset = create_grid(self.data, self.target_altitude, self.safety_distance)
            print("Starting Medial Axis")
            self.skeleton = medial_axis(invert(np.array(self.grid).astype(bool)))
            print("Finished Medial Axis")
            print("Grid and Skeleton created")
            print("Searching for a path ...")
            self.target_position[2] = self.target_altitude
            grid_start = np.array([int(self.local_position[0])-self.north_offset, int(self.local_position[1])-self.east_offset])
            grid_goal = np.array([840,25])
            print("Starting Start Goal)")
            start_ne, goal_ne = find_start_goal(self.skeleton, grid_start, grid_goal)
            print("finishing Start Goal")
            print("Starting AStar")
            self.global_path, cost = a_star(self.global_path,self.skeleton.astype(int), h, tuple(start_ne), tuple(goal_ne))
            print("Finished AStar")
            self.global_path = add_altitude_to_path(self.global_path, self.target_altitude)
            print("Original Path Length:", len(self.global_path))
            original_path = self.global_path.copy()
            cube_size = 2
            self.global_path = self.simplify_path("global", cube_size, grid_goal)
            print("After Simplification Path Length:", len(self.global_path))
            self.global_path = add_heading_to_path(self.global_path, self.north_offset, self.east_offset, self.global_home)
            self.discrete_global_path = self.discretize_global_path()
            #self.waypoints = np.resize(self.waypoints, (len(self.global_path),3))
            self.waypoints = [[int(p[0] + self.north_offset), int(p[1] + self.east_offset), int(p[2]), float(p[3])] for p in self.global_path]
            
            # print("plotting now")
            # plt.imshow(self.grid, cmap='hot', origin='lower')
            # plt.imshow(self.skeleton, cmap='Greys', origin='lower', alpha=0.7)
            # plt.plot(start_ne[1], start_ne[0], 'bx')
            # plt.plot(goal_ne[1], goal_ne[0], 'bx')
            # pp2 = np.array(original_path)
            # plt.plot(pp2[:, 1], pp2[:, 0], 'r')
            # pp = np.array(self.global_path)
            # plt.plot(pp[:, 1], pp[:, 0], 'g')
            # plt.xlabel('EAST')
            # plt.ylabel('NORTH')
            # plt.show()

            self.send_waypoints()
            

        if local_or_global == "local":
            return


    def start(self):
        #self.start_log("Logs", "NavLog.txt")
        print("Connection Starting...")
        self.connection.start()
        # while self.in_mission:   
        #     pass
        self.stop_log
        
        # # Only required if they do threaded
        

    def simplify_path(self, path_type, cube_size, goal):
        # This routine will iterate through the path to determine of removal of B is allowed from
        # a segment ABC.  Segment AC is tested for collision, if none found it will keep the removal 
        # of B, otherwise it will replace B.  It will then move to the next point.  Once complete it will
        # repeat the process until no changes are made to the path.  The number of repetitions through the path 
        # has shown to be a function of halving the original path length.
        if (path_type == "local"):
            path = list(self.local_path)
            grid = self.sub_grid
        elif (path_type == "global"):
            path = list(self.global_path)
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
        #path[-1] = np.append((goal).astype(int),[self.target_altitude,0])
        return np.array(path)
 
if __name__ == "__main__":
    #os.system('sudo ./sim')
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='localhost', help="host address, i.e. '127.0.0.1'")
    #parser.add_argument('--host', type=str, default='10.147.1.222', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()
    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=360)
    drone = MotionPlanning(conn)
    #my_data = MyDrone(conn)          

    time.sleep(1)

    drone.start()            