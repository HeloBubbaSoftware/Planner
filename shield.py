import math
import numpy as np

def shield(radius, ht):
    angle = np.linspace(0, 2*math.pi, 100)
    height = np.linspace(-ht//2, ht//2, ht+1).astype(int)
    compass = np.linspace(0, 360, 100).astype(int)
    loc_north = int(loc[0]-self.north_offset)
    loc_east = int(loc[1]-self.east_offset)
    x = np.array([radius*math.cos(theta) for theta in angle]).astype(int)
    y = np.array([radius*math.sin(theta) for theta in angle]).astype(int)
    coords = np.stack((x,y), axis=1)
    shield_ret = np.zeros((ht+1,101))
    i, j = 0, 0
    for h in height:
        for c in coords:
            if self.grid[int(c[0]+loc_north),int(c[1]+loc_east)] > np.clip(int(abs(loc[2])+h),0,10000):
                shield_ret[i, j] = 1
            j +=1
        i += 1
        j = 0
    vect = bearing(shield_ret)
    ele = elevation(shield_ret)
    if len(vect) > 0:
        diff = list(set(compass)-set(vect))
    return bearing(shield_ret), elevation(shield_ret), diff

def quadrant(s):   
    s = self.shield    
    north = np.stack((s[:,315:359],s[:,0:44],), axis=1)
    east = s[:, 45:134]
    south = s[:, 135:224]
    west = s[:, 225:314]
    return [np.all(north==0),np.all(east==0), np.all(south==0), np.all(west==0)]

def bearing(self):
    bearing = self.bearing
    if not np.all((s==0)):
        for i in range(len(s[1])):
            if not np.all((s[:,i]==0)):
                bearing.append(int(i*(360/len(s[1]))))
    return bearing

def elevation(self):
    s = self.shield
    #elevation = np.zeros(len(s)).astype(int)
    elevation = []
    if not np.all((s==0)): 
        for i in range(len(s)-1):
            if not np.all((s[i,:]==0)):
                elevation.append(int(i))
    return elevation 

def course(self):
    s = self.speed
    n = self._velocity_north
    e = self._velocity_east
    if n > 0 and e > 0:
        return math.degrees(math.asin(abs(e)/abs(s)))
    elif n >  0 and e <=0:
        return math.degrees(2*math.pi - math.asin(abs(e)/abs(s)))   
    elif n <= 0 and e > 0:
        return math.degrees(math.pi - math.asin(abs(e)/abs(s)))  
    elif n <= 0 and e <= 0:
        return math.degrees(math.pi + math.asin(abs(e)/abs(s)))
            
def speed(self):
    n = self._velocity_north
    e = self._velocity_east
    return (np.linalg.norm([abs(n),abs(e)]))