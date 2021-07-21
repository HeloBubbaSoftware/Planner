import numpy as np

def get_sector(heading, width):
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

print(get_sector(180,10))
