import math

def calDistance(a, b):
    centerAx = (a[2] + a[0]) / 2
    centerAy = (a[3] + a[1]) / 2
    centerBx = (b[2] + b[0]) / 2
    centerBy = (b[3] + b[1]) / 2
    d = math.hypot(centerAx - centerBx, centerAy - centerBy)
    return d