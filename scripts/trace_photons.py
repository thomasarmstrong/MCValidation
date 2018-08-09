import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.ion()

fig0=plt.figure(1)
ax0 = fig0.add_subplot(111, projection='3d')

logfile = open('nohup.out','r')

z0 = 0
x1 = 0
y1 = 0
z1 = 0
x2 = 0
y2 = 0
z2 = 0

for line in logfile:
    if line.startswith('Starting position w.r.t. camera front:'):
        data = line.split(' ')[5].split('=')[1].split(',')
        x1 = float(data[0])
        y1 = float(data[1])
        z1 = -float(data[2])
    elif line.startswith('Distance of emission point'):
        z0 = float(line.split(' ')[5])
    elif line.startswith('Line-plane intersection at '):
        data = line.split('(')[1].split(')')[0].split(',')
        x2 = float(data[0])
        y2 = float(data[1])
        z2 = -float(data[2])
        # ax0.plot([0,x1], [0,y1], [z0,z1], color='r')
        ax0.plot([x1,x2], [y1,y2], [z1,z2], color='r')
        plt.pause(0.001)
