import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


file_name = '../02_cameraConfigFile/camera_CHEC-S_prototype_26042018.dat'

file = open(file_name,'r')
pixn=[]
x=[]
y=[]
xr=[]
yr=[]
for line in file:
    if line.startswith('Pixel'):
        data = line.split('\t')
        # print(data)
        pixn.append(int(data[1].split(' ')[0]))
        x.append(float(data[2]))
        y.append(float(data[3]))
        xr.append(round(float(data[2]),1))
        yr.append(round(float(data[3]),1))
        # print(round(float(data[2]),1))
        # exit()
file.close()
file2 = open(file_name, 'r')
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(212)

x2=[]
y2=[]
z1=[]
z2=[]

pixelDiameter = 0.62

for line2 in file2:
    if line2.startswith('MajorityTrigger'):
        data = line2.split(' ')
        d0 = int(data[3].split('[')[0])
        d1 = int(data[3].split('[')[1].split(',')[0])
        d2 = int(data[3].split('[')[1].split(',')[1])
        d3 = int(data[3].split('[')[1].split(',')[2].split(']')[0])

        x2.append(x[d0])
        y2.append(y[d0])
        dist1 = np.sqrt((x[d1]-x[d0])**2 + (y[d1]-y[d0])**2)
        dist2 = np.sqrt((x[d2]-x[d0])**2 + (y[d2]-y[d0])**2)
        dist3 = np.sqrt((x[d3]-x[d1])**2 + (y[d3]-y[d1])**2)
        dist4 = np.sqrt((x[d3]-x[d2])**2 + (y[d3]-y[d2])**2)
        z1.append(round(dist1, ndigits=4))
        z2.append(round(dist2, ndigits=4))
        pixels, edgePixels, offPixels = list(), list(), list()
        spixels, sedgePixels, soffPixels = list(), list(), list()
        square = mpatches.Rectangle((float(x[d0])-pixelDiameter/2, float(y[d0])-pixelDiameter/2), width=pixelDiameter, height=pixelDiameter)
        pixels.append(square)
        ssquare = mpatches.Rectangle((np.mean([x[d0],x[d1],x[d2],x[d3]])-pixelDiameter, np.mean([y[d0],y[d1],y[d2],y[d3]])-pixelDiameter), width=2*pixelDiameter, height=2*pixelDiameter)
        spixels.append(ssquare)

        sspixels = []
        data2 = data[3:-2]
        for i in data2:
            dd0 = int(i.split('[')[0])
            dd1 = int(i.split('[')[1].split(',')[0])
            dd2 = int(i.split('[')[1].split(',')[1])
            dd3 = int(i.split('[')[1].split(',')[2].split(']')[0])
            ssquare = mpatches.Rectangle((float(x[dd0]) - pixelDiameter / 2, float(y[dd0]) - pixelDiameter / 2),
                                        width=pixelDiameter, height=pixelDiameter)
            sspixels.append(ssquare)
            # plt.text(x[dd0], y[dd0], pixn[dd0], fontsize=6)
            ssquare = mpatches.Rectangle((float(x[dd1]) - pixelDiameter / 2, float(y[dd1]) - pixelDiameter / 2),
                                        width=pixelDiameter, height=pixelDiameter)
            sspixels.append(ssquare)
            # plt.text(x[dd1], y[dd1], pixn[dd1], fontsize=6)
            ssquare = mpatches.Rectangle((float(x[dd2]) - pixelDiameter / 2, float(y[dd2]) - pixelDiameter / 2),
                                        width=pixelDiameter, height=pixelDiameter)
            sspixels.append(ssquare)
            # plt.text(x[dd2], y[dd2], pixn[dd2], fontsize=6)
            ssquare = mpatches.Rectangle((float(x[dd3]) - pixelDiameter / 2, float(y[dd3]) - pixelDiameter / 2),
                                        width=pixelDiameter, height=pixelDiameter)
            sspixels.append(ssquare)
            # plt.text(x[dd3], y[dd3], pixn[dd3], fontsize=6)
        # plt.scatter(xr, yr, marker='s', alpha=0.3, color='C1')
        plt.scatter(x, y, marker='.', alpha=0.3, color='C0')
        ax1.add_collection(PatchCollection(sspixels, facecolor='none', edgecolor='y'))
        ax1.add_collection(PatchCollection(pixels, facecolor='r', edgecolor='k'))
        ax1.add_collection(PatchCollection(spixels, facecolor='none', edgecolor='k'))
        fig.canvas.draw()
        plt.pause(0.01)
        xTitle = 'Horizontal scale [cm]'
        yTitle = 'Vertical scale [cm]'
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel(xTitle)
        plt.ylabel(yTitle)
        # if plt.waitforbuttonpress():
        #     continue

        fig.canvas.flush_events()
        plt.cla()



plt.show()