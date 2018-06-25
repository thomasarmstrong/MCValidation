import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mc_cam = np.loadtxt('camera_CHEC-S_GATE.dat', unpack=True)

def focal_plane(x):
	p2 = -5.0e-3
	p4 = -1.25e-7
	p6 = -6.25e-12
	p8 = -3.90625e-16
	p10 = -2.734375e-20
	return p2 *x**2 + p4 *x**4 + p6 *x**6 + p8 *x **8 +  p10 *x**10

fig = plt.figure(1)
ax = fig.add_subplot(111)

# ax.scatter(q_cam[1],q_cam[2], q_cam[3])
# for i in range(len(mc_cam[2])):
	# if mc_cam[2][i] > 0 and mc_cam[3][i] > 0 :
PixID = [1146, 1194,1195,1147,1243,1196,1244,1292,1245,1293]
deadpix = []
for i in range(len(mc_cam[2])):
	if int(mc_cam[0][i]) in PixID:
		ax.scatter(mc_cam[2][i] * 10, mc_cam[3][i] * 10, color='C0')
		plt.text(mc_cam[2][i]*10, mc_cam[3][i]*10, str(mc_cam[0][i]))
	else:
		ax.scatter(mc_cam[2][i] * 10, mc_cam[3][i] * 10, color='C1')
		deadpix.append(int(mc_cam[0][i]))
		# plt.text(mc_cam[2][i] * 10, mc_cam[3][i] * 10, str(mc_cam[0][i]))
# print(deadpix)

plt.xlabel('x [cm]')
plt.ylabel('z [cm]')
plt.title('quarter cam projection')
plt.show() 


