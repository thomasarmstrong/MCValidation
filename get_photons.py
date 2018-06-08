"""Author: Jason Watson - Edited by Thomas Armstrong"""

from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, pathpatch_2d_to_3d
import math as m
from ctapipe.io.hessio import hessio_event_source
from ctapipe.image import tailcuts_clean
from ctapipe.calib import pedestals
from ctapipe.calib import CameraCalibrator, CameraDL1Calibrator
from ctapipe.analysis.camera.chargeresolution import ChargeResolutionCalculator
from scipy.interpolate import interp1d
import argparse


PIXEL_EPSILON = 0.0002

def focal_plane(x):
    x=100*x
    p2 = -5.0e-3
    p4 = -1.25e-7
    p6 = -6.25e-12
    p8 = -3.90625e-16
    p10 = -2.734375e-20
    # return 0
    return (p2 *x**2 + p4 *x**4 + p6 *x**6 + p8 *x **8 +  p10 *x**10)/100.0

def cart2sph(x,y,z):
    l = x**2 + y**2
    r = np.sqrt(l + z**2)
    elev = m.atan2(z,np.sqrt(l))
    az = m.atan2(y,x)
    return r, elev, az


def sph2cart(r, elev, az):
    x = r * np.cos(elev) * np.cos(az)
    y = r * np.cos(elev) * np.sin(az)
    z = r * np.sin(elev)
    return x, y, z


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


class Geometry:
    def __init__(self):
        # TODO Make this more flexible for lab setup
        self.fig = plt.figure(1,figsize=(7, 7))
        self.ax = self.fig.add_subplot(111)
        self.required_pe = 100
        self.camera = "CHEC"
        self.camera_curve_center_x = 1
        self.camera_curve_center_y = 0
        self.camera_curve_radius = 1
        self.total_scale = 0
        self.pixel_mask = 0
        self.ang_dist_file_string = "ang_dist_2.dat"
        self.geom = None
        self.npix = 0
        self.pix_size = None
        self.fiducial_center = None
        self.lightsource_distance = None
        self.fiducial_radius = 0.2
        self.lightsource_distance = 1
        self.lightsource_x = 0
        self.lightsource_y = 0
        self.ang_dist_file = None
        self.lightsource_angle = None
        self.lightsource_sa = None
        self.fiducial_sa = None
        self.fiducial_percent = None
        self.pix_sa = None
        self.pix_percent = None
        self.lamda = None
        self.pde = None
        self.photons_pixel = None
        self.photons_ls = None
        self.photons_fiducial = None

    def setup_geom(self):

        self.geom = CameraGeometry.from_name(self.camera)
        self.npix = len(self.geom.pix_id)
        self.total_scale = np.ones(self.npix)
        self.pixel_mask = np.ones(self.npix)
        # self.pixel_pos = np.load("/Users/armstrongt/Software/CTA/CHECsoft/CHECAnalysis/targetpipe/targetpipe/io/checm_pixel_pos.npy")
        self.pix_size = np.sqrt(self.geom.pix_area[0]).value
        self.fiducial_center = self.camera_curve_center_x + self.camera_curve_radius * np.cos(np.pi) # This is a little confusing, is it going to be set up to change the position of the lightsource
        self.ang_dist_file = np.loadtxt(self.ang_dist_file_string, unpack=True)
        theta_f = np.pi / 2 - np.arccos(self.fiducial_radius / self.lightsource_distance)
        # theta_ls = theta_f#self.lightsource_angle * np.pi / 180
        theta_p = np.arctan(self.pix_size / (2 * self.lightsource_distance))
        # self.lightsource_angle = theta_ls * 180 / np.pi
        self.lightsource_angle = max(self.ang_dist_file[0])
        theta_ls = self.lightsource_angle * np.pi / 180
        self.lightsource_sa = 2 * np.pi * (1 - np.cos(theta_ls))
        self.fiducial_sa = 2 * np.pi * (1 - np.cos(theta_f))
        self.fiducial_percent = self.fiducial_sa / self.lightsource_sa
        if self.fiducial_percent > 1: self.fiducial_percent = 1
        self.pix_sa = 2 * np.pi * (1 - np.cos(theta_p))
        self.pix_percent = self.pix_sa / self.lightsource_sa
        if self.pix_percent > 1: self.pix_percent = 1

        # photons = self.set_illumination(self.required_pe)
        # print(photons)
        # print(self.lightsource_angle)

    def set_illumination(self, lambda_, tote_trans):
        self.lamda = lambda_
        self.pde = tote_trans
        self.photons_pixel = self.lamda / self.pde
        self.photons_ls = self.photons_pixel / self.pix_percent
        self.photons_fiducial = self.photons_ls * self.fiducial_percent
        return self.photons_ls

    def plot_camera_circle(self):
        x = self.camera_curve_center_x
        y = self.camera_curve_center_y
        r = self.camera_curve_radius
        self.ax.add_artist(plt.Circle((x, y), r, color='b', fill=False))

    def plot_pixels(self):
        pix_x = np.unique(np.reshape(self.geom.pix_x, (32, 64))[10:16])
        pix_cy = np.reshape(pix_x, (6, 8))
        module_centers = pix_cy.mean(1)
        x0 = self.camera_curve_center_x
        y0 = self.camera_curve_center_y
        r = self.camera_curve_radius

        mod_y = module_centers
        theta = np.arcsin((mod_y - y0)/r) + np.pi
        x = x0 + r * np.cos(theta)
        mr = (mod_y - y0) / (x - x0)
        mt = -1 / mr
        pix_cx = ((pix_cy - mod_y[:, None]) / mt[:, None]) + x[:, None]
        pix_x1 = pix_cx - (self.pix_size / 2) / np.sqrt(1 + mt[:, None] ** 2)
        pix_x2 = pix_cx + (self.pix_size / 2) / np.sqrt(1 + mt[:, None] ** 2)
        pix_y1 = (pix_x1 - x[:, None]) * mt[:, None] + mod_y[:, None]
        pix_y2 = (pix_x2 - x[:, None]) * mt[:, None] + mod_y[:, None]
        pix_x = np.vstack([pix_x1.ravel(), pix_x2.ravel()])
        pix_y = np.vstack([pix_y1.ravel(), pix_y2.ravel()])
        self.ax.plot(pix_x, pix_y, color='r')

    def plot_focal_plane(self):
        self.ax.axvline(self.fiducial_center)

    def plot_fiducial_sphere(self):
        x0 = self.fiducial_center
        y0 = self.camera_curve_center_y
        r = self.fiducial_radius
        self.ax.add_artist(plt.Circle((x0, y0), r, color='b', fill=False))

    def plot_lightsource(self):
        x = self.fiducial_center - self.lightsource_distance
        y = self.camera_curve_center_y
        self.ax.plot(x, y, 'x')
        angle = self.lightsource_angle
        wedge = patches.Wedge((x, y), 2*self.lightsource_distance, -angle, angle, alpha=0.5)
        self.ax.add_patch(wedge)

    # def plot_values(self):
    #     text = "Lightsource = {} photons \n" \
    #            "Fiducial = {} photons \n" \
    #            "Pixel = {} photons \n" \
    #            "Pixel = {} p.e.".format(self.photons_ls, self.photons_fiducial,
    #                                     self.photons_pixel, self.lamda)
    #     self.ax.text(0.5, 0.5, text, transform=self.ax.transAxes)

    def plot_values(self):
        text = "Pixel = %s p.e\n" \
               "Pixel = %.2f photons \n" \
               "Lightsource = %.2f photons." % (self.lamda,
                                        self.photons_pixel, self.photons_ls)
        self.ax.text(0.1, 0.9, text, transform=self.ax.transAxes)


    def plot_values3d(self):
        text = "Lightsource = {} photons \n" \
               "Fiducial = {} photons \n" \
               "Pixel = {} photons \n" \
               "Pixel = {} p.e.".format(self.photons_ls, self.photons_fiducial,
                                        self.photons_pixel, self.lamda)
        self.ax.text(0.5, 0.5, 'test')

    def plot(self):
        self.plot_camera_circle()
        self.plot_pixels()
        self.plot_focal_plane()
        self.plot_fiducial_sphere()
        self.plot_lightsource()
        self.plot_values()
        plt.ylabel('Y')
        plt.xlabel('Z')
        # self.ax.set_xlim([-1, 0.05])
        # self.ax.set_ylim([-0.4, 0.4])

    def get_pixel_angscale(self, xc, yc):
        r, e, a = cart2sph(xc-self.lightsource_x, yc-self.lightsource_y, focal_plane(
            np.sqrt((xc) ** 2 + (yc) ** 2)) - self.lightsource_distance)
        index = np.searchsorted(self.ang_dist_file[0], e*180/np.pi+90)
        if index==0: index=0
        else: index = index -1
        return self.ang_dist_file[1][index]

    def get_2dverts(self, xc, yc):
        verts = []
        verts.append((xc - self.pix_size / 2 -  PIXEL_EPSILON, yc - self.pix_size / 2 - PIXEL_EPSILON))
        verts.append((xc + self.pix_size / 2 + PIXEL_EPSILON, yc - self.pix_size / 2 - PIXEL_EPSILON))
        verts.append((xc + self.pix_size / 2 + PIXEL_EPSILON, yc + self.pix_size / 2 + PIXEL_EPSILON))
        verts.append((xc - self.pix_size / 2 - PIXEL_EPSILON, yc + self.pix_size / 2 + PIXEL_EPSILON))
        return verts

    def get_3dverts(self, xc, yc):
        verts = []
        verts.append((focal_plane(np.sqrt((xc - self.pix_size / 2) ** 2 + (yc - self.pix_size / 2) ** 2)),
                       xc - self.pix_size / 2, yc - self.pix_size / 2))
        verts.append((focal_plane(np.sqrt((xc + self.pix_size / 2) ** 2 + (yc - self.pix_size / 2) ** 2)),
                       xc + self.pix_size / 2, yc - self.pix_size / 2))
        verts.append((focal_plane(np.sqrt((xc + self.pix_size / 2) ** 2 + (yc + self.pix_size / 2) ** 2)),
                       xc + self.pix_size / 2, yc + self.pix_size / 2))
        verts.append((focal_plane(np.sqrt((xc - self.pix_size / 2) ** 2 + (yc + self.pix_size / 2) ** 2)),
                       xc - self.pix_size / 2, yc + self.pix_size / 2))
        return verts

    def get_pixel_scaledarea(self, xc, yc):
        dist_to_pix = np.sqrt((xc-self.lightsource_x) ** 2 + (yc-self.lightsource_y) ** 2 + (self.lightsource_distance - focal_plane(np.sqrt(xc ** 2 + yc ** 2))) ** 2)

        r1, e1, a1 = cart2sph(xc - self.pix_size / 2 - self.lightsource_x, yc - self.pix_size / 2 - self.lightsource_y, focal_plane(
            np.sqrt((xc - self.pix_size / 2) ** 2 + (yc - self.pix_size / 2) ** 2)) - self.lightsource_distance)
        r2, e2, a2 = cart2sph(xc + self.pix_size / 2 - self.lightsource_x, yc - self.pix_size / 2 - self.lightsource_y, focal_plane(
            np.sqrt((xc + self.pix_size / 2) ** 2 + (yc - self.pix_size / 2) ** 2)) - self.lightsource_distance)
        r3, e3, a3 = cart2sph(xc + self.pix_size / 2 - self.lightsource_x, yc + self.pix_size / 2 - self.lightsource_y, focal_plane(
            np.sqrt((xc + self.pix_size / 2) ** 2 + (yc + self.pix_size / 2) ** 2)) - self.lightsource_distance)
        r4, e4, a4 = cart2sph(xc - self.pix_size / 2 - self.lightsource_x, yc + self.pix_size / 2 - self.lightsource_y, focal_plane(
            np.sqrt((xc - self.pix_size / 2) ** 2 + (yc + self.pix_size / 2) ** 2)) - self.lightsource_distance)

        x1, y1, z1 = sph2cart(dist_to_pix, e1, a1)
        x2, y2, z2 = sph2cart(dist_to_pix, e2, a2)
        x3, y3, z3 = sph2cart(dist_to_pix, e3, a3)
        x4, y4, z4 = sph2cart(dist_to_pix, e4, a4)

        area = PolyArea([x1, x2, x3, x4], [y1, y2, y3, y4])
        return area

    def plot3dcamera(self, verts, scale=None):
        fig = plt.figure()
        ax1 = fig.add_subplot(111,projection='3d')
        poly3d = Poly3DCollection(verts)
        if not scale:
            poly3d.set_array(np.array(self.total_scale)/max(self.total_scale))
        else:
            poly3d.set_array(np.array(scale) / max(scale))
        ax1.add_collection3d(poly3d)

        theta = np.linspace(0, 2 * np.pi, 90)
        r = -np.arange(0, self.fiducial_radius, 0.01)
        T, R = np.meshgrid(theta, r)
        X = R * np.cos(T)
        Y = R * np.sin(T)
        Z = -np.sqrt(X ** 2 + Y ** 2)
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = self.fiducial_radius* np.cos(u) * np.sin(v)
        y = self.fiducial_radius*np.sin(u) * np.sin(v)
        z = self.fiducial_radius* np.cos(v)
        ax1.plot_wireframe(self.lightsource_distance+Z*(self.lightsource_distance/0.2),X+ self.lightsource_x,  Y+ self.lightsource_y, alpha = 0.1, color='C0')
        ax1.plot_wireframe(x, y, z, color="C0", alpha = 0.1)

        lim = max(self.geom.pix_x).value

        ax1.set_ylim3d(-lim, lim)
        ax1.set_zlim3d(-lim, lim)
        ax1.set_xlabel('Z')
        ax1.set_ylabel('X')
        ax1.set_zlabel('Y')
        fig.colorbar(poly3d, ax=ax1)

        colors = ['0.6', '0.5', '0.2', 'c', 'm', 'y']
        # for i, (z, zdir) in enumerate(product([self.lightsource_distance, self.lightsource_distance+0.3], ['x', 'y', 'z'])):
        side = Rectangle((self.lightsource_distance, -0.01+self.lightsource_x), 0.1, 0.02, facecolor=colors[0])
        ax1.add_patch(side)
        pathpatch_2d_to_3d(side, z=0.01+self.lightsource_x, zdir='z')

        side = Rectangle((self.lightsource_distance, -0.01+self.lightsource_x), 0.1, 0.02, facecolor=colors[0])
        ax1.add_patch(side)
        pathpatch_2d_to_3d(side, z=-0.01+self.lightsource_x, zdir='z')

        side = Rectangle((self.lightsource_distance, -0.01+ self.lightsource_y), 0.1, 0.02, facecolor=colors[1])
        ax1.add_patch(side)
        pathpatch_2d_to_3d(side, z=0.01+self.lightsource_x, zdir='y')

        side = Rectangle((self.lightsource_distance, -0.01 + self.lightsource_y), 0.1, 0.02, facecolor=colors[1])
        ax1.add_patch(side)
        pathpatch_2d_to_3d(side, z=-0.01+self.lightsource_x, zdir='y')

        side = Rectangle((-0.01+self.lightsource_x, -0.01+ self.lightsource_y), 0.02, 0.02, facecolor=colors[1])
        ax1.add_patch(side)
        pathpatch_2d_to_3d(side, z=self.lightsource_distance, zdir='x')

        side = Rectangle((-0.01 + self.lightsource_x, -0.01 + self.lightsource_y), 0.02, 0.02, facecolor=colors[1])
        ax1.add_patch(side)
        pathpatch_2d_to_3d(side, z=self.lightsource_distance+0.1, zdir='x')

        self.plot_values3d()

    def plot2dscales(self, verts, area2, scale2):
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        poly2d1 = PolyCollection(verts)
        poly2d1.set_array(np.array(area2)/max(area2))

        poly2d2 = PolyCollection(verts)
        poly2d2.set_array(np.array(scale2)/max(scale2))

        poly2d3 = PolyCollection(verts)
        poly2d3.set_array(np.array(self.total_scale)/max(self.total_scale))

        poly2d4 = PolyCollection(verts)
        poly2d4.set_array(self.required_pe * np.array(self.total_scale)/max(self.total_scale))

        lim = max(self.geom.pix_x).value

        ax1.add_collection(poly2d1)
        ax1.set_xlim(-lim, lim)
        ax1.set_ylim(-lim, lim)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Geometry Scale')

        ax2.add_collection(poly2d2)
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Angular Pulse Scale')

        ax3.add_collection(poly2d3)
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('Total Scale')

        ax4.add_collection(poly2d4)
        ax4.set_xlim(-lim, lim)
        ax4.set_ylim(-lim, lim)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title('p.e. per pixel')

        fig.colorbar(poly2d1, ax=ax1)
        fig.colorbar(poly2d2, ax=ax2)
        fig.colorbar(poly2d3, ax=ax3)
        fig.colorbar(poly2d4, ax=ax4)

    def getscale(self, plot=False):

        area2 = []
        scale2 = []
        verts2d=[]
        verts3d=[]
        for ipix in range(self.npix):
            xc = self.geom.pix_x[ipix].value
            yc = self.geom.pix_y[ipix].value

            verts2d.append(self.get_2dverts(xc, yc))
            verts3d.append(self.get_3dverts(xc, yc))

            if self.pixel_mask[ipix] == 0:
                self.total_scale[ipix] = 0
                scale2.append(0)
                area2.append(0)
                continue

            area = self.get_pixel_scaledarea(xc, yc)
            scale = self.get_pixel_angscale(xc, yc)
            scale2.append(scale)
            area2.append(area)
            self.total_scale[ipix] = area*scale

        if plot:
            self.plot3dcamera(verts3d)
            self.plot2dscales(verts2d, area2, scale2)

        return 0


def main():

    print('WARNING THIS NEEDS WORK, mostly hardcoded')
    parser = argparse.ArgumentParser(description='Get number of photons for LightEmission package for a requested p.e. per pixel')
    parser.add_argument('--runfile', help='Path to runfile that contains run number and requested/estimated p.e.')
    parser.add_argument('--pdefile', help='Path to PDE file, used to obtain efficiency at given wavelength')
    parser.add_argument('--transmission', help='Path to additional transmission file, used to obtain efficiency at given wavelength', default=None)
    parser.add_argument('--wavelength', default=405, type=float, help='wavelength of lightsource used')
    parser.add_argument('--camera', default="CHEC", help='ctapipe camera geometery to use')
    parser.add_argument('--ls_distance', default=1, type=float, help='distance of lightsource to camera focal plane [m]')
    parser.add_argument('--angular_distribution', default="ang_dist_2.dat", help='file containing angular distribution')
    args = parser.parse_args()

    # To convert range of desired pe to Lightsource values
    try:
        fl = np.loadtxt(args.runfile, unpack=True)
        pe = fl[2]
        run = fl[0]
    except OSError:
        pe = [float(args.runfile)]
        run =[1]

    ph = []
    pe2 =[]
    # run = []

    pde = np.loadtxt(args.pdefile,unpack=True)
    fpde = interp1d(pde[0], pde[1])
    if args.transmission != None:
        trans = np.loadtxt(args.transmission, unpack=True)
        ftrans = interp1d(trans[0], trans[1])

        tote_trans = fpde(args.wavelength) * ftrans(args.wavelength)
    else:
        tote_trans = fpde(args.wavelength)

    # print(tote_trans)
    # exit()

    g = Geometry()
    g.lightsource_distance = args.ls_distance
    g.ang_dist_file_string = args.angular_distribution
    g.camera = args.camera
    g.setup_geom()
    # pe = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 28, 30, 32, 35, 37, 40, 43, 46, 49, 53, 57, 61, 65, 70, 75, 81, 86, 93, 100, 107, 114, 123, 132, 141, 151, 162, 174, 187, 200, 215, 231, 247, 265, 284, 305, 327, 351, 376, 403, 432, 464, 497, 533, 572, 613, 657, 705, 756, 811, 869, 932, 1000]
    # pe = [   1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   12, 14,   16,   19,   22,   25,   29,   33,   39,   44,   51,   59, 68,   79,   91,  104,  120,  138,  159,  184,  212,  244,  281, 323,  372,  429,  494,  568,  655,  754,  868, 1000]

    for n,i in enumerate(pe):
        # g.required_pe = i
        photons = g.set_illumination(i, tote_trans)
        # print(i, ' ', photons)
        ph.append(round(photons,2))
        pe2.append(round(i,2))
        # run.append(int(fl[0][n]))
        # g.getscale(plot=False)
    print('number of requested photoelectrons')
    print(pe2)
    print('number of required emitted photons')
    print(ph)
    print('Run number')
    print(run)



if __name__ == '__main__':
    main()