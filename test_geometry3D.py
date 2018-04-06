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

PIXEL_EPSILON = 0.0002

def focal_plane(x):
    x=100*x
    p2 = -5.0e-3
    p4 = -1.25e-7
    p6 = -6.25e-12
    p8 = -3.90625e-16
    p10 = -2.734375e-20
    return 0
    # return (p2 *x**2 + p4 *x**4 + p6 *x**6 + p8 *x **8 +  p10 *x**10)/100.0


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
        self.fig = plt.figure(1,figsize=(7, 7))
        self.ax = self.fig.add_subplot(111)


        # TODO Make this more flexible for lab setup

        self.required_pe = 100
        self.geom = CameraGeometry.from_name("CHEC")
        self.npix = len(self.geom.pix_id)
        self.total_scale = np.ones(self.npix)
        self.pixel_mask = np.ones(self.npix)
        self.camera_curve_center_x = 1
        self.camera_curve_center_y = 0
        self.camera_curve_radius = 1
        self.pixel_pos = np.load("/Users/armstrongt/Software/CTA/CHECsoft/CHECAnalysis/targetpipe/targetpipe/io/checm_pixel_pos.npy")
        self.pix_size = np.sqrt(self.geom.pix_area[0]).value
        self.fiducial_radius = 0.2
        self.fiducial_center = self.camera_curve_center_x + self.camera_curve_radius * np.cos(np.pi) # This is a little confusing, is it going to be set up to change the position of the lightsource
        self.lightsource_distance = 1
        self.lightsource_x = 0
        self.lightsource_y = 0
        self.ang_dist_file = np.loadtxt("/Users/armstrongt/Workspace/CTA/MCValidation/src/ang_dist_2.dat", unpack=True)
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

        photons = self.set_illumination(self.required_pe)
        print(photons)
        print(self.lightsource_angle)

    def set_illumination(self, lambda_):
        self.lamda = lambda_
        self.pde = 0.3936
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
        pix_x = np.unique(np.reshape(self.pixel_pos[0], (32, 64))[10:16])
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


class AnalyseData:
    def __init__(self):
        self.pixel_mask = [1]*2048
        self.calib = CameraCalibrator(None,None)
        self.filename = '/Users/armstrongt/Workspace/CTA/MCValidation/data/bypass2_enoise_pe100.simtel.gz'

    # def __init__(self):
    #     self.file_reader = None
    #     self.r1 = None
    #     self.dl0 = None
    #     self.dl1 = None
    #     self.calculator = None


    def plot_peds(self, event, peds, pedvars):
        """ make a quick plot of the pedestal values"""
        pixid = np.arange(len(peds))
        plt.subplot(1, 2, 1)
        plt.scatter(pixid, peds)
        plt.title("Pedestals for event {}".format(event.r0.event_id))

        plt.subplot(1, 2, 2)
        plt.scatter(pixid, pedvars)
        plt.title("Ped Variances for event {}".format(event.r0.event_id))

    def calc_pedestals(self):
        start = 15
        end = None

        for event in hessio_event_source(self.filename):
            for telid in event.r0.tels_with_data:
                for chan in range(event.r0.tel[telid].adc_samples.shape[0]):
                    traces = event.r0.tel[telid].adc_samples[chan,...]
                    peds, pedvars = pedestals.calc_pedestals_from_traces(traces, start,  end)

                    print("Number of samples: {}".format(traces.shape[1]))
                    print("Calculate over window:({},{})".format(start, end))
                    print("PEDS:", peds)
                    print("VARS:", pedvars)
                    print("-----")

        self.plot_peds(event, peds, pedvars)
        plt.show()


    def get_image(self):

        all_pe = []
        for event in hessio_event_source(self.filename):
            for telid in event.r0.tels_with_data:
                self.calib.calibrate(event)
                im = event.dl1.tel[telid].image
                # print(im[0])
                all_pe = all_pe+list(im[0])
        return all_pe


    def get_charge(self):
        fig = plt.figure(1)
        ax = fig.add_subplot(111)


        calculator = ChargeResolutionCalculator(config=None, tool=None)
        calculator2 = ChargeResolutionCalculator(config=None, tool=None)

        # pevals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 28, 30, 32, 35, 37,
        #           40, 43, 46, 49, 53, 57, 61, 65, 70, 75, 81, 86, 93, 100, 107, 114, 123, 132, 141, 151, 162, 174, 187,
        #           200, 215, 231, 247, 265, 284, 305, 327, 351, 376, 403, 432, 464, 497, 533, 572, 613, 657, 705, 756,
        #           811, 869, 932, 1000]
        pevals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 25, 29, 33, 39, 44, 51, 59, 68, 79, 91, 104, 120, 138,
                  159, 184, 212, 244, 281, 323, 372]

        for n, i in enumerate(pevals):
            self.filename = '/Users/armstrongt/Workspace/CTA/MCValidation/data/bypass2_enoise_pe%s_mask2.simtel.gz' % i
            ntrig = 0
            source = hessio_event_source(self.filename, allowed_tels=None, max_events=None)
            print('evaluating source %s' % self.filename)
            for event in source:
                self.calib.calibrate(event)
                true_charge = event.mc.tel[1].photo_electron_image * self.pixel_mask
                measured_charge = event.dl1.tel[1].image[0] * self.pixel_mask
                true_charge2 = np.asarray([int(i)] * len(measured_charge))* self.pixel_mask
                calculator.add_charges(true_charge, measured_charge)
                calculator2.add_charges(true_charge2, measured_charge)
                ntrig = ntrig + 1

        x, res, res_error, scaled_res, scaled_res_error = calculator.get_charge_resolution()
        x2, res2, res_error2, scaled_res2, scaled_res_error2 = calculator2.get_charge_resolution()
        ax.errorbar(x, res, yerr=res_error, marker='x', linestyle="None", label='MC Charge Res')
        ax.errorbar(x2, res2, yerr=res_error2, marker='x', color='C1', linestyle="None", label='\'Lab\' Charge Res')
        x = np.logspace(np.log10(0.9), np.log10(1000 * 1.1), 100)
        requirement = ChargeResolutionCalculator.requirement(x)
        goal = ChargeResolutionCalculator.goal(x)
        poisson = ChargeResolutionCalculator.poisson(x)
        r_p = ax.plot(x, requirement, 'r', ls='--', label='Requirement')
        g_p = ax.plot(x, goal, 'g', ls='--', label='Goal')
        p_p = ax.plot(x, poisson, c='0.75', ls='--', label='Poisson')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('true charge')
        plt.ylabel('charge resolution')
        plt.show()

    def get_trig_eff(self):
        figtrig = plt.figure(2)
        ax = figtrig.add_subplot(111)



        # pevals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 28, 30, 32, 35, 37,
        #           40, 43, 46, 49, 53, 57, 61, 65, 70, 75, 81, 86, 93, 100, 107, 114, 123, 132, 141, 151, 162, 174, 187,
        #           200, 215, 231, 247, 265, 284, 305, 327, 351, 376, 403, 432, 464, 497, 533, 572, 613, 657, 705, 756,
        #           811, 869, 932, 1000]
        pevals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 25, 29, 33, 39, 44, 51, 59, 68, 79, 91, 104, 120, 138,
                  159, 184, 212, 244, 281, 323, 372]

        sum_image=[]
        sum_true_charge=[]
        sum_true_charge2=[]
        trig=[]
        for n, i in enumerate(pevals):
            self.filename = '/Users/armstrongt/Workspace/CTA/MCValidation/data/bypass2_enoise_pe%s_mask2.simtel.gz' % i
            ntrig = 0
            source = hessio_event_source(self.filename, allowed_tels=None, max_events=None)
            sum_image_i = []
            sum_true_charge_i = []
            print('evaluating source %s' % self.filename)
            for event in source:
                self.calib.calibrate(event)
                true_charge = event.mc.tel[1].photo_electron_image * self.pixel_mask
                measured_charge = event.dl1.tel[1].image[0]
                true_charge2 = np.asarray([int(i)] * len(measured_charge))
                sum_image_i.append(sum(measured_charge))
                sum_true_charge_i.append(sum(true_charge))
                ntrig = ntrig + 1

            sum_true_charge2.append(10*i)
            sum_image.append(np.mean(sum_image_i))
            sum_true_charge.append(np.mean(sum_true_charge_i))
            trig.append(ntrig)

        plt.plot(sum_image, trig, label = 'measured')
        plt.plot(sum_true_charge2, trig, label = 'true input')
        plt.plot(sum_true_charge, trig, label = 'true measured')
        plt.xlabel('total charge')
        plt.ylabel('trigger efficiency')
        plt.legend()

        plt.show()


    def go(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        disp = None
        source = hessio_event_source(self.filename, requested_event=24)
        for event in source:
            self.calib.calibrate(event)
            for i in range(50):
                ipix = np.random.randint(0,2048)
                samp =  event.dl0.tel[1]['pe_samples'][0][ipix]
                # plt.plot(range(len(samp)),samp)

            plt.show()
            if disp is None:
                geom = event.inst.subarray.tel[1].camera
                disp = CameraDisplay(geom)
                # disp.enable_pixel_picker()
                disp.add_colorbar()
                plt.show(block=False)
            #
            im = event.dl1.tel[1].image[0]
            mask = tailcuts_clean(geom, im, picture_thresh=10, boundary_thresh=5)
            im[~mask] = 0.0
            maxpe = max(event.dl1.tel[1].image[0])
            disp.image = im
            print(np.mean(im), '+/-', np.std(im))

        plt.show()


def main():

    # # Just to plot the geometry
    # g = Geometry()
    # # g.pixel_mask = np.random.randint(0,2, 2048)
    # g.plot()
    # g.getscale(plot=True)
    # plt.show()



    # To convert range of desired pe to Lightsource values
    fl = np.loadtxt('/Users/armstrongt/Workspace/CTA/MCValidation/data/d2018-02-09_DynRange_NSB0_GainMatched200mV/runlist.txt', unpack=True)
    g = Geometry()
    # pe = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 28, 30, 32, 35, 37, 40, 43, 46, 49, 53, 57, 61, 65, 70, 75, 81, 86, 93, 100, 107, 114, 123, 132, 141, 151, 162, 174, 187, 200, 215, 231, 247, 265, 284, 305, 327, 351, 376, 403, 432, 464, 497, 533, 572, 613, 657, 705, 756, 811, 869, 932, 1000]
    # pe = [   1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   12, 14,   16,   19,   22,   25,   29,   33,   39,   44,   51,   59, 68,   79,   91,  104,  120,  138,  159,  184,  212,  244,  281, 323,  372,  429,  494,  568,  655,  754,  868, 1000]
    pe = fl[2]
    ph = []
    pe2 =[]
    run = []
    for n,i in enumerate(pe):
        # g.required_pe = i
        photons = g.set_illumination(i)
        # print(i, ' ', photons)
        ph.append(photons)
        pe2.append(round(i,2))
        run.append(int(fl[0][n]))
        # g.getscale(plot=False)
    print(pe2)
    print(ph)
    print(run)

    #
    # # To calculate either the charge resolution of the trigger efficiency, including setting any pixel masks
    # a = AnalyseData()
    # a.pixel_mask = [0]*2048
    # onpix = [838,839,886,887,836,837,884,885]
    # for i in onpix:
    #     a.pixel_mask[i] = 1
    #
    # # a.filename = '/Users/armstrongt/Workspace/CTA/MCValidation/data/bypass2_enoise_pe_e14.simtel.gz'
    # # a.go()
    # a.get_trig_eff()
    # # a.get_charge()

if __name__ == '__main__':
    main()