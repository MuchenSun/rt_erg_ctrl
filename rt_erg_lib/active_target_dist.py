# import rospy

import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal as mvn

class TargetDist(object):
    '''
    This is going to be a test template for the code,
    eventually a newer version will be made to interface with the
    unity env
    '''

    def __init__(self, num_nodes=2, num_pts=50, size=1.0, \
                 means=[[0.3,0.3],[0.7,0.7]], cov=0.01, sensor_range=0.5, \
                 fi_weight=0.5, mi_weight=0.5, phi_weight=1.0):

        # TODO: create a message class for this
        # rospy.Subscriber('/target_distribution',  CLASSNAME, self.callback)

        self.num_pts = num_pts
        self.size = size
        self.sensor_range = sensor_range

        grid = np.meshgrid(*[np.linspace(0, self.size, num_pts+1) for _ in range(2)])
        self.raw_grid = [grid[0][0:num_pts,0:num_pts], grid[1][0:num_pts,0:num_pts]]
        self.grid = np.c_[grid[0][0:num_pts,0:num_pts].ravel(), grid[1][0:num_pts,0:num_pts].ravel()]
        # self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]
        self.grid_x = self.grid[:,0]
        self.grid_y = self.grid[:,1]
        self.diff_x = self.grid_x - self.grid_x[:, np.newaxis]
        self.diff_y = self.grid_y - self.grid_y[:, np.newaxis]
        self.dist_xy = np.sqrt(self.diff_x**2 + self.diff_y**2)
        self.dist_flag = (self.dist_xy < self.sensor_range).astype(int)

        self.fi_weight = fi_weight
        self.mi_weight = mi_weight
        self.phi_weight = phi_weight

        # self.means = [npr.uniform(0.2, 0.8, size=(2,))
        #                     for _ in range(num_nodes)]
        self.means = np.array(means)
        self.vars  = [np.array([cov,cov]) for _ in range(len(means))]

        print("means: ", self.means)

        # self.vars  = [npr.uniform(0.05, 0.2, size=(2,))**2
        #                     for _ in range(num_nodes)]

        self.has_update = False
        self.grid_vals = self.__call__(self.grid)
        self.og_vals = np.ones((self.num_pts, self.num_pts)) * 0.5

        range_flag_1 = (0.0 < self.dist_xy).astype(int)
        range_flag_2 = (self.dist_xy < self.sensor_range).astype(int)
        self.range_flag = range_flag_1 * range_flag_2
        self.zero_flag = np.ones((self.num_pts**2, self.num_pts**2)) - np.eye(self.num_pts**2)

    def get_grid_spec(self):
        xy = []
        for g in self.grid.T:
            xy.append(
                np.reshape(g, newshape=(self.num_pts, self.num_pts))
            )
        return xy, self.grid_vals.reshape(self.num_pts, self.num_pts)


    def __call__(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        assert x.shape[1] == 2, 'Does not have right exploration dim'

        val = np.zeros(x.shape[0])
        for m, v in zip(self.means, self.vars):
            innerds = np.sum((x-m)**2 / v, 1)
            val += np.exp(-innerds/2.0)# / np.sqrt((2*np.pi)**2 * np.prod(v))
        # normalizes the distribution
        val /= np.sum(val)
        # val -= np.max(val)
        # val = np.abs(val)
        return val

    def update_og(self, meann, covv):
        mean = meann.copy()
        cov = covv.copy()
        # update og
        dx = 1.0 * self.size / self.num_pts
        clip_idx_x = int(mean[0]/dx)
        if clip_idx_x > self.num_pts-1:
            clip_idx_x = self.num_pts-1
        if clip_idx_x < 0:
            clip_idx_x = 0
        clip_idx_y = int(mean[1]/dx)
        if clip_idx_y > self.num_pts-1:
            clip_idx_y = self.num_pts-1
        if clip_idx_y < 0:
            clip_idx_y = 0
        clip_mean = np.array([self.raw_grid[0][0][clip_idx_x], self.raw_grid[0][0][clip_idx_y]])

        rv = mvn(mean=clip_mean[0:2], cov=cov[0:2, 0:2])
        prob = 0.5 * np.exp( -1. / (rv.pdf(self.grid)+1e-09) ) ** 2.
        addi = np.sum(self.dist_flag * prob[:,np.newaxis], axis=0)
        self.og_vals += addi.reshape(self.num_pts, self.num_pts)
        self.og_vals = np.clip(self.og_vals, a_min=None, a_max=1.)

    def update_dist_val(self, meann, covv, imcov):
        mean = meann.copy()
        cov = covv.copy()

        # fisher information
        xaxis = self.raw_grid[0][0]
        yaxis = self.raw_grid[0][0]
        dx = xaxis[1] - xaxis[0]
        landmark_mean = []
        landmark_cov = []
        num_lm = int((mean.shape[0]-3)/2)
        for i in range(num_lm):
            temp_mean = mean[3+2*i : 5+2*i]
            temp_cov = cov[3+2*i:5+2*i, 3+2*i:5+2*i]
            clip_idx_x = int(temp_mean[0]/dx)
            if clip_idx_x > self.num_pts-1:
                clip_idx_x = self.num_pts-1
            if clip_idx_x < 0:
                clip_idx_x = 0
            clip_idx_y = int(temp_mean[1]/dx)
            if clip_idx_y > self.num_pts:
                clip_idx_y = 49
            if clip_idx_y < 0:
                clip_idx_y = 0
            clip_mean = np.array([xaxis[clip_idx_x], yaxis[clip_idx_y]])
            landmark_mean.append(clip_mean)
            landmark_cov.append(temp_cov)
        landmark_mean = np.array(landmark_mean)
        landmark_cov = np.array(landmark_cov)

        dist_xy = self.dist_xy + 1e-09
        grid_x = self.grid_x.copy()
        grid_y = self.grid_y.copy()
        diff_x = self.diff_x.copy()
        diff_y = self.diff_y.copy()

        dm11 = -diff_x / dist_xy * self.zero_flag
        dm12 = -diff_y / dist_xy * self.zero_flag
        dm21 =  diff_y / dist_xy**2 * self.zero_flag
        dm22 = -diff_x / dist_xy**2 * self.zero_flag

        fim11 = dm11 * (dm11*imcov[0,0] + dm21*imcov[1,0]) + dm21 * (dm11*imcov[0,1] + dm21*imcov[1,1])
        fim12 = dm12 * (dm11*imcov[0,0] + dm21*imcov[1,0]) + dm22 * (dm11*imcov[0,1] + dm21*imcov[1,1])
        fim21 = dm11 * (dm12*imcov[0,0] + dm22*imcov[1,0]) + dm21 * (dm12*imcov[0,1] + dm22*imcov[1,1])
        fim22 = dm12 * (dm12*imcov[0,0] + dm22*imcov[1,0]) + dm22 * (dm12*imcov[0,1] + dm22*imcov[1,1])

        det = fim11 * fim22 - fim12 * fim21
        det = det * self.range_flag

        fim_vals = np.zeros(self.num_pts**2)
        for i in range(landmark_mean.shape[0]):
            distr = mvn.pdf(self.grid, landmark_mean[i], landmark_cov[i] * 1.)
            scaled_det = det * distr[:, np.newaxis]
            det_vals = np.sum(scaled_det, axis=0)
            fim_vals += det_vals
        # fim_vals = fim_vals.reshape(self.num_pts, self.num_pts)
        if np.sum(fim_vals) != 0.:
            fim_vals /= np.sum(fim_vals)

        # mutual information
        mi_vals = np.zeros((self.num_pts, self.num_pts))
        og = self.og_vals.copy()
        full_entropy = np.sum(self.entropy(self.og_vals))
        for i in range(self.num_pts):
            for j in range(self.num_pts):
                if og[i][j] > 0.85:
                    g = np.array([self.raw_grid[0][i][j], self.raw_grid[1][i][j]])
                    dist = (self.raw_grid[0]-g[0])**2 + (self.raw_grid[1]-g[1])**2
                    dist_flag = (dist < self.sensor_range**2).astype(int)
                    obsv_flag = np.ones((self.num_pts, self.num_pts)) - self.og_vals
                    mi_flag = dist_flag * obsv_flag
                    new_og = self.og_vals + mi_flag
                    mi_vals[i][j] = full_entropy - np.sum(self.entropy(new_og))
        if np.sum(mi_vals) != 0:
            mi_vals /= np.sum(mi_vals)
        mi_vals = mi_vals.reshape(-1)

        self.grid_vals = self.phi_weight * (self.fi_weight*fim_vals + self.mi_weight*mi_vals)

        return self.grid_vals.copy()

    def entropy(self, c):
        return -c*np.log(c+1e-9) - (1-c)*np.log(1-c+1e-9)


