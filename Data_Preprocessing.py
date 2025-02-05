

import tensorflow as tf
import scipy.io
import numpy as np
import random



class data_preprocessing():
    def __init__(self,DTYPE):
        ##############load data###############
        hsp = scipy.io.loadmat(r'Input_Data/hsp.mat')
        bsp = scipy.io.loadmat(r'Input_Data/bsp.mat')
        heart_coords = scipy.io.loadmat(r'Input_Data/heart_coordinates')
        t = scipy.io.loadmat(r'Input_Data/time.mat')
        
        norm = scipy.io.loadmat(r'Input_Data/normals.mat')
        self.norm = norm['normals']

        self.t = t['time'].T
        self.hsp_original = hsp['hsp']
        self.bsp_original = bsp['bsp']
        self.DTYPE = DTYPE

        ######## Coordinates #######
        self.heart_coords = heart_coords['heart_coordinates']
        self.t_shape = self.t.shape[0]

    def whole_coordinates(self):
        x = self.heart_coords[:,0]
        y = self.heart_coords[:,1]
        z = self.heart_coords[:,2]
        tt = self.t.repeat(x.shape[0]).flatten()[:,None]
        xx = np.tile(x, self.t_shape).flatten()[:,None]
        yy = np.tile(y, self.t_shape).flatten()[:,None]
        zz = np.tile(z, self.t_shape).flatten()[:,None]

        X_whole = np.hstack((xx, yy, zz, tt))
        # Lower and upper bounds for input
        lb = tf.constant(X_whole.min(0), dtype=self.DTYPE)
        ub = tf.constant(X_whole.max(0), dtype=self.DTYPE)
        return X_whole, lb, ub


    ################# Grab collocation points #############
    def collocation_points(self,tf_shape):
        np.random.seed(121)
        subset_t=np.random.choice(self.t_shape, tf_shape, replace=False)
        x = self.heart_coords[:,0]
        y = self.heart_coords[:,1]
        z = self.heart_coords[:,2]
        t_f=self.t[subset_t]

        tt = t_f.repeat(x.shape[0]).flatten()[:,None]
        xx = np.tile(x, t_f.shape[0]).flatten()[:,None]
        yy = np.tile(y, t_f.shape[0]).flatten()[:,None]
        zz = np.tile(z, t_f.shape[0]).flatten()[:,None]
        x_f = xx.flatten()[:,None]
        y_f = yy.flatten()[:,None]
        z_f = zz.flatten()[:,None]
        tt_f = tt.flatten()[:,None]

        X_f = np.hstack((x_f, y_f, z_f, tt_f))
        X_f = tf.constant(X_f, dtype=self.DTYPE)
     
        xn = self.norm[:,0]
        yn = self.norm[:,1]
        zn = self.norm[:,2]
        
        xxn = np.tile(xn, t_f.shape[0]).flatten()[:,None]
        yyn = np.tile(yn, t_f.shape[0]).flatten()[:,None]
        zzn = np.tile(zn, t_f.shape[0]).flatten()[:,None]
        x_n = xxn.flatten()[:,None]
        y_n = yyn.flatten()[:,None]
        z_n = zzn.flatten()[:,None]
        
        X_n = np.hstack((x_n, y_n, z_n))
        X_n = tf.constant(X_n, dtype=self.DTYPE)
        
        return X_f, X_n
    
        #### True Measurement Data with noise ############
    def sensor_measurements(self, sigma_noise, num_obs_channels):
        
        np.random.seed()
        bsp = self.bsp_original + sigma_noise*np.std(self.bsp_original.flatten())*np.random.randn(self.bsp_original.shape[0], self.bsp_original.shape[1])
        hsp = self.hsp_original + sigma_noise*np.std(self.hsp_original.flatten())*np.random.randn(self.hsp_original.shape[0], self.hsp_original.shape[1])
        
        np.random.seed(121)
        subset = np.random.choice(self.heart_coords.shape[0], num_obs_channels, replace=False)

        hsp_m = hsp[subset,:]
        hsp_m = tf.constant(hsp_m, dtype=self.DTYPE)
        bsp_m = bsp
        bsp_m = tf.constant(bsp_m, dtype=self.DTYPE)

        obs_channels = tf.constant(subset)
        return hsp_m, bsp_m, obs_channels, hsp






