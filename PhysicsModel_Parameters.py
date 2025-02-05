

import scipy.io
import tensorflow as tf
###### Model parameters and Laplacian #######
def Physics_Parameters(DTYPE):
    Delta = scipy.io.loadmat(r'Input_Data/laplacian.mat')
    Delta = Delta['lap']
    a = tf.constant(0.1, dtype=DTYPE)
    c = tf.constant(8, dtype=DTYPE)
    e0 = tf.constant(0.002, dtype=DTYPE)
    D = tf.constant(10, dtype=DTYPE)
    mu1 = tf.constant(0.3, dtype=DTYPE)
    mu2 = tf.constant(0.3, dtype=DTYPE)
    Delta = Delta.todense()
    Delta = tf.constant(Delta,dtype=DTYPE)
    
    Z_BH = scipy.io.loadmat(r'Input_Data/Z_BH.mat')
    Z_BH = Z_BH['Z_BH']
    Z_BH = tf.constant(Z_BH, dtype=DTYPE)
   
    
    return a,c,e0,D,mu1,mu2,Delta,Z_BH




