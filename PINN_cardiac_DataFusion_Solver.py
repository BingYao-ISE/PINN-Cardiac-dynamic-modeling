

import tensorflow as tf
from time import time
import numpy as np

class PINN_Net(tf.keras.Model):
    """ Set architecture of the PINN model."""

    def __init__(self, lb, ub, DTYPE, adaptive_weight=True, output_dim=2,
            num_hidden_layers=5, 
            num_neurons_per_layer=10, **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lb = lb
        self.ub = ub
        self.DTYPE = DTYPE
        kernel_initializer =  tf.keras.initializers.RandomUniform(-1/2,1/2,seed = 123456)
        # Trainable parameter in self-designed activation function
        self.act_par = tf.Variable([0.1], trainable=True, dtype=self.DTYPE)
        
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        # set the neural network model
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=self.self_activation,
                             kernel_initializer=kernel_initializer,
                                            bias_initializer='zeros')
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim,  kernel_initializer=kernel_initializer)    
        
        # Initialize the uncertainty of each learning task: weights for loss function
        self.sig1 = tf.Variable(1.0,trainable=adaptive_weight, dtype=self.DTYPE)
        self.sig2 = tf.Variable(1.0,trainable=adaptive_weight, dtype=self.DTYPE)
        self.sig3 = tf.Variable(1.0,trainable=adaptive_weight, dtype=self.DTYPE)
    
    def self_activation(self,x):    
        act=tf.tanh(5.0*self.act_par[0]*x) 
        return act
    
    def call(self, X):
        """Forward-pass through neural network."""
        Z = self.scale(X)
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)


class PINN_Solver():
    def __init__(self, model, X_f, Delta, tf_shape, a, c, e0, D, mu1, mu2, Z_BH):
        self.model = model
        
        # Store collocation points
        self.x = X_f[:,0]
        self.y = X_f[:,1]
        self.z = X_f[:,2]
        self.t = X_f[:,3]
 
        # Physics model parameters
        self.Delta = Delta
        self.tf_shape = tf_shape
        self.a = a
        self.c = c
        self.e0 = e0
        self.D = D
        self.mu1 = mu1
        self.mu2 = mu2
        self.Z_BH = Z_BH
        
        # Initialize history of losses and iteration counter
        self.hist_loss = []
        self.hist_loss_phys = []
        self.hist_loss_d = []
        self.hist_loss_hb = []
   
        self.iter = 0
        

    def get_residual(self):
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t during this GradientTape
            tape.watch(self.t)
            
            # Compute current values u(t,x), v(t,x)            
            pred = self.model(tf.stack([self.x,self.y,self.z, self.t], axis=1))           
            u = pred[:,0]
            v = pred[:,1]
       
        # Compute the derivative with respect to t
        u_t = tape.gradient(u, self.t)       
        v_t = tape.gradient(v, self.t)
        
        del tape
        
        # Reshape the u, v, and their derivatives
        u = tf.transpose(tf.reshape(u,[self.tf_shape,-1]))
        v = tf.transpose(tf.reshape(v,[self.tf_shape,-1]))
        u_t = tf.transpose(tf.reshape(u_t,[self.tf_shape,-1]))
        v_t = tf.transpose(tf.reshape(v_t,[self.tf_shape,-1]))
       
        # Calculate residuals
        f_u = u_t - self.c*u*(u - self.a)*(1 - u) + u*v -self.D*tf.matmul(self.Delta,u) 
        f_v = v_t - (self.e0 + self.mu1*v/(u+self.mu2)) * (-v - self.c*u*(u-self.a-1))
        f_u = tf.reshape(f_u,[-1])
        f_v = tf.reshape(f_v,[-1])
        
        return f_u, f_v
        
    def loss_ftn(self, X, obs_channels, hsp_m, bsp_m, eps = 1.0):
        
        # Compute residuals
        r_u, r_v = self.get_residual()
        
        phi_ru = tf.reduce_mean(tf.square(r_u))
        phi_rv = tf.reduce_mean(tf.square(r_v))
        
        # Compute physics loss
        loss_phys = phi_ru+phi_rv
              
        # Compute data-driven loss 
        preds = self.model(tf.stack([X[:,0],X[:,1],X[:,2], X[:,3]], axis=1))        
        u_pred= preds[:,0]
        u_pred = tf.transpose(tf.reshape(u_pred,[bsp_m.shape[1],-1]))
        
        # Compute data-driven loss for hsp measurements         
        loss_d= tf.reduce_mean(tf.square(tf.gather(u_pred,obs_channels) - hsp_m))
        
        # Transfer loss from hsp to bsp     
        loss_hb= tf.reduce_mean(tf.square(tf.matmul(self.Z_BH,u_pred) - bsp_m))
        
        # Compute the total loss weighted by the model uncertainty
        loss=loss_phys/tf.square(self.model.sig1)+tf.math.log(tf.square(self.model.sig1)+eps)+\
                loss_d/tf.square(self.model.sig2)+tf.math.log(tf.square(self.model.sig2)+eps)+\
                loss_hb/tf.square(self.model.sig3)+tf.math.log(tf.square(self.model.sig3)+eps)
        
        return loss, loss_phys, loss_d, loss_hb
    
    def get_grad(self, X, obs_channels, hsp_m, bsp_m):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss, loss_phys, loss_d, loss_hb = self.loss_ftn(X, obs_channels, hsp_m, bsp_m)
        
        g = tape.gradient(loss, self.model.trainable_variables)
        
        del tape
        
        return loss, loss_phys, loss_d, loss_hb, g
 
    
    def train_loop(self, X_whole, obs_channels, hsp_m, bsp_m, hsp, optimizer, N_iter):
        """Network training."""
        @tf.function
        def train_step(optimizer):
            loss, loss_f, loss_d, loss_hb, grad_theta = self.get_grad(X_whole, obs_channels, hsp_m, bsp_m)
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss, loss_f, loss_d, loss_hb,
        
        self.t0 = time()
        for i in range(N_iter):
            
            loss,loss_phys, loss_d, loss_hb = train_step(optimizer)

            self.current_loss = loss.numpy()
            self.current_loss_phys = loss_phys.numpy()
            self.current_loss_d = loss_d.numpy()
            self.current_loss_hb = loss_hb.numpy()

            self.callback(X_whole, hsp)
       
    def callback(self, X_whole, hsp):
        if self.iter % 100 == 0:
            print('It {:05d}: loss={:10.4e}, l_phys={:10.3e}, l_d={:10.3e}, l_hb={:10.3e}, t= {:.2f}s'.format(self.iter,                self.current_loss,self.current_loss_phys,self.current_loss_d,self.current_loss_hb,time()-self.t0))
          
            self.t0 = time()
     
        self.hist_loss.append(self.current_loss)
        self.hist_loss_phys.append(self.current_loss_phys)
        self.hist_loss_d.append(self.current_loss_d)
        self.hist_loss_hb.append(self.current_loss_hb)
        self.iter+=1
        









