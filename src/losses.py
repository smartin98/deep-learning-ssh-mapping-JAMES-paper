import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

def tracked_mse_interp_grads_gulf_stream():
    # Gulf Stream
    std1 = 0.12818
    std2 = 0.14088
    
    lambda1 = 0.05
    lambda2 = 0.05
    def tracked_mse_internal_interp_grads(y_true, y_pred):
        # losses = []
        for i in range(y_pred.shape[1]):
            data = y_pred[:,i,]
            data = tf.stack([data],axis=-1)
            warp = y_true[:,i,:,:-1]
            y_pred_loss = tfa.image.resampler(data, warp)
            y_true_loss = tf.stack([y_true[:,i,:,-1]], axis = -1)
            y_pred_loss = y_pred_loss*tf.cast((y_true_loss!=0), dtype='float32')

            dx =(tf.roll(warp[:,:,0], shift = 1, axis = -1) - tf.roll(warp[:,:,0], shift = -1, axis = -1))/2
            dy =(tf.roll(warp[:,:,1], shift = 1, axis = -1) - tf.roll(warp[:,:,1], shift = -1, axis = -1))/2
            dl = (dx**2+dy**2)**0.5

            dl = tf.stack([dl],axis=-1)
            dy_pred = (tf.roll(y_pred_loss, shift = 1, axis = -2) - tf.roll(y_pred_loss, shift = -1, axis = -2))/(2*dl+keras.backend.epsilon())
            dy_true = (tf.roll(y_true_loss, shift = 1, axis = -2) - tf.roll(y_true_loss, shift = -1, axis = -2))/(2*dl+keras.backend.epsilon())
            dy_pred = dy_pred*tf.cast((y_true_loss!=0), dtype='float32')
            dy_true =dy_true*tf.cast((y_true_loss!=0), dtype='float32')
            
            d2y_pred = (tf.roll(y_pred_loss, shift = 1, axis = -2) - 2*y_pred_loss + tf.roll(y_pred_loss, shift = -1, axis = -2))/(dl**2+keras.backend.epsilon())
            d2y_true = (tf.roll(y_true_loss, shift = 1, axis = -2) - 2*y_true_loss + tf.roll(y_true_loss, shift = -1, axis = -2))/(dl**2+keras.backend.epsilon())
            d2y_pred = d2y_pred*tf.cast((y_true_loss!=0), dtype='float32')
            d2y_true =d2y_true*tf.cast((y_true_loss!=0), dtype='float32')
            
            N_nz = tf.math.reduce_sum(tf.cast((y_true_loss!=0), dtype='float32'))
            N = tf.math.reduce_sum(tf.cast((y_true_loss!=0), dtype='float32'))+tf.math.reduce_sum(tf.cast((y_true_loss==0), dtype='float32'))
            
            loss_d1 = tf.clip_by_value((N/(N_nz+keras.backend.epsilon()))*keras.losses.MSE(dy_true, dy_pred),0,2)
            loss_d2 = tf.clip_by_value((N/(N_nz+keras.backend.epsilon()))*keras.losses.MSE(d2y_true, d2y_pred),0,2)
            loss_ssh = (N/(N_nz+keras.backend.epsilon()))*keras.losses.MSE(y_true_loss, y_pred_loss)
            
            loss_loop = loss_ssh + (lambda1/std1**2)*loss_d1 + (lambda2/std2**2)*loss_d2
        
            if i ==0:
                loss = loss_loop
            else:
                loss = tf.concat((loss,loss_loop),axis=0)
        return tf.reduce_mean(loss)
    return tracked_mse_internal_interp_grads

def tracked_mse_interp_grads_global():
    # Global
    std1 = 0.14309
    std2 = 0.18899
    
    lambda1 = 0.05
    lambda2 = 0.05
    def tracked_mse_internal_interp_grads(y_true, y_pred):
        # losses = []
        for i in range(y_pred.shape[1]):
            data = y_pred[:,i,]
            data = tf.stack([data],axis=-1)
            warp = y_true[:,i,:,:-1]
            y_pred_loss = tfa.image.resampler(data, warp)
            y_true_loss = tf.stack([y_true[:,i,:,-1]], axis = -1)
            y_pred_loss = y_pred_loss*tf.cast((y_true_loss!=0), dtype='float32')

            dx =(tf.roll(warp[:,:,0], shift = 1, axis = -1) - tf.roll(warp[:,:,0], shift = -1, axis = -1))/2
            dy =(tf.roll(warp[:,:,1], shift = 1, axis = -1) - tf.roll(warp[:,:,1], shift = -1, axis = -1))/2
            dl = (dx**2+dy**2)**0.5

            dl = tf.stack([dl],axis=-1)
            dy_pred = (tf.roll(y_pred_loss, shift = 1, axis = -2) - tf.roll(y_pred_loss, shift = -1, axis = -2))/(2*dl+keras.backend.epsilon())
            dy_true = (tf.roll(y_true_loss, shift = 1, axis = -2) - tf.roll(y_true_loss, shift = -1, axis = -2))/(2*dl+keras.backend.epsilon())
            dy_pred = dy_pred*tf.cast((y_true_loss!=0), dtype='float32')
            dy_true =dy_true*tf.cast((y_true_loss!=0), dtype='float32')
            
            d2y_pred = (tf.roll(y_pred_loss, shift = 1, axis = -2) - 2*y_pred_loss + tf.roll(y_pred_loss, shift = -1, axis = -2))/(dl**2+keras.backend.epsilon())
            d2y_true = (tf.roll(y_true_loss, shift = 1, axis = -2) - 2*y_true_loss + tf.roll(y_true_loss, shift = -1, axis = -2))/(dl**2+keras.backend.epsilon())
            d2y_pred = d2y_pred*tf.cast((y_true_loss!=0), dtype='float32')
            d2y_true =d2y_true*tf.cast((y_true_loss!=0), dtype='float32')
            
            N_nz = tf.math.reduce_sum(tf.cast((y_true_loss!=0), dtype='float32'))
            N = tf.math.reduce_sum(tf.cast((y_true_loss!=0), dtype='float32'))+tf.math.reduce_sum(tf.cast((y_true_loss==0), dtype='float32'))
            
            loss_d1 = tf.clip_by_value((N/(N_nz+keras.backend.epsilon()))*keras.losses.MSE(dy_true, dy_pred),0,2)
            loss_d2 = tf.clip_by_value((N/(N_nz+keras.backend.epsilon()))*keras.losses.MSE(d2y_true, d2y_pred),0,2)
            loss_ssh = (N/(N_nz+keras.backend.epsilon()))*keras.losses.MSE(y_true_loss, y_pred_loss)
            
            loss_loop = loss_ssh + (lambda1/std1**2)*loss_d1 + (lambda2/std2**2)*loss_d2
        
            if i ==0:
                loss = loss_loop
            else:
                loss = tf.concat((loss,loss_loop),axis=0)
        return tf.reduce_mean(loss)
    return tracked_mse_internal_interp_grads