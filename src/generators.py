import tensorflow as tf
from tensorflow import keras
import numpy as np
import copy

class DataGenerator_ssh_sst_interp(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, val = 'train', datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        X1 = np.empty((self.batch_size, self.dim[0], 64, 64, 1))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))

        Y_length = []
        Y_list = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data_ssh = np.load(self.datadir + 'input_ssh_grid'+ ID + self.val+'.npy')
            ssh = copy.deepcopy(data_ssh)
            ssh[np.isnan(ssh)] = 0
            ssh = ssh[:,:,:,0]
            ssh[ssh!=0] = (ssh[ssh!=0]-mean_ssh)/std_ssh
            
            sst = np.load(self.datadir + 'sst' + ID+ self.val + '.npy')
            sst[np.isnan(sst)] = 0

            sst[sst<273]=0
            sst[sst!=0] = (sst[sst!=0]-mean_sst)/std_sst

            X1[i,:,:,:,0] = ssh[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:]
            X2[i,:,:,:,0] = sst[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:]
            
            ssh_out = np.load(self.datadir + 'output_tracks'+ ID + self.val+'.npy')
            ssh_out = ssh_out[30-int(self.dim[0]/2):30+int(self.dim[0]/2),]

            ssh_out[np.isnan(ssh_out)] = 0
            x = copy.deepcopy(ssh_out[:,:,0])
            x[x!=0] = ((x[x!=0]+0.5*960e3)/960e3)*(128-1)
            y = copy.deepcopy(ssh_out[:,:,1])
            y[y!=0] = ((-y[y!=0]+0.5*960e3)/960e3)*(128-1)
            sla = copy.deepcopy(ssh_out[:,:,2])
            sla[sla!=0] = (sla[sla!=0]-mean_ssh)/std_ssh
            outvar = np.stack((x,y,sla),axis = -1)
            Y_list.append(outvar)
            Y_length.append(outvar.shape[-2])
                        
        Y = np.zeros([self.batch_size, self.dim[0], max(Y_length), 3])
        
        for i in range(self.batch_size):
            Y[i,:,:Y_length[i],:] = Y_list[i]
        # X = [X1, X2, X3]
        X = [X1,X2]
        # X = X1

        return X, Y
    
class DataGenerator_ssh_sst_preprocessed(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, val = 'train', datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        # mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        Y = np.zeros((self.batch_size, self.dim[0], 1000, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            in_ssh = np.load(self.datadir+'input_ssh_grid_high_res'+ID+self.val+'.npy')
            out_ssh = np.load(self.datadir+'output_tracks'+ID+self.val+'.npy')
            sst = np.load(self.datadir+'sst'+ID+self.val+'.npy')
            X1[i,:,:,:,0] = in_ssh[:,:,:,0]
            X2[i,] = sst
            Y[i,:,:out_ssh.shape[1],:] = out_ssh
                        
        
        X = [X1,X2]

        return X, Y
    
class DataGenerator_ssh_sst_preprocessed_UNet(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, val = 'train', datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        # mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128))
        Y = np.zeros((self.batch_size, self.dim[0], 1000, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            in_ssh = np.load(self.datadir+'input_ssh_grid_high_res'+ID+self.val+'.npy')
            out_ssh = np.load(self.datadir+'output_tracks'+ID+self.val+'.npy')
            sst = np.load(self.datadir+'sst'+ID+self.val+'.npy')
            X1[i,:,:,:] = in_ssh[:,:,:,0]
            X2[i,] = sst[:,:,:,0]
            Y[i,:,:out_ssh.shape[1],:] = out_ssh
                        
        
        X = np.stack((X1,X2),axis=-1)

        return X, Y
    
class DataGenerator_ssh_sst_preprocessed_transfer(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, val = 'train', datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        self.datadir = datadir
        self.stats = stats
        self.gulf_stream = (0.0761, 0.176, 290.4, 3.914)
        self.globe = (0.064, 0.0712, 290.1, 1.433)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        # gs_mean_ssh, gs_std_ssh, gs_mean_sst, gs_std_sst = self.gulf_stream
        # glob_mean_ssh, glob_std_ssh, glob_mean_sst, glob_std_sst = self.globe
        
        X1 = np.empty((self.batch_size, self.dim[0], 64, 64, 1))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        Y = np.zeros((self.batch_size, self.dim[0], 700, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            in_ssh = np.load(self.datadir+'input_ssh_grid'+ID+self.val+'.npy')
            # in_ssh[in_ssh!=0] = ((in_ssh[in_ssh!=0]*gs_std_ssh+gs_mean_ssh)-glob_mean_ssh)/glob_std_ssh
            out_ssh = np.load(self.datadir+'output_tracks'+ID+self.val+'.npy')
            # x = out_ssh[:,:,0].copy()
            # y = out_ssh[:,:,1].copy()
            # sla= out_ssh[:,:,2].copy()
            # sla[sla!=0] = ((sla[sla!=0]*gs_std_ssh+gs_mean_ssh)-glob_mean_ssh)/glob_std_ssh
            # out_ssh = np.stack((x,y,sla),axis=-1)
            sst = np.load(self.datadir+'sst'+ID+self.val+'.npy')
            # sst[sst!=0] = ((sst[sst!=0]*gs_std_sst+gs_mean_sst)-glob_mean_sst)/glob_std_sst
            X1[i,] = in_ssh
            X2[i,] = sst
            Y[i,] = out_ssh
                             
        X = [X1,X2]

        return X, Y
    
class DataGenerator_ssh_interp(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, val = 'train', datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        # Initialization
        X1 = np.empty((self.batch_size, self.dim[0], 64, 64, 1))
        # X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        # X3 = np.empty((self.batch_size, 30, 128, 128, 1))
        # Y = np.empty((self.batch_size, *self.dim, 1))
        Y_length = []
        Y_list = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data_ssh = np.load(self.datadir + 'input_ssh_grid'+ ID + self.val+'.npy')
            
            ssh = copy.deepcopy(data_ssh)
            ssh[np.isnan(ssh)] = 0
            ssh = ssh[:,:,:,0]
            ssh[ssh!=0] = (ssh[ssh!=0]-mean_ssh)/std_ssh
            
            X1[i,:,:,:,0] = ssh[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:]
            
            ssh_out = np.load(self.datadir + 'output_tracks'+ ID + self.val+'.npy')

            ssh_out = ssh_out[30-int(self.dim[0]/2):30+int(self.dim[0]/2),]
            
            ssh_out[np.isnan(ssh_out)] = 0
            x = copy.deepcopy(ssh_out[:,:,0])
            x[x!=0] = ((x[x!=0]+0.5*960e3)/960e3)*(128-1)
            y = copy.deepcopy(ssh_out[:,:,1])
            y[y!=0] = ((-y[y!=0]+0.5*960e3)/960e3)*(128-1)
            sla = copy.deepcopy(ssh_out[:,:,2])
            sla[sla!=0] = (sla[sla!=0]-mean_ssh)/std_ssh
            outvar = np.stack((x,y,sla),axis = -1)
            Y_list.append(outvar)
            Y_length.append(outvar.shape[-2])

        Y = np.zeros([self.batch_size, self.dim[0], max(Y_length), 3])
        
        for i in range(self.batch_size):
            Y[i,:,:Y_length[i],:] = Y_list[i]

        X = X1

        return X, Y
    
class DataGenerator_test_ssh_sst(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        # Initialization
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        Y = np.empty((self.batch_size, *self.dim, 1))
        # Y_length = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data_ssh = np.load(self.datadir + 'input_ssh_grid_high_res'+ ID + '.npy')
            # ssh = copy.deepcopy(data_ssh)
            # ssh[np.isnan(ssh)] = 0
            # ssh = ssh[:,:,:,0]
            # ssh[ssh!=0] = (ssh[ssh!=0]-mean_ssh)/std_ssh
            sst = np.load(self.datadir + 'sst' + ID + '.npy')
            sst[np.isnan(sst)] = 0
            sst[sst<273]=0

            sst[sst!=0] = (sst[sst!=0]-mean_sst)/std_sst
            X1[i,:,:,:,0] = data_ssh[:,:,:,0]#ssh[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:]
            X2[i,:,:,:,0] = sst[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:]
            
            ssh_out = np.load(self.datadir + 'output_ssh_grid'+ ID + '.npy')

            ssh_out[np.isnan(ssh_out)] = 0
            ssh_out= ssh_out[:,:,:,0]
            ssh_out[ssh_out!=0] = (ssh_out[ssh_out!=0]-mean_ssh)/std_ssh

            y = np.reshape(ssh_out, (ssh_out.shape[0], ssh_out.shape[1], ssh_out.shape[2], 1))
            Y[i,] = y[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:,:]

        X = [X1,X2]

        return X, Y

class DataGenerator_test_ssh_sst_UNet(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        # Initialization
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128))
        Y = np.empty((self.batch_size, *self.dim, 1))
        # Y_length = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data_ssh = np.load(self.datadir + 'input_ssh_grid_high_res'+ ID + '.npy')
            # ssh = copy.deepcopy(data_ssh)
            # ssh[np.isnan(ssh)] = 0
            # ssh = ssh[:,:,:,0]
            # ssh[ssh!=0] = (ssh[ssh!=0]-mean_ssh)/std_ssh
            sst = np.load(self.datadir + 'sst' + ID + '.npy')
            sst[np.isnan(sst)] = 0
            sst[sst<273]=0

            sst[sst!=0] = (sst[sst!=0]-mean_sst)/std_sst
            X1[i,:,:,:] = data_ssh[:,:,:,0]#ssh[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:]
            X2[i,:,:,:] = sst[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:]
            
            ssh_out = np.load(self.datadir + 'output_ssh_grid'+ ID + '.npy')

            ssh_out[np.isnan(ssh_out)] = 0
            ssh_out= ssh_out[:,:,:,0]
            ssh_out[ssh_out!=0] = (ssh_out[ssh_out!=0]-mean_ssh)/std_ssh

            y = np.reshape(ssh_out, (ssh_out.shape[0], ssh_out.shape[1], ssh_out.shape[2], 1))
            Y[i,] = y[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:,:]

        X = np.stack((X1,X2),axis=-1)

        return X, Y
    
class DataGenerator_test_ssh(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        # Initialization
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        # X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        Y = np.empty((self.batch_size, *self.dim, 1))
        # Y_length = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data_ssh = np.load(self.datadir + 'input_ssh_grid_high_res'+ ID + '.npy')
            # ssh = copy.deepcopy(data_ssh)
            # ssh[np.isnan(ssh)] = 0
            # ssh = ssh[:,:,:,0]
            # ssh[ssh!=0] = (ssh[ssh!=0]-mean_ssh)/std_ssh
#             sst = np.load(self.datadir + 'sst' + ID + '.npy')
#             sst[np.isnan(sst)] = 0
#             sst[sst<273]=0

#             sst[sst!=0] = (sst[sst!=0]-mean_sst)/std_sst
            X1[i,:,:,:,0] = data_ssh[:,:,:,0]#ssh[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:]
            # X2[i,:,:,:,0] = sst[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:]
            
            ssh_out = np.load(self.datadir + 'output_ssh_grid'+ ID + '.npy')

            ssh_out[np.isnan(ssh_out)] = 0
            ssh_out= ssh_out[:,:,:,0]
            ssh_out[ssh_out!=0] = (ssh_out[ssh_out!=0]-mean_ssh)/std_ssh

            y = np.reshape(ssh_out, (ssh_out.shape[0], ssh_out.shape[1], ssh_out.shape[2], 1))
            Y[i,] = y[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:,:]

        X = X1

        return X, Y
    
class DataGenerator_test_ssh_high_res(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        # Initialization
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        Y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data_ssh = np.load(self.datadir + 'input_ssh_grid_high_res'+ ID + '.npy')
            # ssh = copy.deepcopy(data_ssh)
            # ssh[np.isnan(ssh)] = 0
            # ssh = ssh[:,:,:,0]
            # ssh[ssh!=0] = (ssh[ssh!=0]-mean_ssh)/std_ssh

            X1[i,:,:,:,0] = data_ssh[:,:,:,0]

            ssh_out = np.load(self.datadir + 'output_ssh_grid'+ ID + '.npy')

            ssh_out[np.isnan(ssh_out)] = 0
            ssh_out= ssh_out[:,:,:,0]
            ssh_out[ssh_out!=0] = (ssh_out[ssh_out!=0]-mean_ssh)/std_ssh

            y = np.reshape(ssh_out, (ssh_out.shape[0], ssh_out.shape[1], ssh_out.shape[2], 1))
            Y[i,] = y[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:,:]

        X = X1

        return X, Y 
    
    
    
class DataGenerator_ssh_sst_varying_nt_paper(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, val = 'train', datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        # mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        Y = np.zeros((self.batch_size, self.dim[0], 1000, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            in_ssh = np.load(self.datadir+'input_ssh_grid_high_res_processed'+ID+self.val+'.npy')
            out_ssh = np.load(self.datadir+'output_tracks_processed'+ID+self.val+'.npy')
            sst = np.load(self.datadir+'sst_processed'+ID+self.val+'.npy')
            X1[i,:,:,:,0] = in_ssh[:self.dim[0],:,:,0]
            X2[i,] = sst[:self.dim[0],]
            Y[i,] = out_ssh[:self.dim[0],]
                        
        
        X = [X1,X2]

        return X, Y
    
class DataGenerator_test_ssh_sst_varying_nt_paper(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        Y = np.zeros((self.batch_size, self.dim[0], 1000, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            in_ssh = np.load(self.datadir+'input_ssh_grid_high_res_processed'+ID+'.npy')
            out_ssh = np.load(self.datadir+'output_tracks_processed'+ID+'.npy')
            sst = np.load(self.datadir+'sst_processed'+ID+'.npy')
            X1[i,:,:,:,0] = in_ssh[30-int(self.dim[0]/2):30+int(self.dim[0]/2),:,:,0]
            X2[i,] = sst[30-int(self.dim[0]/2):30+int(self.dim[0]/2),]
            Y[i,] = out_ssh[30-int(self.dim[0]/2):30+int(self.dim[0]/2),]
                        
        
        X = [X1,X2]

        return X, Y
    
    
class DataGenerator_ssh_sst_varying_nt_paper_short_t(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, val = 'train', datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        # mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        Y = np.zeros((self.batch_size, self.dim[0], 1000, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            in_ssh = np.load(self.datadir+'input_ssh_grid_high_res_processed'+ID+self.val+'.npy')
            out_ssh = np.load(self.datadir+'output_tracks_processed'+ID+self.val+'.npy')
            sst = np.load(self.datadir+'sst_processed'+ID+self.val+'.npy')
            start_idx = np.random.randint(0,60-self.dim[0]-1)
            X1[i,:,:,:,0] = in_ssh[start_idx:start_idx+self.dim[0],:,:,0]
            X2[i,] = sst[start_idx:start_idx+self.dim[0],]
            Y[i,] = out_ssh[start_idx:start_idx+self.dim[0],]
                        
        
        X = [X1,X2]

        return X, Y
    
class DataGenerator_ssh_preprocessed(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, val = 'train', datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        # mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        # X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        Y = np.zeros((self.batch_size, self.dim[0], 1000, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            in_ssh = np.load(self.datadir+'input_ssh_grid_high_res'+ID+self.val+'.npy')
            out_ssh = np.load(self.datadir+'output_tracks'+ID+self.val+'.npy')
            # sst = np.load(self.datadir+'sst'+ID+self.val+'.npy')
            X1[i,:,:,:,0] = in_ssh[:,:,:,0]
            # X2[i,] = sst
            Y[i,:,:out_ssh.shape[1],:] = out_ssh
                        
        
        X = X1

        return X, Y