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
        
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
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

            X1[i,:,:,:,0] = ssh[:self.dim[0],:,:]
            X2[i,:,:,:,0] = sst[:self.dim[0],:,:]
            
            ssh_out = np.load(self.datadir + 'output_tracks'+ ID + self.val+'.npy')
            ssh_out = ssh_out[:self.dim[0],]

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
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
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
            
            X1[i,:,:,:,0] = ssh[:self.dim[0],:,:]
            
            ssh_out = np.load(self.datadir + 'output_tracks'+ ID + self.val+'.npy')

            ssh_out = ssh_out[:self.dim[0],]
            
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
    
