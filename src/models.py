import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

def create_ConvLSTM_SLA(n_t):
    
    def down_block(x,n_filters):
        y = keras.layers.Conv3D(filters = n_filters, kernel_size = (1,4,4), strides = (1,2,2), activation = 'relu', padding = 'same')(x)
        y = keras.layers.BatchNormalization()(y) 
        return y
    
    def res_block(x,filters_in,filters_out):
        if filters_in==filters_out:
            skip = x
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), activation = 'relu', padding = 'same')(x)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), padding = 'same')(y)
            y = y + skip
            y = keras.layers.Activation('relu')(y)
        else:
            skip = x
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), activation = 'relu', padding = 'same')(x)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), padding = 'same')(y)
            skip = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,1,1), padding = 'same')(skip)
            y = y + skip
            y = keras.layers.Activation('relu')(y)
        return y
        
    
    sla = keras.layers.Input(shape=(n_t,128,128,1))
    
    y = down_block(sla,16)

    y = res_block(y,16,16)
    
    y = down_block(y,32)
    
    y = res_block(y,32,32)
    
    y = down_block(y,32)
    
    y = res_block(y,32,32)
    
    encoder = keras.models.Model(inputs=sla, outputs=y)
    
    y = keras.layers.Bidirectional(keras.layers.ConvLSTM2D(filters = 16, kernel_size = (4,4), padding='same', return_sequences=True))(encoder.output)
    
    y = res_block(y,32,32)
    
    y = keras.layers.UpSampling3D(size = (1,2,2))(y)
    
    y = res_block(y,32,16)
    
    y = keras.layers.UpSampling3D(size = (1,2,2))(y)
    
    y = res_block(y,16,8)
        
    y = keras.layers.UpSampling3D(size = (1,2,2))(y)
    
    y = keras.layers.Conv3D(filters = 8, kernel_size = (1,4,4), padding = 'same', activation = 'relu')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv3D(filters = 1, kernel_size = (1,1,1), padding = 'same', activation = 'linear')(y)

    model = keras.models.Model(inputs=sla, outputs=y)
    
    return model

def create_ConvLSTM_SLA_SST(n_t):
    
    ssh_names_used = ['ssh0']
    sst_names_used = ['sst0']
    combined_names_used = ['com0']
    
    def down_block(x,n_filters,variable='combined'):
        if variable=='ssh':
            names_used = ssh_names_used.copy()
        elif variable == 'sst':
            names_used = sst_names_used.copy()
        else:
            names_used = combined_names_used.copy()
        name_next = names_used[0][:3]+f'{len(names_used)}'
        names_used.append(name_next)
        y = keras.layers.Conv3D(filters = n_filters, kernel_size = (1,4,4), strides = (1,2,2), activation = 'relu', padding = 'same',name=name_next)(x)
        name_next = names_used[0][:3]+f'{len(names_used)}'
        names_used.append(name_next)
        y = keras.layers.BatchNormalization(name=name_next)(y) 
        return y, names_used
    
    def res_block(x,filters_in,filters_out, variable = 'combined'):
        if variable=='ssh':
            names_used = ssh_names_used.copy()
        elif variable == 'sst':
            names_used = sst_names_used.copy()
        else:
            names_used = combined_names_used.copy() 
        if filters_in==filters_out:
            skip = x
            name_next = names_used[0][:3]+f'{len(names_used)}'
            names_used.append(name_next)
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), activation = 'relu', padding = 'same', name = name_next)(x)
            name_next = names_used[0][:3]+f'{len(names_used)}'
            names_used.append(name_next)
            y = keras.layers.BatchNormalization(name = name_next)(y)
            name_next = names_used[0][:3]+f'{len(names_used)}'
            names_used.append(name_next)
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), padding = 'same', name = name_next)(y)
            name_next = names_used[0][:3]+f'{len(names_used)}'
            names_used.append(name_next)
            y = y + skip
            y = keras.layers.Activation('relu', name = name_next)(y)
        else:
            skip = x
            name_next = names_used[0][:3]+f'{len(names_used)}'
            names_used.append(name_next)
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), activation = 'relu', padding = 'same', name = name_next)(x)
            name_next = names_used[0][:3]+f'{len(names_used)}'
            names_used.append(name_next)
            y = keras.layers.BatchNormalization(name = name_next)(y)
            name_next = names_used[0][:3]+f'{len(names_used)}'
            names_used.append(name_next)
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), padding = 'same', name = name_next)(y)
            name_next = names_used[0][:3]+f'{len(names_used)}'
            names_used.append(name_next)
            skip = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,1,1), padding = 'same', name = name_next)(skip)
            name_next = names_used[0][:3]+f'{len(names_used)}'
            names_used.append(name_next)
            y = y + skip
            y = keras.layers.Activation('relu', name = name_next)(y)
        return y, names_used
    
    
    sla = keras.layers.Input(shape=(n_t,128,128,1))
    sst = keras.layers.Input(shape=(n_t,128,128,1))
    
    y, ssh_names_used = down_block(sla,16,variable='ssh')

    y, ssh_names_used = res_block(y,16,16,variable='ssh')
    
    y, ssh_names_used = down_block(y,32, variable = 'ssh')
    
    y, ssh_names_used = res_block(y,32,32, variable = 'ssh')
    
    y, ssh_names_used = down_block(y,32, variable = 'ssh')
    
    y, ssh_names_used = res_block(y,32,32, variable = 'ssh')
    
    name_next = ssh_names_used[0][:3]+f'{len(ssh_names_used)}'
    ssh_names_used.append(name_next)
    y = keras.layers.Bidirectional(keras.layers.ConvLSTM2D(filters = 16, kernel_size = (4,4), padding='same', return_sequences=True, name = name_next))(y)
    
    encoder = keras.models.Model(inputs=sla, outputs=y)
    sla_encoder = keras.models.Model(inputs=sla, outputs=y)
    
    y2, sst_names_used = down_block(sst,16,variable='sst')
    
    y2, sst_names_used = res_block(y2,16,16, variable = 'sst')
    
    y2, sst_names_used = down_block(y2,16,variable='sst')
    
    y2, sst_names_used = res_block(y2,16,32,variable='sst')
    
    y2, sst_names_used = down_block(y2,32,variable='sst')
    
    y2, sst_names_used = res_block(y2,32,32,variable='sst')
    
    name_next = sst_names_used[0][:3]+f'{len(sst_names_used)}'
    sst_names_used.append(name_next)
    y2 = keras.layers.Bidirectional(keras.layers.ConvLSTM2D(filters = 16, kernel_size = (4,4), padding='same', return_sequences=True, name = name_next))(y2)
    sst_encoder = keras.models.Model(inputs=sst, outputs=y2)
    
    # combine the output of the two branches
    combined = keras.layers.concatenate([sla_encoder.output, sst_encoder.output])
    
    y = keras.layers.Bidirectional(keras.layers.ConvLSTM2D(filters = 16, kernel_size = (4,4), padding='same', return_sequences=True))(combined)
    y = keras.layers.BatchNormalization()(y)
    y, combined_names_used = res_block(y,32,32)
    
    y = keras.layers.UpSampling3D(size = (1,2,2))(y)
    y, combined_names_used = res_block(y,32,16)
    
    y = keras.layers.UpSampling3D(size = (1,2,2))(y)
    y, combined_names_used = res_block(y,16,8)
    
    y = keras.layers.UpSampling3D(size = (1,2,2))(y)
    y = keras.layers.Conv3D(filters = 8, kernel_size = (1,4,4), padding = 'same', activation = 'relu')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv3D(filters = 1, kernel_size = (1,1,1), padding = 'same', activation = 'linear')(y)

    model = keras.models.Model(inputs=[sla,sst], outputs=y)
    
    return model
