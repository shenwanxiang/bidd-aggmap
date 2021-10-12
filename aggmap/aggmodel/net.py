import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.layers import MaxPool2D, GlobalMaxPool2D, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Concatenate,Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2, l1

#from tensorflow_addons.layers import  GroupNormalization

def count_trainable_params(model):
    p = 0
    for layer in model.layers:
        if len(layer.trainable_variables) == 0:
            continue
        else:
            for variables in layer.trainable_variables:
                a = variables.shape.as_list()
                if len(a) == 1:
                    p += a[0]
                else:
                    p += a[0]*a[1]
    return p


def resnet_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x


def Inception(inputs, units = 8, strides = 1):
    """
    naive google inception block
    """
    x1 = Conv2D(units, 5, padding='same', activation = 'relu', strides = strides)(inputs)
    x2 = Conv2D(units, 3, padding='same', activation = 'relu', strides = strides)(inputs)
    x3 = Conv2D(units, 1, padding='same', activation = 'relu', strides = strides)(inputs)
    outputs = Concatenate()([x1, x2, x3])    
    return outputs


def _AggMapNet(input_shape,  
               n_outputs = 1, 
               conv1_kernel_size = 11,
               batch_norm = False,
               n_inception = 2,
               dense_layers = [128, 32], 
               dense_avf = 'relu', 
               dropout = 0,
               last_avf = None):

    """
    parameters
    ----------------------
    molmap_shape: w, h, c
    n_outputs: output units
    n_inception: number of the inception layers
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    dropout: dropout of the dense layers
    """
    tf.keras.backend.clear_session()
    assert len(input_shape) == 3
    inputs = Input(input_shape)
    
    if batch_norm:
        conv1 = Conv2D(48,  conv1_kernel_size, padding = 'same', strides = 1)(inputs) #activation='relu',  use_bias=False, kernel_regularizer=l1(1e-3)     
        conv1 = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.8)(conv1)
        incept = Activation("relu")(conv1)    
    else:
        incept = Conv2D(48,  conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(inputs)
    
    for i in range(n_inception):
        incept = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(incept) #p1
        incept = Inception(incept, strides = 1, units = 32*(2**i))

    #flatten
    x = GlobalMaxPool2D()(incept)
    
    ## dense layer
    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)
        if dropout:
            x = Dropout(rate = dropout)(x)

    #last layer
    outputs = Dense(n_outputs,activation=last_avf)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def _AggMapDualPathNet(molmap1_size, 
                    molmap2_size, 
                    n_outputs = 1,
                    conv1_kernel_size = 11,
                    dense_layers = [256, 128, 32], 
                    dense_avf = 'relu', 
                    last_avf = None):
    """
    parameters
    ----------------------
    molmap1_size: w, h, c, shape of first molmap
    molmap2_size: w, h, c, shape of second molmap
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    tf.keras.backend.clear_session()
    ## first inputs
    d_inputs1 = Input(molmap1_size)
    d_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(d_inputs1)
    d_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_conv1) #p1
    d_incept1 = Inception(d_pool1, strides = 1, units = 32)
    d_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_incept1) #p2
    d_incept2 = Inception(d_pool2, strides = 1, units = 64)
    d_flat1 = GlobalMaxPool2D()(d_incept2)

    
    ## second inputs
    f_inputs1 = Input(molmap2_size)
    f_conv1 = Conv2D(48, conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(f_inputs1)
    f_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_conv1) #p1
    f_incept1 = Inception(f_pool1, strides = 1, units = 32)
    f_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_incept1) #p2
    f_incept2 = Inception(f_pool2, strides = 1, units = 64)
    f_flat1 = GlobalMaxPool2D()(f_incept2)    
    
    ## concat
    x = Concatenate()([d_flat1, f_flat1]) 
    
    ## dense layer
    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)
        #x = BatchNormalization()(x)
        
    ## last layer
    outputs = Dense(n_outputs, activation=last_avf)(x)
    model = tf.keras.Model(inputs=[d_inputs1, f_inputs1], outputs=outputs)
    
    return model



def _AggMapAddPathNet(molmap_shape,  additional_shape,
                    n_outputs = 1,              
                    dense_layers = [128, 32], 
                    dense_avf = 'relu', 
                    last_avf = None):
    
    
    """
    parameters
    ----------------------
    molmap_shape: w, h, c
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    tf.keras.backend.clear_session()
    assert len(molmap_shape) == 3
    inputs = Input(molmap_shape)
    inputs_actvie_amount = Input(additional_shape)
    
    conv1 = Conv2D(48, 13, padding = 'same', activation='relu', strides = 1)(inputs)
    
    conv1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(conv1) #p1
    
    incept1 = Inception(conv1, strides = 1, units = 32)
    
    incept1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(incept1) #p2
    
    incept2 = Inception(incept1, strides = 1, units = 64)
    
    #flatten
    x = GlobalMaxPool2D()(incept2)
    x = tf.keras.layers.concatenate([x, inputs_actvie_amount], axis=-1)
    
    ## dense layer
    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)
        #x = BatchNormalization()(x)
    #last layer
    outputs = Dense(n_outputs,activation=last_avf)(x)
    
    
    model = tf.keras.Model(inputs=[inputs, inputs_actvie_amount], outputs=outputs)
    
    return model



def _AggMapResNet(input_shape,
                 num_resnet_blocks = 8,
                n_outputs = 1, 
                dense_layers = [128, 32], 
                dense_avf = 'relu', 
                last_avf = None                
                ):
    
    
    """
    parameters
    ----------------------
    molmap_shape: w, h, c
    num_resnet_blocks: int
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(input_shape) #input_shape = (24, 24, 3)
#     x = layers.Conv2D(32, 11, activation='relu')(inputs)
#     x = layers.Conv2D(64, 3, activation='relu')(x)
#     x = layers.MaxPooling2D(3)(x)
    x = Conv2D(64,  13, padding = 'same', activation='relu', strides = 1)(inputs)
    x = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x) #p1
    
    ## renet block 
    for i in range(num_resnet_blocks):
        x = resnet_block(x, 64, 3)

    x = layers.Conv2D(256, 3, activation='relu')(x)
    x = layers.GlobalMaxPool2D()(x)

     ## dense layer
    for units in dense_layers:
        x = layers.Dense(units, activation = dense_avf)(x)
        #x = BatchNormalization()(x)
    #last layer
    outputs = Dense(n_outputs,activation=last_avf)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model




