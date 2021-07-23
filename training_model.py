from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras import backend as K
from keras.layers.core import Dropout
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras.layers import Conv1D, MaxPool1D, BatchNormalization, Lambda
from keras.callbacks import ModelCheckpoint
import tensorflow as tf


# This is the CNN that i have the best results and can load the weights
# All the parts that has comments is other configurations that i tried


def textCNNwithoutembedding(height,width,classes,activ,filter_sizes = [1,2,3,5,7],num_filters = 36, losstype = 'binary_crossentropy'):

    inp = Input(shape=(height,width), name='input')
    #x = Lambda(lambda x: K.reverse(x,axes=-1))(inp)
    x = SpatialDropout1D(0.2, name = 'drop1')(inp)

    #dilation_rate=1

    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]),
                                 kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu', name = 'conv1')(x)
    #conv_0dilated = Conv1D(num_filters, kernel_size=(filter_sizes[0]), dilation_rate = 4, padding="same",
    #                             kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu')(x)

    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]),
                                 kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu', name = 'conv2')(x)
    #conv_1dilated = Conv1D(num_filters, kernel_size=(filter_sizes[1]), dilation_rate= 4, padding="same",
    #                             kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu')(x)

    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]), 
                                 kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu', name = 'conv3')(x)
    #conv_2dilated = Conv1D(num_filters, kernel_size=(filter_sizes[2]), dilation_rate= 4, padding="same",
    #                             kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu')(x)                             

    conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]),
                                 kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu', name = 'conv4')(x)
    #conv_3dilated = Conv1D(num_filters, kernel_size=(filter_sizes[3]), dilation_rate= 4, padding="same",
    #                             kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu')(x)

    conv_4 = Conv1D(num_filters, kernel_size=(filter_sizes[4]),
                                 kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu', name = 'conv5')(x)
    #conv_4dilated = Conv1D(num_filters, kernel_size=(filter_sizes[4]), dilation_rate= 4, padding="same",
    #                             kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu')(x)                                                         
                                                           
    
    maxpool_0 = MaxPool1D(pool_size=(height - filter_sizes[0] + 1), name = 'pool1')(conv_0)
    #maxpool_0dilated = MaxPool1D(pool_size=(height - filter_sizes[0] + 1))(conv_0dilated)
    
    maxpool_1 = MaxPool1D(pool_size=(height - filter_sizes[1] + 1), name = 'pool2')(conv_1)
    #maxpool_1dilated = MaxPool1D(pool_size=(height - filter_sizes[1] + 1))(conv_1dilated)

    maxpool_2 = MaxPool1D(pool_size=(height - filter_sizes[2] + 1), name = 'pool3')(conv_2)
    #maxpool_2dilated = MaxPool1D(pool_size=(height - filter_sizes[2] + 1))(conv_2dilated)

    maxpool_3 = MaxPool1D(pool_size=(height - filter_sizes[3] + 1), name = 'pool4')(conv_3)
    #maxpool_3dilated = MaxPool1D(pool_size=(height - filter_sizes[3] + 1))(conv_3dilated)

    maxpool_4 = MaxPool1D(pool_size=(height - filter_sizes[4] + 1), name = 'pool5')(conv_4)
    #maxpool_4dilated = MaxPool1D(pool_size=(height - filter_sizes[4] + 1))(conv_4dilated)

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3,maxpool_4])
    #z = Concatenate(axis=1)([maxpool_0, maxpool_0dilated, maxpool_1, maxpool_1dilated, maxpool_2, maxpool_2dilated, maxpool_3, maxpool_3dilated, maxpool_4, maxpool_4dilated])


    z = Flatten(name = 'flatten')(z)
    z = BatchNormalization()(z) 

    #dense1 = Dense(1024,activation='relu')(z) #,kernel_regularizer=tf.keras.regularizers.l2(0.05)
    #drop = Dropout(0.3)(dense1)
    #dense2 = Dense(512,activation='relu')(dense1) #,kernel_regularizer=tf.keras.regularizers.l2(0.05)
    #drop2 = Dropout(0.3)(dense2)
    #dense3 = Dense(256,activation='relu')(dense2) #,kernel_regularizer=tf.keras.regularizers.l2(0.05)
    #dense4 = Dense(128,activation='relu')(dense3) #,kernel_regularizer=tf.keras.regularizers.l2(0.05)

    outp = Dense(classes, name = 'dense1')(z)

    modelnoactivation = Model(inputs = inp, outputs = outp)

    outp = Activation(activ, name = 'activ')(outp)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss=losstype, optimizer='adam', metrics=['accuracy'] )
    #model.summary()

    return model,modelnoactivation

# This is the code of the MLP, i tried other configurations 
# but i ended up using the CNN

def build_modelMLP(n_classes, shape): 

    #drop = tf.keras.layers.Dropout(0.3)(dense)

    inp = tf.keras.layers.Input(shape=(shape,), name="input_ids")
    dense = tf.keras.layers.Dense(128, activation='relu')(inp)
    dense = tf.keras.layers.Dense(128, activation='relu')(dense)
    #dense = tf.keras.layers.Dense(128, activation='relu')(dense)
    #dense = tf.keras.layers.Dense(128, activation='relu')(dense)
    dense = tf.keras.layers.Dense(128, activation='relu')(dense)
    dense = tf.keras.layers.Dense(64, activation='relu')(dense)
    pred = tf.keras.layers.Dense(n_classes, activation='softmax')(dense)
    
    model = tf.keras.models.Model(inputs=inp, outputs=pred)
    Adam = tf.keras.optimizers.Adam(lr = 0.005)
    model.compile(loss= 'binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    #model.summary()

    return model
