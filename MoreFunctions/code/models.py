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

def getmodelcheckpoint(filepath, verbose = 1):

	return ModelCheckpoint(filepath=filepath, verbose=verbose, save_best_only=True)


def AlexNet(width, height, depth, classes,activ):
  model = Sequential()
  inputShape = (height, width, depth)
  model.add(Conv2D(32, (3, 3), input_shape=inputShape, padding = 'same'))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=2))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(32, (3, 3), padding = 'same'))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=2))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))


  model.add(Conv2D(128, (3, 3), padding = 'same'))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=2))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, (3, 3), padding = 'same'))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=2))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))


  model.add(Conv2D(64, (3, 3), padding = 'same'))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=2))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  # first (and only) set of FC => RELU layers
  model.add(Flatten())
  model.add(Dense(1024))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dense(512))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  # softmax classifier
  model.add(Dense(classes))
  model.add(Activation(activ))
  
  return model


def VGG16(width, height, depth, classes,activ):
  # initialize the model
  model = Sequential()
  inputShape = (height, width, depth)
  
  model.add(Conv2D(64, (3, 3), padding="same",activation = 'relu', input_shape=inputShape))
  model.add(Conv2D(64, (3, 3), padding="same",activation = 'relu'))
  model.add(BatchNormalization(axis=2))
  
  model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2)))
  
  model.add(Conv2D(128, (3, 3), padding="same",activation = 'relu'))
  model.add(Conv2D(128, (3, 3), padding="same", activation = 'relu'))
  model.add(BatchNormalization(axis=2))
  
  model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2)))
  
  model.add(Conv2D(256, (3, 3), padding="same",activation = 'relu'))
  model.add(Conv2D(256, (3, 3), padding="same", activation = 'relu'))
  model.add(BatchNormalization(axis=2))
  
  model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2)))
  
  model.add(Conv2D(512, (3, 3), padding="same",activation = 'relu'))
  model.add(Conv2D(512, (3, 3), padding="same", activation = 'relu'))
  model.add(Conv2D(512, (3, 3), padding="same", activation = 'relu'))
  model.add(BatchNormalization(axis=2))
  
  model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2)))
  
  model.add(Conv2D(512, (3, 3), padding="same",activation = 'relu'))
  model.add(Conv2D(512, (3, 3), padding="same", activation = 'relu'))
  model.add(Conv2D(512, (3, 3), padding="same", activation = 'relu'))
  model.add(BatchNormalization(axis=2))
  
  model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2)))
    
  model.add(Flatten())
  model.add(Dense(4096,activation = 'relu'))
  model.add(keras.layers.Dropout(0.4))
  model.add(Dense(4096,activation = 'relu'))
  model.add(keras.layers.Dropout(0.4))
  model.add(Dense(1000,activation = 'relu'))
  model.add(keras.layers.Dropout(0.4))
  
  # softmax classifier
  
  model.add(Dense(classes,activation = activ))
  
  #model.summary()
  
  # return the constructed network architecture
  return model



def LeNet(width, height, depth, classes,activ):
  # initialize the model
  model = Sequential()
  inputShape = (height, width, depth)

  # first set of CONV => RELU => POOL layers
  model.add(Conv2D(20, (5, 5), padding="same",
    input_shape=inputShape))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  # second set of CONV => RELU => POOL layers
  model.add(Conv2D(50, (5, 5), padding="same"))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  # first (and only) set of FC => RELU layers
  model.add(Flatten())
  model.add(Dense(500))
  model.add(Activation("relu"))

  # softmax classifier
  model.add(Dense(classes))
  model.add(Activation(activ))
  
  model.summary()
  
  # return the constructed network architecture
  return model


def build_MiniVGG(width, height, depth, classes,activ):
  # initialize the model
  model = Sequential()
  inputShape = (height, width, depth)# first CONV => RELU => CONV => RELU => POOL layer set
  model.add(Conv2D(32, (3, 3), padding="same",
  input_shape=inputShape))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=2))
  model.add(Conv2D(32, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=2))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  
  # second CONV => RELU => CONV => RELU => POOL layer set
  model.add(Conv2D(64, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=2))
  model.add(Conv2D(64, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=2))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  
  # first (and only) set of FC => RELU layers
  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  
  # softmax classifier
  model.add(Dense(classes))
  model.add(Activation(activ))
  
  model.summary()
  
  # return the constructed network architecture
  return model

def textCNN(embedding_matrix,maxlen,vocab_size,embed_size,classes,activ,filter_sizes = [1,2,3,5],num_filters = 36, losstype = 'binary_crossentropy'):

    inp = Input(shape=(maxlen,))
    #x = Lambda(lambda x: K.reverse(x,axes=-1))(inp)
    x = Embedding(vocab_size, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.4)(x)

    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]),
                                 kernel_initializer='he_normal', activation='elu')(x)
    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]),
                                 kernel_initializer='he_normal', activation='elu')(x)
    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]), 
                                 kernel_initializer='he_normal', activation='elu')(x)
    conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]),
                                 kernel_initializer='he_normal', activation='elu')(x)
    
    maxpool_0 = MaxPool1D(pool_size=(maxlen - filter_sizes[0] + 1))(conv_0)
    maxpool_1 = MaxPool1D(pool_size=(maxlen - filter_sizes[1] + 1))(conv_1)
    maxpool_2 = MaxPool1D(pool_size=(maxlen - filter_sizes[2] + 1))(conv_2)
    maxpool_3 = MaxPool1D(pool_size=(maxlen - filter_sizes[3] + 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = BatchNormalization()(z)

    
        
    outp = Dense(classes)(z)

    modelnoactivation = Model(inputs = inp, outputs = outp)

    outp = Activation(activ)(outp)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss=losstype, optimizer='adam', metrics=['accuracy'] )

    return model,modelnoactivation


def textCNNwithoutembedding(height,width,classes,activ,filter_sizes = [1,2,3,5],num_filters = 36, losstype = 'binary_crossentropy'):

    inp = Input(shape=(height,width))
    #x = Lambda(lambda x: K.reverse(x,axes=-1))(inp)
    x = SpatialDropout1D(0.4)(inp)

    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]),
                                 kernel_initializer='he_normal', activation='elu')(x)
    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]),
                                 kernel_initializer='he_normal', activation='elu')(x)
    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]), 
                                 kernel_initializer='he_normal', activation='elu')(x)
    conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]),
                                 kernel_initializer='he_normal', activation='elu')(x)
    
    maxpool_0 = MaxPool1D(pool_size=(height - filter_sizes[0] + 1))(conv_0)
    maxpool_1 = MaxPool1D(pool_size=(height - filter_sizes[1] + 1))(conv_1)
    maxpool_2 = MaxPool1D(pool_size=(height - filter_sizes[2] + 1))(conv_2)
    maxpool_3 = MaxPool1D(pool_size=(height - filter_sizes[3] + 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = BatchNormalization()(z) 
        
    outp = Dense(classes)(z)

    modelnoactivation = Model(inputs = inp, outputs = outp)

    outp = Activation(activ)(outp)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss=losstype, optimizer='adam', metrics=['accuracy'] )

    return model,modelnoactivation