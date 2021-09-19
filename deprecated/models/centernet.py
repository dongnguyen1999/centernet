from tensorflow.keras.layers import Dense,Dropout, Conv2D,Conv2DTranspose, BatchNormalization, Activation,AveragePooling2D,GlobalAveragePooling2D, Input, Concatenate, MaxPool2D, Add, UpSampling2D, LeakyReLU,ZeroPadding2D
from tensorflow.keras.models import Model
from utils.config import Config

def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):
    x_deep= Conv2DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(x_deep)
    x_deep = BatchNormalization()(x_deep)   
    x_deep = LeakyReLU(alpha=0.1)(x_deep)
    x = Concatenate()([x_shallow, x_deep])
    x=Conv2D(out_ch, kernel_size=1, strides=1, padding="same")(x)
    x = BatchNormalization()(x)   
    x = LeakyReLU(alpha=0.1)(x)
    return x
  


def conv_batchnormal_relu(x, out_layer, kernel, stride):
    x = Conv2D(out_layer, kernel_size=kernel, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def resblock(x_in,layer_n):
    x=conv_batchnormal_relu(x_in,layer_n,3,1)
    x=conv_batchnormal_relu(x,layer_n,3,1)
    x=Add()([x,x_in])
    return x  


#I use the same network at CenterNet
def create_model(config: Config):

    output_layer_n = config.num_classes + 4
    input_layer = Input((config.input_size, config.input_size, 3)) #512 as default
    
    #resized input

    input_layer_1=AveragePooling2D(2)(input_layer)
    input_layer_2=AveragePooling2D(2)(input_layer_1)

    #### ENCODER ####

    x_0= conv_batchnormal_relu(input_layer, 16, 3, 2)#512->256
    concat_1 = Concatenate()([x_0, input_layer_1])

    x_1= conv_batchnormal_relu(concat_1, 32, 3, 2)#256->128
    concat_2 = Concatenate()([x_1, input_layer_2])

    x_2= conv_batchnormal_relu(concat_2, 64, 3, 2)#128->64
    x=conv_batchnormal_relu(x_2,64,3,1)
    x=resblock(x,64)
    x=resblock(x,64)
    
    x_3= conv_batchnormal_relu(x, 128, 3, 2)#64->32
    x= conv_batchnormal_relu(x_3, 128, 3, 1)
    x=resblock(x,128)
    x=resblock(x,128)
    x=resblock(x,128)
    
    x_4= conv_batchnormal_relu(x, 256, 3, 2)#32->16
    x= conv_batchnormal_relu(x_4, 256, 3, 1)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
 
    x_5= conv_batchnormal_relu(x, 512, 3, 2)#16->8
    x= conv_batchnormal_relu(x_5, 512, 3, 1) 
    x=resblock(x,512)
    x=resblock(x,512)
    x=resblock(x,512)
    
    #### DECODER ####
    x_1= conv_batchnormal_relu(x_1, output_layer_n, 1, 1)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_2= conv_batchnormal_relu(x_2, output_layer_n, 1, 1)
    x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_3= conv_batchnormal_relu(x_3, output_layer_n, 1, 1)
    x_3 = aggregation_block(x_3, x_4, output_layer_n, output_layer_n) 
    x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

    x_4= conv_batchnormal_relu(x_4, output_layer_n, 1, 1)

    x=conv_batchnormal_relu(x, output_layer_n, 1, 1)
    x= UpSampling2D(size=(2, 2))(x)#8->16 

    x = Concatenate()([x, x_4])
    x=conv_batchnormal_relu(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)#16->32

    x = Concatenate()([x, x_3])
    x=conv_batchnormal_relu(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)#32->64 

    x = Concatenate()([x, x_2])
    x=conv_batchnormal_relu(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)#64->128 

    x = Concatenate()([x, x_1])
    x=Conv2D(output_layer_n, kernel_size=3, strides=1, padding="same")(x)
    out = Activation("sigmoid")(x)
    
    model=Model(input_layer, out)
    
    return model