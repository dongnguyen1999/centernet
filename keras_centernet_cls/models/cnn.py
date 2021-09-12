from tensorflow.keras.layers import Dense,Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import ResNet152
from utils.config import Config

def create_model(config: Config, architecture='vgg_1_block', weights=None, feature_weights=None, predictor_weights=None):
	model_map = {
		'vgg_1_block': vgg_1block,
		'vgg_2_block': vgg_2block,
	}

	transfer_learning_map = {
		'pretrained_vgg16': VGG16,
		'pretrained_vgg19': VGG19,
		'pretrained_inceptionv3': InceptionV3,
		'pretrained_resnet50': ResNet50,
		'pretrained_resnet101': ResNet101,
		'pretrained_resnet152': ResNet152,
	}

	if architecture in model_map:
		model = model_map[architecture](config)
	if architecture in transfer_learning_map:
		model = pretrained_feature_extractor(config, transfer_learning_map[architecture])
	return model

# define cnn model
def vgg_1block(config: Config):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(config.input_size, config.input_size, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(3, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	return model

def vgg_2block(config: Config):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(config.input_size, config.input_size, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	return model

def pretrained_feature_extractor(config: Config, feature_extractor, units=128):
	model = feature_extractor(include_top=False, input_shape=(config.input_size, config.input_size, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(units, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	return model