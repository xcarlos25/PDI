train_models.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator from tensorflow.keras.models import Sequential, Model from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D from tensorflow.keras.optimizers import Adam from tensorflow.keras.applications import ResNet50, MobileNetV2

Definir parâmetros

IMG_WIDTH, IMG_HEIGHT = 48, 48 BATCH_SIZE = 64 TRAIN_DIR = 'dataset/treino' VALIDATION_DIR = 'dataset/validacao' NUM_CLASSES = 7

Geradores de imagem

train_datagen_gray = ImageDataGenerator( rescale=1./255, rotation_range=30, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest' )

validation_datagen_gray = ImageDataGenerator(rescale=1./255)

train_generator_gray = train_datagen_gray.flow_from_directory( TRAIN_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, color_mode='grayscale', class_mode='categorical' )

validation_generator_gray = validation_datagen_gray.flow_from_directory( VALIDATION_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, color_mode='grayscale', class_mode='categorical' )

train_datagen_rgb = ImageDataGenerator( rescale=1./255, rotation_range=30, shear_range
