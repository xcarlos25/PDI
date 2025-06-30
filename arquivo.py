pip install tensorflow opencv-python numpy
/dataset/
|-- treino/
|   |-- raiva/
|   |   |-- img1.jpg
|   |   +-- ...
|   |-- felicidade/
|   |-- tristeza/
|   +-- ...
|
|-- validacao/
|   |-- raiva/
|   |-- felicidade/
|   |-- tristeza/
|   +-- ...
# Crie um arquivo chamado 'train_models.py'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir parâmetros
IMG_WIDTH, IMG_HEIGHT = 48, 48
BATCH_SIZE = 64
TRAIN_DIR = 'caminho/para/seu/dataset/treino'
VALIDATION_DIR = 'caminho/para/seu/dataset/validacao'
NUM_CLASSES = 7

# Gerador para o modelo CNN (escala de cinza)
train_datagen_gray = ImageDataGenerator(
        rescale=1./255,
            rotation_range=30,
                shear_range=0.2,
                    zoom_range=0.2,
                        horizontal_flip=True,
                            fill_mode='nearest'
)
validation_datagen_gray = ImageDataGenerator(rescale=1./255)

train_generator_gray = train_datagen_gray.flow_from_directory(
        TRAIN_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                    color_mode='grayscale',
                        class_mode='categorical'
)
validation_generator_gray = validation_datagen_gray.flow_from_directory(
        VALIDATION_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                    color_mode='grayscale',
                        class_mode='categorical'
)

# Gerador para ResNet e MobileNet (RGB)
# Modelos pré-treinados esperam 3 canais de cor
train_datagen_rgb = ImageDataGenerator(
        rescale=1./255,
            rotation_range=30,
                shear_range=0.2,
                    zoom_range=0.2,
                        horizontal_flip=True,
                            fill_mode='nearest'
)
validation_datagen_rgb = ImageDataGenerator(rescale=1./255)

train_generator_rgb = train_datagen_rgb.flow_from_directory(
        TRAIN_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                    color_mode='rgb',
                        class_mode='categorical'
)
validation_generator_rgb = validation_datagen_rgb.flow_from_directory(
        VALIDATION_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                    color_mode='rgb',
                        class_mode='categorical'
)

)
)
)
)
)
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_custom_cnn(input_shape, num_classes):
    model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                    BatchNormalization(),
                            MaxPooling2D(pool_size=(2, 2)),
                                    
                                            Conv2D(64, (3, 3), activation='relu'),
                                                    BatchNormalization(),
                                                            MaxPooling2D(pool_size=(2, 2)),
                                                                    
                                                                            Conv2D(128, (3, 3), activation='relu'),
                                                                                    BatchNormalization(),
                                                                                            MaxPooling2D(pool_size=(2, 2)),
                                                                                                    
                                                                                                            Flatten(),
                                                                                                                    Dense(256, activation='relu'),
                                                                                                                            Dropout(0.5),
                                                                                                                                    Dense(num_classes, activation='softmax')
                                                                                                                                        ])
                                                                                                                                            return model

                                                                                                                                            # Construindo e compilando o modelo CNN
                                                                                                                                            model_cnn = build_custom_cnn((IMG_WIDTH, IMG_HEIGHT, 1), NUM_CLASSES)
                                                                                                                                            model_cnn.compile(optimizer=Adam(learning_rate=0.001),
                                                                                                                                                              loss='categorical_crossentropy',
                                                                                                                                                                                metrics=['accuracy'])

                                                                                                                                                                                print("Treinando o modelo CNN Customizado...")
                                                                                                                                                                                # Descomente as linhas abaixo para treinar
                                                                                                                                                                                # history_cnn = model_cnn.fit(
                                                                                                                                                                                #     train_generator_gray,
                                                                                                                                                                                #     epochs=50,
                                                                                                                                                                                #     validation_data=validation_generator_gray
                                                                                                                                                                                # )
                                                                                                                                                                                # model_cnn.save('emotion_model_cnn.h5')
                                                                                          from tensorflow.keras.applications import ResNet50
                                                                                          from tensorflow.keras.layers import GlobalAveragePooling2D
                                                                                          from tensorflow.keras.models import Model

                                                                                          def build_resnet50(input_shape, num_classes):
                                                                                              base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
                                                                                                  base_model.trainable = False  # Congela a base

                                                                                                      x = base_model.output
                                                                                                          x = GlobalAveragePooling2D()(x)
                                                                                                              x = Dense(256, activation='relu')(x)
                                                                                                                  x = Dropout(0.5)(x)
                                                                                                                      predictions = Dense(num_classes, activation='softmax')(x)

                                                                                                                              model = Model(inputs=base_model.input, outputs=predictions)
                                                                                                                                  return model

                                                                                                                                  # Construindo e compilando o modelo ResNet50
                                                                                                                                  model_resnet = build_resnet50((IMG_WIDTH, IMG_HEIGHT, 3), NUM_CLASSES)
                                                                                                                                  model_resnet.compile(optimizer=Adam(learning_rate=0.001),
                                                                                                                                                       loss='categorical_crossentropy',
                                                                                                                                                                            metrics=['accuracy'])

                                                                                                                                                                            print("\nTreinando o modelo ResNet-50...")
                                                                                                                                                                            # Descomente as linhas abaixo para treinar
                                                                                                                                                                            # history_resnet = model_resnet.fit(
                                                                                                                                                                            #     train_generator_rgb,
                                                                                                                                                                            #     epochs=30,
                                                                                                                                                                            #     validation_data=validation_generator_rgb
                                                                                                                                                                            # )
                                                                                                                                                                            # model_resnet.save('emotion_model_resnet50.h5')
                                                                                                                                          from tensorflow.keras.applications import MobileNetV2

                                                                                                                                          def build_mobilenetv2(input_shape, num_classes):
                                                                                                                                              base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
                                                                                                                                                  base_model.trainable = False

                                                                                                                                                      x = base_model.output
                                                                                                                                                          x = GlobalAveragePooling2D()(x)
                                                                                                                                                              x = Dense(256, activation='relu')(x)
                                                                                                                                                                  x = Dropout(0.5)(x)
                                                                                                                                                                      predictions = Dense(num_classes, activation='softmax')(x)

                                                                                                                                                                              model = Model(inputs=base_model.input, outputs=predictions)
                                                                                                                                                                                  return model

                                                                                                                                                                                  # Construindo e compilando o modelo MobileNetV2
                                                                                                                                                                                  model_mobilenet = build_mobilenetv2((IMG_WIDTH, IMG_HEIGHT, 3), NUM_CLASSES)
                                                                                                                                                                                  model_mobilenet.compile(optimizer=Adam(learning_rate=0.001),
                                                                                                                                                                                                          loss='categorical_crossentropy',
                                                                                                                                                                                                                                  metrics=['accuracy'])

                                                                                                                                                                                                                                  print("\nTreinando o modelo MobileNetV2...")
                                                                                                                                                                                                                                  # Descomente as linhas abaixo para treinar
                                                                                                                                                                                                                                  # history_mobilenet = model_mobilenet.fit(
                                                                                                                                                                                                                                  #     train_generator_rgb,
                                                                                                                                                                                                                                  #     epochs=30,
                                                                                                                                                                                                                                  #     validation_data=validation_generator_rgb
                                                                                                                                                                                                                                  # )
                                                                                                                                                                                                                                  # model_mobilenet.save('emotion_model_mobilenetv2.h5')
                                                                                                                   # Crie um arquivo chamado 'run_video.py'

                                                                                                                   import cv2
                                                                                                                   import numpy as np
                                                                                                                   from tensorflow.keras.models import load_model
                                                                                                                   from tensorflow.keras.preprocessing.image import img_to_array

                                                                                                                   # --- CONFIGURAÇÃO ---
                                                                                                                   # Carregar os modelos treinados
                                                                                                                   try:
                                                                                                                       model_cnn = load_model('emotion_model_cnn.h5')
                                                                                                                           model_resnet = load_model('emotion_model_resnet50.h5')
                                                                                                                               model_mobilenet = load_model('emotion_model_mobilenetv2.h5')
                                                                                                                               except Exception as e:
                                                                                                                                   print(f"Erro ao carregar os modelos: {e}")
                                                                                                                                       print("Certifique-se de que os arquivos .h5 estão no mesmo diretório ou treine os modelos primeiro.")
                                                                                                                                           exit()

                                                                                                                                           # Dicionário de emoções e cores para exibição
                                                                                                                                           EMOTION_LABELS = {0: 'Raiva', 1: 'Nojo', 2: 'Medo', 3: 'Feliz', 4: 'Triste', 5: 'Surpreso', 6: 'Neutro'}
                                                                                                                                           MODEL_COLORS = {
                                                                                                                                               "CNN": (0, 255, 0),       # Verde
                                                                                                                                                   "ResNet50": (255, 255, 0), # Ciano
                                                                                                                                                       "MobileNetV2": (0, 165, 255) # Laranja
                                                                                                                                                       }

                                                                                                                                                       # Carregar o classificador de faces Haar Cascade do OpenCV
                                                                                                                                                       face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

                                                                                                                                                       # Iniciar a captura de vídeo
                                                                                                                                                       cap = cv2.VideoCapture(0) # Use 0 para webcam, ou o caminho para um arquivo de vídeo

                                                                                                                                                       if not cap.isOpened():
                                                                                                                                                           print("Erro: Não foi possível abrir a câmera.")
                                                                                                                                                               exit()

                                                                                                                                                               # --- LOOP PRINCIPAL ---
                                                                                                                                                               while True:
                                                                                                                                                                   ret, frame = cap.read()
                                                                                                                                                                       if not ret:
                                                                                                                                                                               print("Fim do stream de vídeo.")
                                                                                                                                                                                       break

                                                                                                                                                                                           # Converter para escala de cinza para detecção de rosto
                                                                                                                                                                                               gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                                                                                                                                                                                                       # Detectar rostos
                                                                                                                                                                                                           faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

                                                                                                                                                                                                               for (x, y, w, h) in faces:
                                                                                                                                                                                                                       # Desenhar retângulo ao redor do rosto detectado
                                                                                                                                                                                                                               cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                                                                                                                                                                                                                                               # Extrair o ROI (Region of Interest - o rosto)
                                                                                                                                                                                                                                                       roi_gray = gray_frame[y:y+h, x:x+w]
                                                                                                                                                                                                                                                               roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                                                                                                                                                                                                                                                                       # 1. Predição com CNN Customizada (entrada em escala de cinza)
                                                                                                                                                                                                                                                                               roi_cnn = roi_gray.astype('float') / 255.0
                                                                                                                                                                                                                                                                                       roi_cnn = img_to_array(roi_cnn)
                                                                                                                                                                                                                                                                                               roi_cnn = np.expand_dims(roi_cnn, axis=0)
                                                                                                                                                                                                                                                                                                       pred_cnn = model_cnn.predict(roi_cnn, verbose=0)[0]
                                                                                                                                                                                                                                                                                                               label_cnn = EMOTION_LABELS[pred_cnn.argmax()]

                                                                                                                                                                                                                                                                                                                               # 2. Predição com ResNet e MobileNet (entrada RGB)
                                                                                                                                                                                                                                                                                                                                       roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
                                                                                                                                                                                                                                                                                                                                               roi_rgb = roi_rgb.astype('float') / 255.0
                                                                                                                                                                                                                                                                                                                                                       roi_rgb = img_to_array(roi_rgb)
                                                                                                                                                                                                                                                                                                                                                               roi_rgb = np.expand_dims(roi_rgb, axis=0)

                                                                                                                                                                                                                                                                                                                                                                               pred_resnet = model_resnet.predict(roi_rgb, verbose=0)[0]
                                                                                                                                                                                                                                                                                                                                                                                       label_resnet = EMOTION_LABELS[pred_resnet.argmax()]

                                                                                                                                                                                                                                                                                                                                                                                               pred_mobilenet = model_mobilenet.predict(roi_rgb, verbose=0)[0]
                                                                                                                                                                                                                                                                                                                                                                                                       label_mobilenet = EMOTION_LABELS[pred_mobilenet.argmax()]

                                                                                                                                                                                                                                                                                                                                                                                                               # Exibir os resultados na tela
                                                                                                                                                                                                                                                                                                                                                                                                                       y_offset = y - 10
                                                                                                                                                                                                                                                                                                                                                                                                                               cv2.putText(frame, f'MobileNetV2: {label_mobilenet}', (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MODEL_COLORS["MobileNetV2"], 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                       cv2.putText(frame, f'ResNet50: {label_resnet}', (x, y_offset - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MODEL_COLORS["ResNet50"], 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                               cv2.putText(frame, f'CNN: {label_cnn}', (x, y_offset - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MODEL_COLORS["CNN"], 2)


                                                                                                                                                                                                                                                                                                                                                                                                                                                   # Mostrar o frame resultante
                                                                                                                                                                                                                                                                                                                                                                                                                                                       cv2.imshow('Detector de Emocoes - Pressione Q para Sair', frame)

                                                                                                                                                                                                                                                                                                                                                                                                                                                           # Condição de parada
                                                                                                                                                                                                                                                                                                                                                                                                                                                               if cv2.waitKey(1) & 0xFF == ord('q'):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                       break

                                                                                                                                                                                                                                                                                                                                                                                                                                                                       # --- FINALIZAÇÃO ---
                                                                                                                                                                                                                                                                                                                                                                                                                                                                       cap.release()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                       cv2.destroyAllWindows()
