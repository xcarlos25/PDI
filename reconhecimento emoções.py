import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision.models as models
from torchvision import transforms

# Emoções suportadas
emotions = ['raiva', 'nojo', 'medo', 'feliz', 'triste', 'surpreso', 'neutro']

# --- Funções auxiliares ---

def preprocess_face(face, size=(224,224), gray=True):
    """Pré-processa a imagem facial para ResNet ou SqueezeNet"""
    if gray:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, size)
        face = face[..., np.newaxis]
        face = np.repeat(face, 3, axis=-1)
    else:
        face = cv2.resize(face, size)
    face = face.astype('float32') / 255.0
    return np.expand_dims(face, axis=0)

def load_resnet_model(path):
    """Carrega o modelo treinado em Keras (ResNet-50)"""
    return tf.keras.models.load_model(path)

def predict_emotion_resnet(model, input_img):
    """Predição com ResNet"""
    pred = model.predict(input_img)
    return emotions[np.argmax(pred)]

def load_squeezenet_model(path):
    """Carrega o modelo treinado em PyTorch (SqueezeNet)"""
    model = models.squeezenet1_1(pretrained=False)
    model.classifier[1] = torch.nn.Conv2d(512, len(emotions), kernel_size=(1,1))
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_emotion_squeezenet(model, input_img):
    """Predição com SqueezeNet"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = transform(input_img.squeeze()).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    return emotions[predicted.item()]

# --- Inicialização ---

# Carrega os classificadores de rosto
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Carrega os modelos
print("[INFO] Carregando modelos...")
resnet_model = load_resnet_model('resnet_model.h5')
squeezenet_model = load_squeezenet_model('squeezenet_model.pt')

# Inicia a câmera
cap = cv2.VideoCapture(0)
print("[INFO] Pressione 'q' para sair.")

# --- Loop de vídeo ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        input_resnet = preprocess_face(face_img, (224, 224), gray=True)
        input_squeeze = preprocess_face(face_img, (224, 224), gray=False)

        # Predição com ambos os modelos
        emotion_r = predict_emotion_resnet(resnet_model, input_resnet)
        emotion_s = predict_emotion_squeezenet(squeezenet_model, input_squeeze)

        # Combinação simples das predições
        if emotion_r == emotion_s:
            emotion = emotion_r
        else:
            emotion = f"{emotion_r}/{emotion_s}"

        # Exibe no vídeo
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow('Reconhecimento de Emoções', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalização
cap.release()
cv2.destroyAllWindows()
