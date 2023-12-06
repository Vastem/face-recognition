import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

#Reconocimiento Facial en Python usando CV2

#Personas dentro del dataset (Según el orden dentro del dataset)
#Para el dataset las diferentes personas deben tener su propia carepta
#El nombre de las carpetas debe ser " s<nombre> "
subjects = ["", "AMLO", "Obama"]

#Carpeta de los datos de prueba
testDataFolderPath = "./database/testData/"

#Carpeta de los datos de entrenamiento
trainingDataFolderPath = "./database/trainData"

#Detecta las caras en la imagen dada como parámetro, no regresa nada en caso de no encontrar
def detectFace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0]

    return gray[y:y+w, x:x+h], faces[0]

#Entra a la carpeta dada como parámetro examinando cada carpeta y sus archivos dentro.
#Para poder detectar las carpetas deben estar nombradas con una " s " al inicio
def prepareTrainingData(dataFolderPath):

    dirs = os.listdir(dataFolderPath)

    faces = []

    labels = []

    for dirName in dirs:
        if not dirName.startswith("s"):
            continue

        label = int(dirName.replace("s", ""))

        subjectDirPath = dataFolderPath + "/" + dirName

        subjectImagesName = os.listdir(subjectDirPath)

        for imageName in subjectImagesName:

            if imageName.startswith("."):
                continue

            imagePath = subjectDirPath + "/" + imageName

            image = cv2.imread(imagePath)

            face, rect = detectFace(image)

            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels

##Preparando datos
print("Preparando datos. . .")
faces, labels = prepareTrainingData(trainingDataFolderPath)
print("Datos preparados")

print("Total Faces: ", len(faces))
print("Total lables ", len(labels))

#cv2.face.LBPHFaceRecognizer_create() - depreciado en las nuevas versiones, usando una versión antigua
#Crea el objeto para reconocer caras
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

#Entrena el reconocedor de caras utilizando las caras descubiertas en el dataset
faceRecognizer.train(faces, np.array(labels))

#Dibuja un rectangulo en la imagen
#El rectangulo se pone donde la cara es detectada
def drawRectangle(img, rect):
    (x,y,w,h) = rect
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

#Dibuja un texto en la imagen
#El texto se escribe debajo del rectángulo y pone el nombre de la cara detectada
def drawText(img, text, x, y):
    cv2.putText(img, text,(x, y), cv2.FONT_HERSHEY_PLAIN, 1.5 , (0, 255, 0) , 2)

#Predice cual de las caras encontradas es la cara de la imagen dada
def predict(testImg):
    img = testImg.copy()

    face, rect = detectFace(img)

    (label, confidence) = faceRecognizer.predict(face)

    labelText = subjects[label]

    drawRectangle(img, rect)

    drawText(img, labelText, rect[0], rect[1]-5)

    return img

print("Prediciendo imagenes...")


## Cargar y mostrar imágenes de prueba

dirs = os.listdir(testDataFolderPath)

predictTestList = []

for dirName in dirs:
    if not dirName.startswith("s"):
        continue

    label = int(dirName.replace("s", ""))

    testDirPath = testDataFolderPath + "/" + dirName

    imgTestPath = os.listdir(testDirPath)

    for testImg in imgTestPath:

        if testImg.startswith("."):
            continue

        imagePath = testDirPath + "/" + testImg

        testImageRead = cv2.imread(imagePath)
        predictTest = predict(testImageRead)

        predictTestList.append(predictTest)


for predictResult in predictTestList:
        plt.imshow(predictResult)
        plt.show()