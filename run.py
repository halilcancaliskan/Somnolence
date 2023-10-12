# -*- coding: utf-8 -*-

import cv2
import os
import pygame

# Initialisation de pygame
pygame.init()

# Chargement du son à jouer lors de la détection
# Assurez-vous d'avoir un fichier "alert.wav" dans le répertoire
alert_sound = pygame.mixer.Sound("Quack.wav")

# Obtenez le chemin complet vers le fichier XML du détecteur de visages
face_cascade_xml_path = os.path.join(
    cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

# Créez le détecteur de visages
face_cascade = cv2.CascadeClassifier(face_cascade_xml_path)

# Obtenez le chemin complet vers le fichier XML du détecteur d'yeux
eye_cascade_xml_path = os.path.join(
    cv2.data.haarcascades, 'haarcascade_eye.xml')

# Créez le détecteur d'yeux
eye_cascade = cv2.CascadeClassifier(eye_cascade_xml_path)

# Ouvrez la capture vidéo de la caméra (0 est généralement la caméra par défaut)
capture = cv2.VideoCapture(0)

eyes_opened = True  # Les yeux sont initialement ouverts

while True:
    ret, frame = capture.read()

    # Convertissez le frame en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Utilisez le détecteur de visages pour trouver des visages dans l'image en niveaux de gris
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Pour chaque visage détecté, recherchez les yeux
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        # Ajustez les paramètres pour réduire la sensibilité
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))

        # Si au moins un œil est détecté, les yeux sont ouverts
        if len(eyes) > 0:
            eyes_opened = True
        else:
            # Si aucun œil n'est détecté, les yeux sont fermés
            eyes_opened = False

    # Si les yeux sont fermés, déclenchez l'alerte sonore
    if not eyes_opened:
        alert_sound.play()

    # Dessinez des rectangles autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Affichez le frame avec les visages et les yeux détectés
    cv2.imshow('Webcam', frame)

    # Quittez la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérez la capture et fermez la fenêtre
capture.release()
cv2.destroyAllWindows()
