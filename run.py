import cv2
import os
import pygame
import time

# Initialisation de pygame
pygame.init()

# Chargement du son à jouer lors de la détection
alert_sound = pygame.mixer.Sound("Quack.wav")

# Chemin complet vers le fichier XML du détecteur de visages
face_cascade_xml_path = cv2.data.haarcascades + \
    'haarcascade_frontalface_default.xml'

# Création du détecteur de visages
face_cascade = cv2.CascadeClassifier(face_cascade_xml_path)

# Chemin complet vers le fichier XML du détecteur d'yeux
eye_cascade_xml_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

# Création du détecteur d'yeux
eye_cascade = cv2.CascadeClassifier(eye_cascade_xml_path)

# Ouvrir la capture vidéo de la caméra (0 est généralement la caméra par défaut)
capture = cv2.VideoCapture(0)

eyes_opened = False  # Les yeux sont initialement fermés
eyes_were_open = False  # Les yeux étaient initialement fermés

while True:
    # Lire le frame de la capture vidéo
    ret, frame = capture.read()

    # Convertir le frame en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages dans l'image en niveaux de gris
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    eyes_detected = False  # Réinitialisation du statut des yeux détectés pour ce frame

    # Pour chaque visage détecté, recherche des yeux
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        # Ajustement des paramètres pour réduire la sensibilité
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Dessin des contours des yeux détectés
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            contours, _ = cv2.findContours(
                eye_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

            for contour in contours:
                # Ajouter le décalage pour les coordonnées globales
                contour += (x + ex, y + ey)
                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

            # Affichage "Yeux ouverts" ou "Yeux fermés" sur chaque œil détecté
            cv2.putText(frame, "Yeux ouverts" if eyes_opened else "Yeux fermés", (x + ex, y + ey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if eyes_opened else (0, 0, 255), 2)

        # Si des yeux sont détectés, les marquer comme ouverts
        if len(eyes) > 0:
            eyes_detected = True

    # Vérification si les yeux sont ouverts
    if eyes_detected:
        eyes_opened = True
    else:
        eyes_opened = False

    # Si les yeux étaient ouverts et maintenant ils sont fermés, déclencher l'alerte
    if eyes_were_open and not eyes_opened:
        alert_sound.play()
        print("Alerte ! Clignement des yeux détecté.")

    # Mise à jour de l'état précédent des yeux
    eyes_were_open = eyes_opened

    # Affichage du frame avec les visages et les yeux détectés
    cv2.imshow('Webcam', frame)

    # Sortie de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération de la capture et fermeture de la fenêtre
capture.release()
cv2.destroyAllWindows()
