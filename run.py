import cv2
import pygame
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import time

# Initialisation de pygame
pygame.init()

# Chargement du son à jouer lors de la détection
alert_sound = pygame.mixer.Sound("Quack.wav")

# Initialisation du détecteur de visage dlib (HOG-based)
detector = dlib.get_frontal_face_detector()

# Chargement du prédicteur de points de repère pour les yeux
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Définition des indices des points de repère pour les yeux
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Ouvrir la capture vidéo de la caméra (0 est généralement la caméra par défaut)
capture = cv2.VideoCapture(0)

eyes_opened = False  # Les yeux sont initialement fermés
eyes_were_open = False  # Les yeux étaient initialement fermés
eyes_closed_start_time = None  # Temps de début de fermeture des yeux
alert_triggered = False  # Alerte déjà déclenchée

# Définir le délai minimum (en secondes) avant de déclencher l'alerte si les yeux sont fermés
eyes_closed_threshold = 2


def eye_aspect_ratio(eye):
    # Calcul du ratio d'aspect de l'œil
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


while True:
    # Lire le frame de la capture vidéo
    ret, frame = capture.read()
    if not ret:
        break

    # Convertir le frame en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages dans l'image
    rects = detector(gray, 0)

    # Boucle sur les visages détectés
    for rect in rects:
        # Détection des points de repère faciaux pour les yeux
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extraction des coordonnées des yeux à partir des points de repère
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calcul du ratio d'aspect pour chaque œil
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Calcul du EAR moyen pour les deux yeux
        ear = (leftEAR + rightEAR) / 2.0

        # Dessiner les contours des yeux et détecter les états
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Détermination de l'état des yeux
        if ear < 0.25:  # Valeur seuil à ajuster selon la précision souhaitée
            eyes_closed = True
        else:
            eyes_closed = False

        # Affichage de l'état des yeux sur le frame
        if eyes_closed:
            cv2.putText(frame, "Yeux fermes", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not eyes_were_open:
                eyes_closed_start_time = time.time()
        else:
            cv2.putText(frame, "Yeux ouverts", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            eyes_closed_start_time = None

        # Vérification si les yeux sont fermés pendant plus que le seuil défini
        if eyes_closed_start_time is not None and time.time() - eyes_closed_start_time >= eyes_closed_threshold and not alert_triggered:
            alert_sound.play()
            alert_triggered = True
            print("Alerte ! Clignement des yeux détecté.")

        # Mise à jour de l'état précédent des yeux
        eyes_were_open = not eyes_closed

    # Affichage du frame avec les visages et les yeux détectés
    cv2.imshow('Webcam', frame)

    # Sortie de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération de la capture et fermeture de la fenêtre
capture.release()
cv2.destroyAllWindows()
