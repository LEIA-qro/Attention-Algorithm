import cv2 #OpenCV (cámara)
import mediapipe as mp #modelos de detección faciales 
import time #tiempo en cada estado
from math import hypot #calcular distancia esntre puntos faciales 

mp_face_mesh = mp.solutions.face_mesh #face mesh - puntos de la cara

#FACE MESH
face_mesh = mp_face_mesh.FaceMesh (
    static_image_mode = False, #video en tiempo real
    max_num_faces = 1, #solo 1 cara
    refine_landmarks = True #landmarks mas precisas
    min_detection_confidence = 0.5 #confianza para aceptar una detección
    min_tracking_confidence = 0.5 #confianza para seguir la cara
)

#LANDMARKS
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

#UMBRALES
EAR_THRESHOLD = 0.22 #Eye Aspect Ratio (EAR) en que punto detecta ojos cerrados
EYES_CLOSED_TIME = 1.5 #tiempo antes de marcar los ojos cerrados
HEAD_TURN_THRESHOLD = 0.25 # rango de giro que acepta antes de marcar distracción
NO_FACE_TIME = 1.0 #tiempo antes de marcar ausencia de rostro

#ESTADO
#Van a guardar cuando la persona cerro los ojos o su cara dejo de aparecer en camara 
#y despues se usa para comparar, por ejemplo si pasa poco tiempo es un parpadeo y si es
#mucho somnolencia.

eyes_closed_start = None
no_face_start = None

#FUNCIONES
#Calcula la distancia entre dos puntos  
def distance_points(p1,p2):
    return hypot (p1[0]-p2[0], p1[1]-p2[1])

#Convierte un ladnmark de MediaPipe (coordenadas normalizadas entre 0 y 1) a coordenadas 
#reales en pixeles para OpenCV.
    #Parametros:
        #landmarks: lista de puntos que detecta MediaPiepe
        #index: indice del landmark que se desea obtener
        #width: ancho de la imagen en pixeles
        #height: alto de la imagen en pixeles

def get_point (landmarks, index, width,height):
    lm = landmarks[index]
    return (int(lm.x * width), int(lm.y *height))


#Calcula que tan abierto esta el ojo usando la distancia entre puntos
def eye_aspect_ratio(landmarks, eye_indices, width, height):

    p1 = get_point(landmarks, eye_indices[0], width, height)
    p2 = get_point(landmarks, eye_indices[1], width, height)
    p3 = get_point(landmarks, eye_indices[2], width, height)
    p4 = get_point(landmarks, eye_indices[3], width, height)
    p5 = get_point(landmarks, eye_indices[4], width, height)
    p6 = get_point(landmarks, eye_indices[5], width, height)

    vertical_1 = distance_points(p2, p5)
    vertical_2 = distance_points(p3, p6)
    horizontal = distance_points(p1, p4)

    if horizontal == 0:
        return 0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


#Calcula la desviación horizontal de la cabeza, usando la nariz respecto al centro 
#entre los ojos y el valor se normaliza con la distancia entre ojos.

def head_turn_score(landmarks, width, height):

    nose = get_point(landmarks, NOSE_TIP, width, height)
    left_eye = get_point(landmarks, LEFT_EYE_OUTER, width, height)
    right_eye = get_point(landmarks, RIGHT_EYE_OUTER, width, height)

    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_distance = distance_points(left_eye, right_eye)

    if eye_distance == 0:
        return 0

    score = (nose[0] - eye_center_x) / eye_distance
    return score


#Le asigna a un numero el riesgo con una etiqueta y color para hacerlo mas intuitivo 
#(los números son los colores, verde, amrillo y azul).
def risk_label(score):
    if score <= 0:
        return "BAJO", (0, 255, 0) #verde
    elif score == 1:
        return "MEDIO", (0, 255, 255) #amarillo
    else:
        return "ALTO", (0, 0, 255)#rojo



#CAMARA

cap = cv2.VideoCapture(0) #Abre la camara con OpencCV

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

prev_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        print("No se pudo leer el frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    current_time = time.time()
    state_text = "ATENTO"
    risk_score = 0

    if results.multi_face_landmarks:
        no_face_start = None
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Dibujar landmarks de forma ligera
        for idx in [NOSE_TIP, LEFT_EYE_OUTER, RIGHT_EYE_OUTER] + LEFT_EYE + RIGHT_EYE:
            x, y = get_point(face_landmarks, idx, w, h)
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        # EAR de ambos ojos
        left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Giro de cabeza
        turn = head_turn_score(face_landmarks, w, h)

        # Estado: ojos cerrados
        if avg_ear < EAR_THRESHOLD:
            if eyes_closed_start is None:
                eyes_closed_start = current_time
            closed_duration = current_time - eyes_closed_start

            if closed_duration >= EYES_CLOSED_TIME:
                state_text = "SOMNOLENCIA DETECTADA"
                risk_score = 2
            else:
                state_text = "PARPADEO / OJOS CERRADOS"
                risk_score = max(risk_score, 1)
        else:
            eyes_closed_start = None

        # Estado: cabeza desviada
        if abs(turn) > HEAD_TURN_THRESHOLD:
            if risk_score < 2:
                state_text = "DISTRACCION DETECTADA"
            risk_score = max(risk_score, 1)

        # Mostrar métricas
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Head turn: {turn:.2f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    else:
        eyes_closed_start = None

        if no_face_start is None:
            no_face_start = current_time

        no_face_duration = current_time - no_face_start

        if no_face_duration >= NO_FACE_TIME:
            state_text = "USUARIO NO PRESENTE"
            risk_score = 2
        else:
            state_text = "BUSCANDO ROSTRO..."
            risk_score = 1

    level_text, level_color = risk_label(risk_score)

    # Cuadro superior con estado
    cv2.rectangle(frame, (10, 100), (620, 180), (30, 30, 30), -1)
    cv2.putText(frame, f"Estado: {state_text}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Riesgo: {level_text}", (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, level_color, 2)

    cv2.imshow("MVP - Fatiga y Distraccion", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


