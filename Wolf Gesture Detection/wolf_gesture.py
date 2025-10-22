import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

def mesafe(p1, p2):
    return ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**0.5

def bozkurt_mi(el):
    i = mesafe(el.landmark[8], el.landmark[5]) > 0.15
    s = mesafe(el.landmark[20], el.landmark[17]) > 0.12
    o = mesafe(el.landmark[12], el.landmark[9]) < 0.12
    y = mesafe(el.landmark[16], el.landmark[13]) < 0.12
    return i and s and o and y

kamera = cv2.VideoCapture(0)

while True:
    _, goruntu = kamera.read()
    goruntu = cv2.flip(goruntu, 1)
    sonuc = hands.process(cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB))
    
    if sonuc.multi_hand_landmarks:
        for el in sonuc.multi_hand_landmarks:
            mp_drawing.draw_landmarks(goruntu, el, mp_hands.HAND_CONNECTIONS)
            
            if bozkurt_mi(el):
                cv2.putText(goruntu, "AUUUU!", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
    
    cv2.imshow('Bozkurt', goruntu)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()