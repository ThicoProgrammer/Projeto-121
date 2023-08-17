import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Desenhar círculos nas pontas dos dedos
            for tip_id in finger_tips:
                x, y = int(lm_list[tip_id].x * w), int(lm_list[tip_id].y * h)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

            # Verificar se os dedos estão dobrados ou estendidos
            finger_fold_status = []
            for i in range(1, len(finger_tips)):
                if lm_list[finger_tips[i]].x < lm_list[finger_tips[i] - 3].x:
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # Verificar o polegar (dedo 4) se está para cima ou para baixo
            if lm_list[thumb_tip].y < lm_list[thumb_tip - 2].y:
                cv2.putText(img, "CURTI", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "NAO CURTI", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("detector de maos", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()