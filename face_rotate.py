import cv2
import mediapipe as mp
import numpy as np
import time

left = [ 152, 175, 199, 200, 18, 17, 16, 15, 14, 13, 12, 11, 0 , 164, 2,  94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 10, 109,
          67, 103, 54, 21, 162, 127, 227, 137, 177, 215, 58, 138, 172, 136, 150, 149, 176, 148]
right = [ 152, 175, 199, 200, 18, 17, 16, 15, 14, 13, 12, 11, 0 , 164, 2,  94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 10, 338,
          297, 332, 284, 251, 389, 464, 447, 366, 401, 361, 435, 367, 397, 365, 379, 378, 400, 377]

test = [10,152, 234]
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture("/media/hdd1/UBFC/subject1/vid.avi")

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    pts = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # for idx in left:
            #     x, y = face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y
            #     face_2d.append([ x * img_w , y * img_h])
            # mask = np.zeros((img_h, img_w), dtype=np.uint8)
            # pts = np.array(face_2d, dtype=np.uint8)
            #
            # cv2.drawContours(mask, [pts.astype(int)], -1, (255, 255, 255), -1, cv2.LINE_AA)
            # dst = cv2.bitwise_and(image, image, mask=mask)

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in test:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    pts.append([x, y])

                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x,y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x,y])

                    face_3d.append([x,y,lm.z])

            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            pts = np.array(pts, dtype=np.uint8)
            cv2.fillPoly(mask, pts.astype(int), (255))
            # cv2.drawContours(mask, [pts.astype(int)], -1, (255, 255, 255), -1, cv2.LINE_AA)
            dst = cv2.bitwise_and(image, image, mask=mask)

            face_2d = np.array(face_2d,dtype=np.float32)

            face_3d = np.array(face_3d,dtype=np.float32)

            face_length = 1 * img_w

            cam_matrix = np.array([[face_length, 0, img_w/2],
                                   [0, face_length, img_h/2],
                                   [0, 0, 1]],dtype=np.float32)

            dist_matrix = np.zeros((4,1), dtype=np.float32)

            success, rotation_vector, translation_vector = cv2.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix,flags = 0)

            rmat, jac = cv2.Rodrigues(rotation_vector)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # y < -10 left
            # y > 10 right
            # x < -10 down
            # x > 10 up

            nose_3d_projection, jacobian = cv2.projectPoints(np.array([nose_3d], dtype=np.float64), rotation_vector, translation_vector, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (0, 255, 0), 3)

            cv2.putText(image, f"X: {x}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Y: {y}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Z: {z}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            print("FPS: ", fps)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)


    # cv2.imshow('MediaPipe FaceMesh', image)
    # if cv2.waitKey(5) & 0xFF == 27:
    #     break


# left = [ 152, 175, 199, 200, 18, 17, 16, 15, 14, 13, 12, 11, 0 , 164, 2,  94, 19, 1, 4, 5, 1945, 197, 6, 168, 8, 9, 151, 10, 109,
#           67, 103, 54, 21, 162, 127, 227, 137, 177, 215, 58, 138, 172, 136, 150, 149, 176, 148]
# right = [ 152, 175, 199, 200, 18, 17, 16, 15, 14, 13, 12, 11, 0 , 164, 2,  94, 19, 1, 4, 5, 1945, 197, 6, 168, 8, 9, 151, 10, 338,
#           297, 332, 284, 251, 389, 464, 447, 366, 401, 361, 435, 367, 397, 365, 379, 378, 400, 377, 152]