from symtable import Symbol
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 이미지 파일의 경우을 사용하세요.:
IMAGE_FILES = ["test_image.jpg"]

# 표현되는 랜드마크의 굵기와 반경
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
mean = 0
oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109]
cheek_left = [123, 50, 36, 137, 205, 206, 177, 147, 187, 207, 213, 216, 215, 192, 138,
        214, 212, 135]
cheek_right = [266, 280, 352, 366, 425, 426, 411, 427, 376, 401, 436, 433, 435, 416,
        434, 367, 364, 432]

face_whole = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109, 123, 50, 36, 137, 205, 206, 177, 147, 187, 207, 213, 216, 215, 192, 138,
        214, 212, 135, 266, 280, 352, 366, 425, 426, 411, 427, 376, 401, 436, 433, 435, 416,
        434, 367, 364, 432]

x_list = np.linspace(0, 0, len(face_whole))
y_list = np.linspace(0, 0, len(face_whole))
z_list = np.linspace(0, 0, len(face_whole))

with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        # 얼굴부분 crop 
        # haarcascade 불러오기
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 이미지 불러오기
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 얼굴 찾기        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped = image[y: y+h+100, x: x+w]
            resize = cv2.resize(cropped, (800, 900))
        image = resize
        
        # 작업 전에 BGR 이미지를 RGB로 변환합니다.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        ih, iw, ic = annotated_image.shape
        for face_landmarks in results.multi_face_landmarks:

            # 각 랜드마크를 image에 overlay 시켜줌
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec)
                # connection_drawing_spec=mp_drawing_styles     <---- 이 부분, 눈썹과 눈, 오른쪽 왼쪽 색깔(초록색, 빨강색)
                # .get_default_face_mesh_contours_style())


       # 랜드마크의 좌표 정보 확인
        for id, lm in enumerate(face_landmarks.landmark):
                ih, iw, ic = annotated_image.shape
                x,y = int(lm.x*iw),int(lm.y*ih)
                # print(id,x,y)
                #print(face_landmarks.landmark[id].x, face_landmarks.landmark[id].y, face_landmarks.landmark[id].z)
                if id == 94 : # 코 끝
                        cv2.putText(annotated_image,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
                elif id == 17 : # 아래 입술 끝
                        cv2.putText(annotated_image,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
                elif id == 152 : # 턱 끝
                        cv2.putText(annotated_image,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
                

       
        ## 얼굴 비율 측정 2, 하안부 중 긴 인중 판단
        nose2lip_x = face_landmarks.landmark[94].x - face_landmarks.landmark[17].x
        nose2lip_y = face_landmarks.landmark[94].y - face_landmarks.landmark[17].y
        lip2chin_x = face_landmarks.landmark[17].x - face_landmarks.landmark[152].x
        lip2chin_y = face_landmarks.landmark[17].y - face_landmarks.landmark[152].y
        
        # Nose to Under-Lip length
        NtL_len = np.sqrt(nose2lip_x**2 + nose2lip_y**2)

        # Under-Lip to Chin length
        LtC_len = np.sqrt(lip2chin_x**2 + lip2chin_y**2)

        length_ratio = (NtL_len*0.8)/LtC_len

        
        print("Nose to Under-Lip : ", NtL_len)
        print("Under-Lip to Chin : ", LtC_len)
        print("length ratio : ", length_ratio)


      
        coodinate_list = np.array([x_list, y_list, z_list])
        #print(coodinate_list)

        coodinate_list = coodinate_list.reshape((1, -1))
        #print(coodinate_list)

        coodinate_list = coodinate_list.reshape((3, -1))
        #print(coodinate_list)

        cv2.imshow("Image_ESEntial",annotated_image)
       
        # esc 입력시 종료
        key = cv2.waitKey(50000)
        if key == 27:
            break