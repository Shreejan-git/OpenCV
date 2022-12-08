import mediapipe as mp
import cv2

img = cv2.imread('images/ash2.jpg')
img = cv2.imread('images/twogirls.png')

# img = cv2.imread('images/abhishek3.jpg')


image = img.copy()

mp_face_mesh = mp.solutions.face_mesh


face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
left_eye_landmark = mp_face_mesh.FACEMESH_LEFT_EYE

results = face_mesh.process(image)


print(len(results.multi_face_landmarks))

for landmarks in results.multi_face_landmarks:
    # print(landmarks)
    for eachxy in landmarks.landmark:
        # print(eachxy)
        x = eachxy.x
        y = eachxy.y
        
        relative_x = int(image.shape[1] * x)
        relative_y = int(image.shape[0] * y)
        
        cv2.circle(image, (relative_x, relative_y), 1, (0,0,255),-1)
        
cv2.imshow('image', image)
cv2.waitKey()

# print(left_eye_landmark)

# if results.multi_face_landmarks:
#     for landmarks in results.multi_face_landmarks:
#         for source_idx, target_idx in mp_face_mesh.FACEMESH_LEFT_EYE:
#             source = landmarks.landmark[source_idx]
#             target = landmarks.landmark[target_idx]
            
#             relative_source = (int(source.x * image.shape[1]),int(source.y * image.shape[0]))
#             relative_target = (int(target.x * image.shape[1]),int(target.y * image.shape[0]))
            
            
#             cv2.line(image, relative_source, relative_target, color=(255,0,0), thickness=1)
#             cv2.imshow('image', image)
#             cv2.waitKey()
            
            
# else:
#     print('afadfdasff')
