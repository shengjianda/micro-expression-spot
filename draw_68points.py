import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import crop_face
# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/python_code/shape_predictor_68_face_landmarks.dat')

# 读取图像文件
# img_rd = cv2.imread("D:/dataset/micro_datatset/SAMM_longvideos/SAMM_longvideos/009_4/009_4_0007.jpg")
img_rd = cv2.imread("D:/dataset/micro_datatset/casme2/sub01/EP04_03/img1.jpg")
img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
landmark0, img_rd2, frame_shang, frame_xia, frame_left, frame_right = crop_face.crop_picture(img_rd,256)
cv2.rectangle(img_rd, (frame_left, frame_shang), (frame_right, frame_xia), (0, 255, 0), 1)






print(img_gray.shape)
# 人脸数
faces = detector(img_gray, 0)
print(faces)
# 待会要写的字体
font = cv2.FONT_HERSHEY_SIMPLEX

# 标 68 个点
if len(faces) != 0:
    # 检测到人脸
    for i in range(len(faces)):
        # 取特征点坐标
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68 点的坐标
            pos = (point[0, 0], point[0, 1])
            if(idx==30):
                cv2.rectangle(img_rd, (point[0, 0]-10, point[0, 1]-10), (point[0, 0] + 10, point[0, 1]+10),
                              (255, 0, 255), 2)
            # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
            cv2.circle(img_rd, pos, 2, color=(139, 156, 0))
            # 利用 cv2.putText 写数字 1-68
            cv2.putText(img_rd, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img_rd, "faces: " + str(52), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
else:
    # 没有检测到人脸
    cv2.putText(img_rd, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

# 窗口显示
# 参数取 0 可以拖动缩放窗口，为 1 不可以
# cv2.namedWindow("image", 0)
cv2.namedWindow("image", 1)

cv2.imshow("image", img_rd)
path33="D:/dataset/micro_datatset/imageforpaper/img1.jpg"
cv2.imwrite(path33,img_rd)
cv2.waitKey(0)