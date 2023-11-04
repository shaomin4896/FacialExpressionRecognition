# coding:utf-8
import cv2
import os
from DenseNet121 import *
labeldict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprised', 6:'Normal'}
labelchinese = {0:'生气', 1:'厌恶', 2:'害怕', 3:'高兴', 4:'伤心', 5:'惊讶', 6:'平淡'}
inputs = keras.Input(shape=(48, 48, 1), batch_size=64)
x = create_dense_net(7, inputs, include_top=True, depth=121, nb_dense_block=4, growth_rate=16, nb_filter=-1,
                     nb_layers_per_block=[6, 12, 32, 32], bottleneck=True, reduction=0.5, dropout_rate=0.2,
                     activation='softmax')
model = tf.keras.Model(inputs, x, name='densenet121')
filepath = 'DenseNet121.h5'
model.load_weights(filepath)
# VideoCapture方法是cv2库提供的读取视频方法
cap = cv2.VideoCapture(0)
# 设置需要保存视频的格式“xvid”
# 该参数是MPEG-4编码类型，文件名后缀为.avi
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 设置视频帧频
fps = cap.get(cv2.CAP_PROP_FPS)
# 设置视频大小
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# VideoWriter方法是cv2库提供的保存视频方法
# 按照设置的格式来out输出
# save_video_name = 'video_test.avi'
# save_video_path = os.path.join('', save_video_name)
# out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

# 确定视频打开并循环读取
while (cap.isOpened()):
    # 逐帧读取，ret返回布尔值
    # 参数ret为True 或者False,代表有没有读取到图片
    # frame表示截取到一帧的图片
    ret, frame = cap.read()
    if ret == True:
        frame, faces, locations = face_detect(frame)
        if faces is not None:
            for i in range(len(faces)):
                top, right, bottom, left = locations[i]
                face = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                face = face / 255.0
                print(model.predict(np.reshape(face, (-1, 48, 48, 1))))
                num = np.argmax(model.predict(np.reshape(face, (-1, 48, 48, 1))))
                label = labeldict[num]
                frame = cv2.putText(frame, label, (left, top), cv2.FONT_ITALIC, 0.8, (0, 0, 250), 2,
                                    cv2.LINE_AA)
                print('人物表情{}：'.format(i + 1) + labelchinese[num])

        cv2.imshow('video_show', frame)
        # out.write(face_cvimg)  # 保存检测视频
        key = cv2.waitKey(10)  # 等待一段时间，并且检测键盘输入
        if key == ord('q'):  # 若是键盘输入'q',则退出，释放视频
            break
    else:
        break

# 释放资源
cap.release()
# out.release()