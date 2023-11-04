#coding:utf-8
import cv2
from tensorflow import keras
from DenseNet121 import *

# 照片路径
image_path = 'TestImages/2.jpg'

labeldict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprised', 6:'Normal'}
labelchinese = {0:'生气', 1:'厌恶', 2:'害怕', 3:'高兴', 4:'伤心', 5:'惊讶', 6:'平淡'}
inputs = keras.Input(shape=(48, 48, 1), batch_size=64)
x = create_dense_net(7, inputs, include_top=True, depth=121, nb_dense_block=4, growth_rate=16, nb_filter=-1,
                     nb_layers_per_block=[6, 12, 32, 32], bottleneck=True, reduction=0.5, dropout_rate=0.2,
                     activation='softmax')
model = tf.keras.Model(inputs, x, name='densenet121')
filepath = 'models/DenseNet121.h5'
model.load_weights(filepath)

image = cv2.imread(image_path)
frame, faces, locations = face_detect(image)
if faces is not None:
    for i in range(len(faces)):
        top, right, bottom, left = locations[i]
        face = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        num = np.argmax(model.predict(np.reshape(face, (-1, 48, 48, 1))))
        label = labeldict[num]
        frame = cv2.putText(frame, label, (left, top-10), cv2.FONT_ITALIC, 0.8, (0, 0, 250), 2,
                            cv2.LINE_AA)
        print('人物表情{}：'.format(i+1) + labelchinese[num])

cv2.imshow('frame',frame)
cv2.waitKey(0)


