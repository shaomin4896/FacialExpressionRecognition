"""
基于深度学习的人脸表情识别系统
作者：阿旭
公众号：阿旭算法与机器学习
公众号简介：已分享大量机器学习与深度学习实战案例，欢迎关注--专注于python、机器学习与人工智能相关技术分享。
CSDN博客:https://blog.csdn.net/qq_42589613?type=blog
"""

# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QApplication , QMainWindow, QFileDialog
import sys
import os
import datetime
sys.path.append('UIProgram')
from UIProgram.UiMain import Ui_MainWindow
import sys
from PyQt5.QtCore import QTimer, Qt, QCoreApplication
import detect_tools as tools
import cv2
import Config
from PyQt5.QtGui import QPixmap
from UIProgram.QssLoader import QSSLoader
from DenseNet121 import *

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initMain()
        self.signalconnect()
        # 加载css渲染效果
        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)
        self.attended = set()
        cred = credentials.Certificate("nchu7716-firebase-adminsdk-ko2gm-91ede7329f.json")
        firebase_admin.initialize_app(cred)

    def initMain(self):
        # 加载模型
        self.labeldict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Normal'}
        self.labelchinese = {0: '生气', 1: '厌恶', 2: '害怕', 3: '高兴', 4: '伤心', 5: '惊讶', 6: '平淡'}
        inputs = keras.Input(shape=(48, 48, 1), batch_size=64)
        x = create_dense_net(7, inputs, include_top=True, depth=121, nb_dense_block=4, growth_rate=16, nb_filter=-1,
                             nb_layers_per_block=[6, 12, 32, 32], bottleneck=True, reduction=0.5, dropout_rate=0.2,
                             activation='softmax')
        self.model = tf.keras.Model(inputs, x, name='densenet121')
        filepath = Config.model_path
        self.model.load_weights(filepath)
        self.model.predict(np.zeros((1,48,48,1)))

        self.show_width = 770
        self.show_height = 460

        self.org_path = None

        self.is_camera_open = False
        self.cap = None

        # 更新视频图像
        self.timer_camera = QTimer()

        # 设置主页背景图片border-image: url(:/icons/ui_imgs/icons/camera.png)
        # self.setStyleSheet("#MainWindow{background-image:url(:/bgs/ui_imgs/bg3.jpg)}")

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img)
        self.ui.VideoBtn.clicked.connect(self.vedio_show)
        self.ui.CapBtn.clicked.connect(self.camera_show)
        self.ui.exitBtn.clicked.connect(QCoreApplication.quit)

    def save_punch_record_to_firebase(self, record: dict) -> None:

        db = firestore.client()
        doc_ref = db.collection("records")
        doc_ref.add(record)
    
    def open_img(self):
        if self.cap:
            self.video_stop()
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')
            self.cap = None

        # 弹出的窗口名称：'打开图片'
        # 默认打开的目录：'./'
        # 只能打开.jpg与.gif结尾的图片文件
        # file_path, _ = QFileDialog.getOpenFileName(self.ui.centralwidget, '打开图片', './', "Image files (*.jpg *.gif)")
        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', "Image files (*.jpg *.jepg *.png)")
        if not file_path:
            return

        self.org_path = file_path
        self.cv_img = tools.img_cvread(self.org_path)
        face_cvimg, faces, locations = face_detect(self.cv_img)
        if faces is not None:
            for i in range(len(faces)):
                face_infomation = face_recognize(face_cvimg)
                top, right, bottom, left = locations[i]
                face = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                face = face / 255.0
                conf_res = self.model.predict(np.reshape(face, (-1, 48, 48, 1)))
                print(np.reshape(face, (-1, 48, 48, 1)).shape)
                num = np.argmax(conf_res)
                label = self.labeldict[num]
                face_cvimg = cv2.putText(face_cvimg, label, (left, top-10), cv2.FONT_ITALIC, 0.8, (0, 0, 250), 2,
                                    cv2.LINE_AA)
                # print('人物表情{}：'.format(i + 1) + self.labelchinese[num])
                self.ui.resLb.setText('查無此員工')
                if face_infomation is not None:
                    emp_id = face_infomation["emp_id"]
                    emp_name = face_infomation["emp_name"]
                    emp_dept = face_infomation["emp_dept"]
                    emotion = label
                    punch_time = datetime.datetime.now()
                    msg = f"部門：{emp_dept}\n員編：{emp_id}\n姓名：{emp_name}\n{punch_time.strftime('%m/%d %H:%M:%S')} 打卡上班！\n表情：{emotion}"
                    self.ui.resLb.setText(msg)
                    self.save_punch_record_to_firebase({
                        "emp_id": emp_id,
                        "emp_name": emp_name,
                        "emp_dept": emp_dept,
                        "emotion": emotion,
                        "punch_time": punch_time
                    })
                # self.ui.resLb.setText(self.labeldict[num] + '--'+ self.labelchinese[num])
                icon_name = self.labeldict[num] + '.png'
                icon_path = os.path.join('UIProgram/ui_imgs', icon_name)
                pix = QPixmap(icon_path)
                self.ui.resIcon.setPixmap(pix)
                self.ui.resIcon.setScaledContents(True)
                max_conf = max(conf_res[0]) * 100
                self.ui.confLb.setText('{:.2f}%'.format(max_conf))

        self.img_width, self.img_height = self.get_resize_size(face_cvimg)
        resize_cvimg = cv2.resize(face_cvimg,(self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Image files (*.avi *.mp4)")
        if not file_path:
            return None
        self.org_path = file_path
        return file_path

    def video_start(self):
        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(30)
        self.timer_camera.timeout.connect(self.open_frame)


    def video_stop(self):
        self.is_camera_open = False
        if self.cap is not None:
            self.cap.release()
        self.timer_camera.stop()
        self.ui.label_show.clear()

    def open_frame(self):
        ret, image = self.cap.read()
        if ret:
            face_cvimg, faces, locations = face_detect(image)
            if faces is not None:
                for i in range(len(faces)):
                    face_infomation = face_recognize(face_cvimg)
                    top, right, bottom, left = locations[i]
                    face = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (48, 48))
                    face = face / 255.0
                    conf_res = self.model.predict(np.reshape(face, (-1, 48, 48, 1)))
                    num = np.argmax(conf_res)
                    label = self.labeldict[num]
                    face_cvimg = cv2.putText(face_cvimg, label, (left, top-10), cv2.FONT_ITALIC, 0.8, (0, 0, 250), 2,
                                             cv2.LINE_AA)
                    self.ui.resLb.setText('查無此員工')
                    if face_infomation is not None:
                        emp_id = face_infomation["emp_id"]
                        emp_name = face_infomation["emp_name"]
                        emp_dept = face_infomation["emp_dept"]
                        emotion = label
                        punch_time = datetime.datetime.now()
                        if emp_name not in self.attended:
                            self.attended.add(emp_name)
                            self.save_punch_record_to_firebase({
                            "emp_id": emp_id,
                            "emp_name": emp_name,
                            "emp_dept": emp_dept,
                            "emotion": emotion,
                            "punch_time": punch_time
                            })
                        msg = f"部門：{emp_dept}\n員編：{emp_id}\n姓名：{emp_name}\n{punch_time.strftime('%m/%d %H:%M:%S')} 打卡上班！\n表情：{emotion}"
                        self.ui.resLb.setText(msg)
                        
                    # self.ui.resLb.setText(self.labeldict[num] + '--'+ self.labelchinese[num])
                    icon_name = self.labeldict[num] + '.png'
                    icon_path = os.path.join('UIProgram/ui_imgs', icon_name)
                    pix = QPixmap(icon_path)
                    self.ui.resIcon.setPixmap(pix)
                    self.ui.resIcon.setScaledContents(True)
                    max_conf = max(conf_res[0]) * 100
                    self.ui.confLb.setText('{:.2f}%'.format(max_conf))

            self.img_width, self.img_height = self.get_resize_size(face_cvimg)
            resize_cvimg = cv2.resize(face_cvimg, (self.img_width, self.img_height))

            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)
        else:
            self.cap.release()
            self.timer_camera.stop()

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.CapBtn.setText('关闭摄像头')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
        else:
            self.ui.CapBtn.setText('打开摄像头')
            self.ui.label_show.setText('')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.ui.label_show.clear()

    def get_resize_size(self, img):
        _img = img.copy()
        img_height, img_width , depth= _img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    win.ui.CapBtn.click()
    sys.exit(app.exec_())

