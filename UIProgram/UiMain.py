# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UiMain.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1250, 600)
        MainWindow.setMinimumSize(QtCore.QSize(1250, 600))
        MainWindow.setMaximumSize(QtCore.QSize(1250, 600))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 110, 791, 481))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(10, 10, 771, 461))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label_show = QtWidgets.QLabel(self.frame_2)
        self.label_show.setGeometry(QtCore.QRect(0, 0, 770, 460))
        self.label_show.setMinimumSize(QtCore.QSize(770, 460))
        self.label_show.setMaximumSize(QtCore.QSize(770, 460))
        self.label_show.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_show.setText("")
        self.label_show.setObjectName("label_show")
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(10, 10, 1221, 91))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.label_3 = QtWidgets.QLabel(self.frame_5)
        self.label_3.setGeometry(QtCore.QRect(420, 20, 411, 51))
        font = QtGui.QFont()
        font.setFamily("华文行楷")
        font.setPointSize(30)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_2 = QtWidgets.QLabel(self.frame_5)
        self.label_2.setGeometry(QtCore.QRect(20, 50, 311, 21))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(14)
        font.setUnderline(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.line = QtWidgets.QFrame(self.frame_5)
        self.line.setGeometry(QtCore.QRect(10, 70, 1211, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(10, 600, 791, 201))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.groupBox = QtWidgets.QGroupBox(self.frame_3)
        self.groupBox.setGeometry(QtCore.QRect(10, 0, 771, 191))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(20)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.PicBtn = QtWidgets.QPushButton(self.groupBox)
        self.PicBtn.setGeometry(QtCore.QRect(100, 30, 201, 71))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(16)
        self.PicBtn.setFont(font)
        self.PicBtn.setStyleSheet("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/ui_imgs/icons/img.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        self.PicBtn.setIcon(icon)
        self.PicBtn.setIconSize(QtCore.QSize(40, 40))
        self.PicBtn.setObjectName("PicBtn")
        self.VideoBtn = QtWidgets.QPushButton(self.groupBox)
        self.VideoBtn.setGeometry(QtCore.QRect(470, 30, 201, 71))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(16)
        self.VideoBtn.setFont(font)
        self.VideoBtn.setStyleSheet("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/ui_imgs/icons/video.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        self.VideoBtn.setIcon(icon1)
        self.VideoBtn.setIconSize(QtCore.QSize(40, 40))
        self.VideoBtn.setObjectName("VideoBtn")
        self.CapBtn = QtWidgets.QPushButton(self.groupBox)
        self.CapBtn.setGeometry(QtCore.QRect(100, 110, 201, 71))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(16)
        self.CapBtn.setFont(font)
        self.CapBtn.setStyleSheet("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/ui_imgs/icons/camera.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        self.CapBtn.setIcon(icon2)
        self.CapBtn.setIconSize(QtCore.QSize(40, 40))
        self.CapBtn.setObjectName("CapBtn")
        self.exitBtn = QtWidgets.QPushButton(self.groupBox)
        self.exitBtn.setGeometry(QtCore.QRect(470, 110, 201, 71))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exitBtn.sizePolicy().hasHeightForWidth())
        self.exitBtn.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(16)
        self.exitBtn.setFont(font)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/ui_imgs/icons/退出.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        self.exitBtn.setIcon(icon3)
        self.exitBtn.setIconSize(QtCore.QSize(40, 40))
        self.exitBtn.setObjectName("exitBtn")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(840, 100, 391, 481))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(20)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.resIcon = QtWidgets.QLabel(self.groupBox_2)
        self.resIcon.setGeometry(QtCore.QRect(100, 240, 200, 200))
        self.resIcon.setText("")
        self.resIcon.setAlignment(QtCore.Qt.AlignCenter)
        self.resIcon.setObjectName("resIcon")
        self.resLb = QtWidgets.QLabel(self.groupBox_2)
        self.resLb.setGeometry(QtCore.QRect(30, 60, 341, 171))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.resLb.sizePolicy().hasHeightForWidth())
        self.resLb.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(28)
        self.resLb.setFont(font)
        self.resLb.setAlignment(QtCore.Qt.AlignCenter)
        self.resLb.setObjectName("resLb")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(20, 440, 131, 51))
        self.label.setObjectName("label")
        self.confLb = QtWidgets.QLabel(self.groupBox_2)
        self.confLb.setGeometry(QtCore.QRect(150, 450, 151, 31))
        self.confLb.setText("")
        self.confLb.setObjectName("confLb")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(920, 610, 251, 91))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(24)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(880, 690, 311, 101))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(18)
        font.setUnderline(False)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Face Sync"))
        self.label_3.setText(_translate("MainWindow", "Face Sync"))
        self.label_2.setText(_translate("MainWindow", ""))
        self.groupBox.setTitle(_translate("MainWindow", "操作"))
        self.PicBtn.setText(_translate("MainWindow", "打開圖片"))
        self.VideoBtn.setText(_translate("MainWindow", "打開視頻"))
        self.CapBtn.setText(_translate("MainWindow", "打開攝像頭"))
        self.exitBtn.setText(_translate("MainWindow", "退出"))
        self.groupBox_2.setTitle(_translate("MainWindow", "識別結果"))
        self.resLb.setText(_translate("MainWindow", "暫無結果"))
        self.label.setText(_translate("MainWindow", "置信度："))
import ui_sources_rc
