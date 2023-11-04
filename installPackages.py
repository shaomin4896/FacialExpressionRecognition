import pip
import os
with open("requirements.txt") as f:
    for line in f:
        # call pip's main function with each requirement
        pip.main(['install','-U', line,'-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'])

os.system("pip install libs/dlib-19.19.0-cp38-cp38-win_amd64.whl")
os.system("pip install libs/face_recognition-1.3.0-py2.py3-none-any.whl")