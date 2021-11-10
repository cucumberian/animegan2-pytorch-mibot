from python:3.9

run mkdir -p /usr/src/app/

workdir /usr/src/app/

copy . /usr/src/app/

run apt-get install gcc

run wget --no-verbose -O face_paint_512_v2_0.pt https://drive.google.com/uc?id=18H3iK09_d54qEDoWIc82SyWB2xun4gjU
run wget --no-verbose http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
run bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

run python3 -m pip install --upgrade pip
run pip3 install CMake
run pip3 install dlib
run pip3 install -r requirements.txt

cmd ["python3", "app.py"]
