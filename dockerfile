from cucumber/alpine-python3-pip

run mkdir -p /usr/src/app/

workdir /usr/src/app/

copy ./usr/src/app/

run pip3 install --no-cache-dir -r requirements.txt

cmd ["python3", "app.py"]