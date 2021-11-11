import os
model_fname = "face_paint_512_v2_0.pt"
if not os.path.isfile(model_fname):
    os.system(f"wget -O {model_fname} https://drive.google.com/uc?id=18H3iK09_d54qEDoWIc82SyWB2xun4gjU")

import sys
sys.path.append("animegan2-pytorch")

import torch
#torch.set_grab_enabled(False)

from model import Generator

device = "cpu"

model = Generator().eval().to(device)
model.load_state_dict(torch.load(model_fname))

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

def face2paint(
    img: Image.Image,
    size: int,
    side_by_side: bool = True,
) -> Image.Image:

    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.LANCZOS)

    input = to_tensor(img).unsqueeze(0) * 2 - 1
    output = model(input.to(device)).cpu()[0]

    if side_by_side:
        output = torch.cat([input[0], output], dim=2)

    output = (output * 0.5 + 0.5).clip(0, 1)

    return to_pil_image(output)



#@title Face Detector & FFHQ-style Alignment

# https://github.com/woctezuma/stylegan2-projecting-images

import os
import dlib
import collections
from typing import Union, List
import numpy as np
from PIL import Image


def get_dlib_face_detector(predictor_path: str = "shape_predictor_68_face_landmarks.dat"):

    if not os.path.isfile(predictor_path):
        model_file = "shape_predictor_68_face_landmarks.dat.bz2"
        os.system(f"wget http://dlib.net/files/{model_file}")
        os.system(f"bzip2 -d {model_file}")

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)

    def detect_face_landmarks(img: Union[Image.Image, np.ndarray]):
        if isinstance(img, Image.Image):
            img = np.array(img)
        faces = []
        dets = detector(img)
        for d in dets:
            shape = shape_predictor(img, d)
            faces.append(np.array([[v.x, v.y] for v in shape.parts()]))
        return faces
    
    return detect_face_landmarks

# https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage

def align_and_crop_face(
    img: Image.Image,
    landmarks: np.ndarray,
    expand: float = 1.0,
    output_size: int = 1024, 
    transform_size: int = 4096,
    enable_padding: bool = True,
):
    # Parse landmarks.
    # pylint: disable=unused-variable
    lm = landmarks
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= expand
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.4
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    return img


import telebot
import os
TELEGRAM_API_KEY = os.environ.get("TEL_API_KEY") or "YOU_NEED_AN_API_KEY_FOR_TELEGRAM_BOT"
IMAGE_FOLDER = "./tmp"

bot = telebot.TeleBot(TELEGRAM_API_KEY)

print('bot started')

@bot.message_handler(commands=["start"])
def get_start_message(message):
    bot.send_message(
        chat_id=message.chat.id,
        text=f"Send portrait to a bot, {message.from_user.first_name}."
    )

@bot.message_handler(commands=["help"])
def show_help(message):
    help_str = '''
Based on Google Collab: 
https://colab.research.google.com/drive/1jCqcKekdtKzW7cxiw_bjbbfLsPh-dEds?usp=sharing
github:
https://github.com/bryandlee/animegan2-pytorch
    '''
    bot.send_message(message.chat.id, help_str)


@bot.message_handler(content_types=["photo"])
def get_response_to_photo(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id=file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    file_name = os.path.join(IMAGE_FOLDER, f"{file_id}.png")

    if not os.path.exists(IMAGE_FOLDER):
        os.mkdir(IMAGE_FOLDER)

    with open(file=file_name, mode='wb') as f:
        f.write(downloaded_file)
    
    input_image = Image.open(file_name)

    input_image = input_image.convert('RGB')
    os.remove(file_name)
    output_image_1 = face2paint(img=input_image, size=512, side_by_side=False)
    output_filename, ext = os.path.splitext(os.path.basename(file_name))
    output_filename_1 = os.path.join(IMAGE_FOLDER, f"{output_filename}_out_1.png")
    output_image_1.save(output_filename_1)

    with open(file=output_filename_1, mode='rb') as f:
        bot.send_photo(message.chat.id, f)
    os.remove(output_filename_1)

    output_filename_2 = os.path.join(IMAGE_FOLDER, f"{output_filename}_out_2.png")
    face_detector = get_dlib_face_detector()
    landmarks = face_detector(input_image)
    for landmark in landmarks:
        face = align_and_crop_face(input_image, landmark, expand=1.3)
        face2paint(img=face, size=512, side_by_side=False).save(output_filename_2)
        
        with open(file=output_filename_2, mode='rb') as f:
            bot.send_photo(message.chat.id, f)
        os.remove(output_filename_2)


@bot.message_handler(content_types=["text"])
def get_any_response(message):
    bot.send_message(
        chat_id=message.chat.id,
        text=message.text
    )


bot.polling(none_stop=True, interval=2, timeout=20)
