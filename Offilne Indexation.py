#!/usr/bin/env python3
import cv2
import numpy as np
import json
import re
from pathlib import Path

# Pre-processing functions


def gamma_correction(img, gamma=0.2):
    lut = np.power(np.arange(256) / 255.0, gamma) * 255.0
    return cv2.LUT(img, lut.astype(np.float32))


def difference_of_gaussians(img):
    img_f = img.astype(np.float32)
    g1 = cv2.GaussianBlur(img_f, (5, 5), 1.0)
    g2 = cv2.GaussianBlur(img_f, (9, 9), 2.0)
    return g1 - g2



def contrast_equalization(img):
    # FIX: correct Tan & Triggs implementation
    abs_img = np.abs(img)
    power_img = np.power(abs_img + 1e-6, 0.1)



    m = np.mean(power_img)
    if m > 0:
        img_eq = img / m
    else:
        img_eq = img

    img_clip = np.clip(img_eq, -10, 10)
    img_norm = ((img_clip + 10) / 20.0) * 255.0

    return np.clip(img_norm, 0, 255).astype(np.uint8)


def apply_mask(img, size=5):
    img2 = img.copy()
    h, w = img2.shape

    img2[:size, :] = 0
    img2[-size:, :] = 0
    img2[:, :size] = 0
    img2[:, -size:] = 0

    return img2


def get_ltp(img):
    h, w = img.shape
    t = 5
    upper = np.zeros((h, w), dtype=np.uint8)
    lower = np.zeros((h, w), dtype=np.uint8)

    val = [1, 2, 4, 8, 16, 32, 64, 128]
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1),(0, -1)]

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            c = int(img[i, j])

            u_val = 0
            l_val = 0

            for k in range(8):
                dy, dx = offsets[k]
                p = int(img[i+dy, j+dx])
                diff = p - c

                if diff >= t:
                    u_val += val[k]
                if diff <= -t:
                    l_val += val[k]

            upper[i, j] = u_val
            lower[i, j] = l_val

    h1, _ = np.histogram(upper, bins=256, range=(0, 256))
    h2, _ = np.histogram(lower, bins=256, range=(0, 256))

    h1 = h1.astype(float) / (np.sum(h1) + 1e-7)
    h2 = h2.astype(float) / (np.sum(h2) + 1e-7)

    return np.concatenate([h1, h2])


#   Processing Loop

current_folder = Path(__file__).parent
raw_path = current_folder / "Faces"
out_path = current_folder / "Faces_preprocessed"
json_path = current_folder / "index.json"

file_pattern = re.compile(r"^(\d+)\.pgms(\d+)\.jpg$", re.IGNORECASE)

if not out_path.exists():
    out_path.mkdir()

for i in range(1, 41):
    p_folder = out_path / ("person" + str(i))
    if not p_folder.exists():
        p_folder.mkdir()

images = sorted(list(raw_path.glob("*.jpg")))
db = {}

print("Pre Processing :...")

for f in images:
    name = f.name
    match = file_pattern.match(name)

    if match:
        person_id = int(match.group(2))
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)

        if img is not None:

            img = gamma_correction(img)
            img = difference_of_gaussians(img)
            img = apply_mask(img)
            img = contrast_equalization(img)



            save_folder = out_path / ("person" + str(person_id))
            save_path = save_folder / name
            cv2.imwrite(str(save_path), img)

            feat = get_ltp(img)
            key = "person" + str(person_id) + "/" + name
            db[key] = feat.tolist()

            if len(db) % 40 == 0:
                print("Processed " + str(len(db)) + " images...")


print("Saving json File")

with open(json_path, 'w') as file:
    file.write("{\n")
    keys = list(db.keys())

    for i in range(len(keys)):
        k = keys[i]
        v = db[k]

        line = f'  "{k}": {json.dumps(v)}'
        line += ",\n" if i < len(keys) - 1 else "\n"
        file.write(line)

    file.write("}\n")

print("Works !!!")
