#!/usr/bin/env python3
import cv2
import numpy as np
import json
import re
import pandas as pd
from collections import Counter
from pathlib import Path

INDEX_PATH = 'index.json'
DATASET_ROOT = 'Faces'
OUTPUT_FILE = 'online_evaluation_results.xlsx'

GAMMA_VALUE = 0.2
DOG_SIGMA1 = 1.0
DOG_SIGMA2 = 2.0
CONTRAST_ALPHA = 0.1
CONTRAST_TAU = 10.0
LTP_THRESHOLD = 5

def gamma_correction(img, gamma=0.2):
    lut = np.power(np.arange(256) / 255.0, gamma) * 255.0
    return cv2.LUT(img, lut.astype(np.float32))

def difference_of_gaussians(img):
    img_f = img.astype(np.float32)
    blur1 = cv2.GaussianBlur(img_f, (5, 5), sigmaX=1.0, sigmaY=1.0)
    blur2 = cv2.GaussianBlur(img_f, (9, 9), sigmaX=2.0, sigmaY=2.0)
    return blur1 - blur2

def contrast_equalization(img):
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

def apply_mask(img, border_size=5):
    img_masked = img.copy()
    img_masked[:border_size, :] = 0
    img_masked[-border_size:, :] = 0
    img_masked[:, :border_size] = 0
    img_masked[:, -border_size:] = 0
    return img_masked

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = gamma_correction(img, gamma=0.2)
    img = difference_of_gaussians(img)
    img = apply_mask(img, border_size=5)
    img = contrast_equalization(img)
    return img

def get_ltp_features(image):
    height, width = image.shape
    t = 5

    upper = np.zeros((height, width), dtype=np.uint8)
    lower = np.zeros((height, width), dtype=np.uint8)

    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center_pixel = int(image[y, x])
            upper_code = 0
            lower_code = 0

            for i, (dy, dx) in enumerate(neighbors_offsets):
                neighbor_pixel = int(image[y + dy, x + dx])
                diff = neighbor_pixel - center_pixel

                if diff >= t:
                    upper_code += powers_of_2[i]
                if diff <= -t:
                    lower_code += powers_of_2[i]

            upper[y, x] = upper_code
            lower[y, x] = lower_code

    upper_hist, _ = np.histogram(upper, bins=256, range=(0, 256))
    lower_hist, _ = np.histogram(lower, bins=256, range=(0, 256))

    if np.sum(upper_hist) > 0:
        upper_hist = upper_hist.astype(np.float32) / np.sum(upper_hist)
    if np.sum(lower_hist) > 0:
        lower_hist = lower_hist.astype(np.float32) / np.sum(lower_hist)

    return np.concatenate([upper_hist, lower_hist])

def chi_square_distance(h1, h2):
    h1 = np.array(h1)
    h2 = np.array(h2)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = h1 + h2
        numer = (h1 - h2) ** 2
        terms = np.divide(numer, denom, where=denom!=0)
    return 0.5 * np.sum(terms)

def get_identity_from_path(path):
    match = re.search(r'person(\d+)', path)
    if match:
        return f"person{match.group(1)}"
    match = re.search(r's(\d+)', path)
    if match:
        return f"s{match.group(1)}"
    return "unknown"

def knn_classify(query_hist, index_db, k_values=[3, 5, 7]):
    distances = []
    for fpath, db_hist in index_db.items():
        dist = chi_square_distance(query_hist, db_hist)
        identity = get_identity_from_path(fpath)
        distances.append({'path': fpath, 'identity': identity, 'dist': dist})

    distances.sort(key=lambda x: x['dist'])

    results = {}
    for k in k_values:
        top_k = distances[:k]
        votes = [x['identity'] for x in top_k]

        counts = Counter(votes)
        if not counts:
            results[k] = "None"
            continue

        winner = counts.most_common(1)
        results[k] = winner[0][0]

    return results

def main():
    print(">>> Starting Online Module Evaluation")

    print(f"Loading Index from {INDEX_PATH}...")
    try:
        with open(INDEX_PATH, 'r') as f:
            index_db = json.load(f)
    except FileNotFoundError:
        print("ERROR: index.json not found. Run Offline Module first.")
        return

    target_subjects = ['person1', 'person2', 'person3', 'person4', 'person5']
    image_numbers = [8, 9, 10]

    evaluation_log = []

    print(f"Processing Test Set: {len(target_subjects)} subjects x {len(image_numbers)} images")

    for subj in target_subjects:
        match = re.search(r'person(\d+)', subj)
        if match:
            person_num = match.group(1)
        else:
            person_num = subj.replace('s', '')

        for img_num in image_numbers:
            img_name = f"{img_num}.pgms{person_num}.jpg"
            query_path = Path(DATASET_ROOT) / img_name
            true_id = subj

            print(f"Querying: {query_path} (True ID: {true_id})")

            try:
                processed_img = preprocess_image(str(query_path))
                if processed_img is None:
                    print(f"FAILED to process {query_path}: Could not load image")
                    continue

                features = get_ltp_features(processed_img)

                db_key_pattern = f"person{person_num}/{img_name}"
                temp_db = {k:v for k,v in index_db.items() if k != db_key_pattern}

                predictions = knn_classify(features.tolist(), temp_db, k_values=[3, 5, 7])

                entry = {
                    'Query Image': f"{subj}/{img_name}",
                    'True Identity': true_id,
                    'Pred K=3': predictions[3],
                    'Match K=3': predictions[3] == true_id,
                    'Pred K=5': predictions[5],
                    'Match K=5': predictions[5] == true_id,
                    'Pred K=7': predictions[7],
                    'Match K=7': predictions[7] == true_id
                }
                evaluation_log.append(entry)

            except Exception as e:
                print(f"FAILED to process {query_path}: {e}")

    df = pd.DataFrame(evaluation_log)

    if not df.empty:
        acc_k3 = df['Match K=3'].mean() * 100
        acc_k5 = df['Match K=5'].mean() * 100
        acc_k7 = df['Match K=7'].mean() * 100

        print("\n=== EVALUATION SUMMARY ===")
        print(f"Accuracy (K=3): {acc_k3:.2f}%")
        print(f"Accuracy (K=5): {acc_k5:.2f}%")
        print(f"Accuracy (K=7): {acc_k7:.2f}%")

        print(f"Saving detailed report to {OUTPUT_FILE}...")
        df.to_excel(OUTPUT_FILE, index=False)
        print("Done.")
    else:
        print("No results to report.")

if __name__ == "__main__":
    main()
