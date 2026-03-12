# basicVSR test -> bicubic
# basicVSR++ test -> realistic

import os
import cv2
import numpy as np
import random

def add_gaussian_noise(img, sigma_range=(2, 10)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.randn(*img.shape) * sigma
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def jpeg_compress(img, quality_range=(30, 80)):
    q = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def degrade_video(filename, scale=4, mode="bicubic_clean"):
    path_in = os.path.join('Videos', filename)
    name, ext = os.path.splitext(filename)

    out_dir = os.path.join('Videos_Down', mode)
    os.makedirs(out_dir, exist_ok=True)
    path_out = os.path.join(out_dir, filename)

    cap = cv2.VideoCapture(path_in)
    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {filename}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    low_res_size = (w // scale, h // scale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_out, fourcc, fps, low_res_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "bicubic_clean":
            # Degradazione stile BasicVSR (benchmark)
            degraded = cv2.resize(frame, low_res_size, interpolation=cv2.INTER_CUBIC)

        elif mode == "realistic":
            # Blur
            k = random.choice([3, 5, 7])
            blurred = cv2.GaussianBlur(frame, (k, k), 0)

            # Downscale bicubico
            low_res = cv2.resize(blurred, low_res_size, interpolation=cv2.INTER_CUBIC)

            # Rumore
            low_res = add_gaussian_noise(low_res)

            # Compressione JPEG
            degraded = jpeg_compress(low_res)

        else:
            raise ValueError("Modalità non supportata")

        out.write(degraded)

    cap.release()
    out.release()
    print(f"✅ {filename} degradato in modalità: {mode}")

def main():
    input_dir = "Videos"
    os.makedirs("Videos_Down_test_basicVSR", exist_ok=True)

    for v in os.listdir(input_dir):
        if v.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            degrade_video(v, scale=4, mode="bicubic_clean")
            degrade_video(v, scale=4, mode="realistic")

if __name__ == "__main__":
    main()
