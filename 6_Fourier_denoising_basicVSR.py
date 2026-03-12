import os
import cv2
import numpy as np

INPUT_FOLDER = "Videos/Videos_Restored/VSR_degradation_v2"
SIGMA = 35  # controlla quanto filtri le alte frequenze (20-50 consigliato)

def fourier_denoise_frame(frame, sigma=35):
    # Lavora per canale colore per mantenere RGB
    channels = cv2.split(frame)
    filtered_channels = []

    for ch in channels:
        f = np.fft.fft2(ch)
        fshift = np.fft.fftshift(f)

        rows, cols = ch.shape
        crow, ccol = rows // 2, cols // 2

        # Maschera gaussiana passa-basso
        x = np.linspace(-cols/2, cols/2, cols)
        y = np.linspace(-rows/2, rows/2, rows)
        X, Y = np.meshgrid(x, y)
        mask = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

        fshift_filtered = fshift * mask

        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        filtered_channels.append(np.clip(img_back, 0, 255).astype(np.uint8))

    return cv2.merge(filtered_channels)


def process_video(input_path, output_path, sigma=35):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Errore apertura: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing: {os.path.basename(input_path)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        denoised = fourier_denoise_frame(frame, sigma)
        out.write(denoised)

    cap.release()
    out.release()
    print(f"Salvato: {output_path}")


def main():
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(".mp4"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_name = f"fourier_denoising_{filename}"
            output_path = os.path.join(INPUT_FOLDER, output_name)

            process_video(input_path, output_path, SIGMA)


if __name__ == "__main__":
    main()