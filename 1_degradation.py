import os
import cv2

def degrade_video(filename, scale=4):
    path_in = os.path.join('Videos', filename)
    path_out = os.path.join('Videos_Down', filename)

    cap = cv2.VideoCapture(path_in)
    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {filename}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Risoluzione ridotta
    low_res_size = (w // scale, h // scale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_out, fourcc, fps, low_res_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize bilineare (più sfocato del bicubico, ottimo per test)
        low_res = cv2.resize(frame, low_res_size, interpolation=cv2.INTER_LINEAR)
        out.write(low_res)

    cap.release()
    out.release()
    print(f"✅ Video {filename} degradato con successo.")

def main():
    input_dir = "Videos"
    output_dir = "Videos_Down"

    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    for v in os.listdir(input_dir):
        if v.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            degrade_video(v, scale=4)

if __name__ == "__main__":
    main()
