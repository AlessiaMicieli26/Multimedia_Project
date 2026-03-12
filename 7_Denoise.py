#!/usr/bin/env python3
"""
Denoiser per righe verticali (vertical banding) da Basic VSR.
Rimuove le strisce verticali periodiche tipiche della ricostruzione VSR.

Uso:
    python denoise_vsr.py --batch "Videos/Videos_Restored/VSR_degradation_v2"
    python denoise_vsr.py input.mp4 output.mp4 [--strength leggero|medio|forte]
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


# Approccio: blur orizzontale selettivo (smootens solo la direzione delle bande)
# + deband per ridurre banding cromatico residuo
# fftfilt richiede libavfilter con fft abilitato — verifichiamo disponibilita

PRESETS = {
    "leggero": {
        # Blur orizzontale molto leggero: kernel 3x1, preserva dettagli verticali
        "filters_fft":    "fftfilt=lum_expr='hypot(X,Y) > 2 ? 1 : 0.95'",
        "filters_simple": "boxblur=lx=2:lp=1:ly=0:lp=1,deband=1thr=0.02:2thr=0.02:3thr=0.02:blur=1",
        "filters_basic":  "gblur=sigma=0.5:steps=1",
        "description":    "Leggero — bande sottili, preserva nitidezza"
    },
    "medio": {
        "filters_fft":    "fftfilt=lum_expr='hypot(X,Y) > 1.5 ? 1 : 0.9'",
        "filters_simple": "boxblur=lx=3:lp=1:ly=0:lp=1,deband=1thr=0.04:2thr=0.04:3thr=0.04:blur=1",
        "filters_basic":  "gblur=sigma=0.8:steps=1",
        "description":    "Medio — bande moderate (consigliato)"
    },
    "forte": {
        "filters_fft":    "fftfilt=lum_expr='hypot(X,Y) > 1 ? 1 : 0.8'",
        "filters_simple": "boxblur=lx=5:lp=1:ly=0:lp=1,deband=1thr=0.06:2thr=0.06:3thr=0.06:blur=1",
        "filters_basic":  "gblur=sigma=1.2:steps=2",
        "description":    "Forte — bande evidenti come nell'immagine"
    },
}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ERRORE: ffmpeg non trovato nel PATH.")
        sys.exit(1)


def detect_encoder():
    for enc, extra in [
        ("libx264", ["-crf", "18"]),
        ("libx265", ["-crf", "28"]),
        ("mpeg4",   ["-q:v", "3"]),
    ]:
        test = subprocess.run(
            ["ffmpeg", "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.5",
             "-c:v", enc] + extra + ["-f", "null", "-"],
            capture_output=True
        )
        if test.returncode == 0:
            return enc, extra
    print("ERRORE: nessun encoder disponibile.")
    sys.exit(1)


def detect_best_filter(strength: str) -> str:
    """
    Sceglie il miglior filtro disponibile per rimuovere bande verticali.
    Priorita: fftfilt > boxblur+deband > gblur (sempre disponibile)
    """
    preset = PRESETS[strength]

    # 1. Prova fftfilt (migliore, agisce in frequenza)
    test = subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.3",
         "-vf", preset["filters_fft"], "-f", "null", "-"],
        capture_output=True
    )
    if test.returncode == 0:
        print(f"  [Filtro] fftfilt (ottimale per bande verticali)")
        return preset["filters_fft"]

    # 2. Prova boxblur + deband
    test = subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.3",
         "-vf", preset["filters_simple"], "-f", "null", "-"],
        capture_output=True
    )
    if test.returncode == 0:
        print(f"  [Filtro] boxblur orizzontale + deband")
        return preset["filters_simple"]

    # 3. Fallback: gblur (sempre disponibile, meno preciso)
    print(f"  [Filtro] gblur (fallback)")
    return preset["filters_basic"]


def denoise_video(input_path: str, output_path: str, strength: str = "forte",
                  encoder=None, enc_quality_args=None):
    check_ffmpeg()

    if not Path(input_path).exists():
        print(f"ERRORE: file non trovato -> {input_path}")
        return False

    if strength not in PRESETS:
        print(f"ERRORE: strength deve essere: {', '.join(PRESETS.keys())}")
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if encoder is None:
        encoder, enc_quality_args = detect_encoder()

    filter_chain = detect_best_filter(strength)
    desc = PRESETS[strength]["description"]

    print(f"\n{'─'*58}")
    print(f"  Input   : {Path(input_path).name}")
    print(f"  Output  : {Path(output_path).name}")
    print(f"  Preset  : {strength} — {desc}")
    print(f"  Encoder : {encoder}")
    print(f"  Filtro  : {filter_chain}")
    print(f"{'─'*58}")

    cmd = (
        ["ffmpeg", "-i", input_path,
         "-vf", filter_chain,
         "-c:v", encoder]
        + enc_quality_args
        + ["-c:a", "copy", "-y", output_path]
    )

    result = subprocess.run(cmd)

    if result.returncode == 0:
        out_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"[OK] {Path(output_path).name} ({out_mb:.1f} MB)")
        return True
    else:
        print(f"[ERRORE] codice {result.returncode} per: {input_path}")
        return False


def denoise_batch(folder: str, strength: str = "forte", suffix: str = "_denoised"):
    check_ffmpeg()

    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"ERRORE: cartella non trovata -> {folder}")
        sys.exit(1)

    video_files = sorted([f for f in folder_path.iterdir()
                          if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS])

    if not video_files:
        print(f"Nessun video trovato in: {folder}")
        sys.exit(0)

    output_folder = folder_path / "denoised"
    output_folder.mkdir(exist_ok=True)

    encoder, enc_quality_args = detect_encoder()

    print(f"\n{'='*58}")
    print(f"  BATCH — rimozione bande verticali VSR")
    print(f"  Video   : {len(video_files)}")
    print(f"  Preset  : {strength} — {PRESETS[strength]['description']}")
    print(f"  Input   : {folder_path}")
    print(f"  Output  : {output_folder}")
    print(f"  Encoder : {encoder}")
    print(f"{'='*58}")

    ok = fail = 0
    for i, video in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] {video.name}")
        out_path = output_folder / (video.stem + suffix + video.suffix)
        if denoise_video(str(video), str(out_path), strength, encoder, enc_quality_args):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*58}")
    print(f"  COMPLETATO: {ok} ok  |  {fail} errori")
    print(f"  Output in: {output_folder}")
    print(f"{'='*58}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rimuove bande verticali (vertical banding) da video Basic VSR"
    )
    parser.add_argument("--batch", metavar="CARTELLA",
                        help="Processa tutti i video in una cartella")
    parser.add_argument("input",  nargs="?", help="Video di input")
    parser.add_argument("output", nargs="?", help="Video di output")
    parser.add_argument("--strength", choices=["leggero", "medio", "forte"],
                        default="forte",
                        help="Intensita rimozione bande (default: forte)")
    parser.add_argument("--suffix", default="_denoised",
                        help="Suffisso file in modalita batch (default: _denoised)")

    args = parser.parse_args()

    if args.batch:
        denoise_batch(args.batch, args.strength, args.suffix)
    elif args.input and args.output:
        denoise_video(args.input, args.output, args.strength)
    else:
        parser.print_help()
        print("\nEsempi:")
        print('  Batch : python denoise_vsr.py --batch "Videos/Videos_Restored/VSR_degradation_v2"')
        print('  File  : python denoise_vsr.py input.mp4 output.mp4 --strength forte')
        sys.exit(1)


if __name__ == "__main__":
    main()