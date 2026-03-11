# Restauro di Video Degradati tramite Interpolazione e Super-Resolution

## Descrizione del progetto

Questo progetto ha come obiettivo il restauro di video degradati utilizzando tecniche di **interpolazione dei frame** e **super-resolution**.  

L'idea principale è simulare diversi tipi di degradazione su video originali e successivamente applicare metodi di ricostruzione per recuperare qualità visiva, dettagli e informazioni temporali.

Il sistema permette di:
- generare versioni degradate di video originali
- applicare metodi di interpolazione dei frame
- migliorare la risoluzione dei video tramite tecniche di super-resolution
- confrontare i risultati ottenuti con i video originali

---

## Obiettivi

Gli obiettivi principali del progetto sono:

- simulare degradazioni su video originali
- applicare tecniche di interpolazione temporale
- applicare tecniche di super-resolution
- ricostruire video degradati
- valutare la qualità della ricostruzione

---

## Pipeline del sistema

La pipeline del progetto è composta dalle seguenti fasi:

1. Video originale
2. Applicazione della degradazione
3. Generazione del video degradato
4. Interpolazione dei frame
5. Applicazione della super-resolution
6. Valutazione dei risultati

Flusso generale:

Video originale → Degradazione → Video degradato → Interpolazione → Super-Resolution → Video ricostruito

---

## Degradazione dei video

Per testare le tecniche di restauro, vengono create versioni degradate dei video originali.

Sono state utilizzate diverse tipologie di degradazione, tra cui:

### Bicubic Clean

Riduzione della risoluzione tramite **downsampling bicubico**.  
Questa degradazione simula una perdita di dettaglio dovuta alla riduzione della qualità del video.

### Realistic Degradation

Simula degradazioni più realistiche che possono includere:

- rumore
- compressione
- perdita di dettaglio
- artefatti visivi

---

## Metodo Baseline

Come punto di partenza è stato implementato un **metodo baseline** basato su interpolazione.

Questo metodo utilizza i frame vicini nel tempo per ricostruire frame mancanti o migliorare la continuità temporale del video.

Il baseline serve come riferimento per valutare eventuali metodi più avanzati.

---

## Tecnologie utilizzate

Il progetto utilizza strumenti e concetti provenienti da:

- Python
- Computer Vision
- Video Processing
- Frame Interpolation
- Image Super-Resolution

---

## Valutazione delle prestazioni

Le prestazioni dei metodi utilizzati vengono valutate confrontando i video ricostruiti con i video originali.

Le metriche utilizzate includono:

- **PSNR (Peak Signal-to-Noise Ratio)**  
- **SSIM (Structural Similarity Index)**  

Queste metriche permettono di misurare la qualità visiva e la fedeltà della ricostruzione.

---

## Possibili sviluppi futuri

Il progetto può essere esteso introducendo:

- modelli di **deep learning per super-resolution**
- tecniche avanzate di **video frame interpolation**
- modelli neurali per **video restoration**
- dataset più ampi e più vari

---

## Autore

Alessia Micieli

Progetto realizzato per il corso di **Multimedia / Computer Vision**.
