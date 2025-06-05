
# JPEG Compression and Decompression Tool

**Author:** Noor Fatima (Decoder) & Nimra Hanif(Encoder)  
**Purpose:** JPEG Encoder and Decoder for Digital Image Processing (DIP) Project

---

## Overview

This project implements a basic JPEG compression and decompression system using Python.  
It includes:

- A **JPEG Encoder** that reads an image, applies DCT, quantization, zigzag scanning, and run-length encoding, then saves the compressed data.
- A **JPEG Decoder** that reads compressed data, performs inverse operations (dequantization, inverse DCT, etc.) and reconstructs the image.
- A simple GUI built with `tkinter` to load, compress/decompress images, and save results.

---

## Features

- Supports color (YCrCb) and grayscale images.
- Implements custom quantization matrix.
- Uses DCT and inverse DCT matrix multiplication.
- Zigzag scanning and run-length encoding for compression.
- File save/load in binary format (`.bin`).
- GUI for user-friendly interaction.
- Save compressed images and reconstructed images.

---

## Requirements

- Python 3.x
- Packages:
  - `numpy`
  - `scipy`
  - `opencv-python`
  - `Pillow`

Install dependencies using:

```bash
pip install numpy scipy opencv-python Pillow
```

---

## How to Run

### Encoder

```bash
python encoder.py
```

- Open an image file (.jpg, .png, etc.)
- Choose to compress in color or grayscale
- Compress the image
- Save compressed binary file (.bin)
- Optionally save compressed JPEG preview with lower quality

### Decoder

```bash
python decoder.py
```

- Open a compressed `.bin` file
- View the decompressed image preview
- Save the reconstructed image as PNG or JPEG

---

## File Structure

- `encoder.py` — Contains the JPEG encoding logic and encoder GUI.
- `decoder.py` — Contains the JPEG decoding logic and decoder GUI.
- `quantization_table` — Standard quantization matrix used in encoding and decoding.

---

## Notes

- This project is for educational purposes and does not fully comply with JPEG standards.
- The encoder and decoder assume an 8x8 block size.
- The compressed binary format is a custom format using Python's `pickle`.

---

## License

This project is provided as-is for academic use.

---

## Author

Noor Fatima & Nimra Hanif
Digital Image Processing Project - Semester 8

---
