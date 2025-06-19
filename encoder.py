import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
from scipy.fftpack import dct
import os
import pickle

# Author: NIMRA
# Purpose: JPEG ENCODER FOR DIP PROJECT

QUANT_TABLE = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99]
])

def split_into_blocks(channel, block_size=8):
    h, w = channel.shape
    return [channel[i:i+block_size, j:j+block_size] 
            for i in range(0, h - h % block_size, block_size)
            for j in range(0, w - w % block_size, block_size)]

def dct2d(block):
    dct_matrix = DCTMatrix()
    return np.dot(np.dot(dct_matrix, block), dct_matrix.T)

def DCTMatrix(n=8):
    mat = np.zeros((n, n))
    for u in range(n):
        for x in range(n):
            mat[u][x] = np.cos((2*x + 1)*u*np.pi/(2*n))
    return mat

def quantize(block, qtable):
    return np.round(block / qtable).astype(np.int32)

def zigzagAlgo(block):
    coords = [(x, y) for x in range(8) for y in range(8)]
    coords.sort(key=lambda p: (p[0]+p[1], -p[1] if (p[0]+p[1]) % 2 else p[1]))
    return [block[x, y] for x, y in coords]

def inverse_dct(block, mat):
    return np.dot(np.dot(mat.T, block), mat)

def zigzag_inverse(data):
    coords = [(x, y) for x in range(8) for y in range(8)]
    coords.sort(key=lambda p: (p[0]+p[1], -p[1] if (p[0]+p[1]) % 2 else p[1]))
    matrix = np.zeros((8, 8), dtype=np.int32)
    for idx, (x, y) in enumerate(coords):
        if idx < len(data):
            matrix[x, y] = data[idx]
    return matrix

def decompress_preview(data_dict):
    h, w = data_dict["shape"][:2]
    color = data_dict["color"]
    mat = DCTMatrix()

    channels = []
    for idx, channel_data in enumerate(data_dict["data"]):
        blocks = []
        for rle in channel_data:
            flat = []
            for val, count in rle:
                if val == 0:
                    flat.extend([0]*count)
                else:
                    flat.append(val)
            zigzagged = zigzag_inverse(flat)
            q_table = QUANT_TABLE if idx == 0 else np.round(QUANT_TABLE * 0.75).astype(np.int32)
            dequant = zigzagged * q_table
            block = inverse_dct(dequant, mat) + 128
            blocks.append(np.clip(block, 0, 255))

        img_channel = np.zeros((h, w), dtype=np.uint8)
        i = 0
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                img_channel[y:y+8, x:x+8] = blocks[i]
                i += 1
        channels.append(img_channel)

    if color and len(channels) == 3:
        y, cr, cb = channels
        merged = cv2.merge([y, cr, cb])
        rgb_img = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
    else:
        rgb_img = channels[0]

    return rgb_img

def encode(image_path, use_color=True):
    print(f"Reading image from path: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if use_color:
        print("Converting RGB to YCrCb color space")
        ycbcr_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        channels = cv2.split(ycbcr_image)
    else:
        print("Converting RGB to Grayscale")
        grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        channels = [grayscale]

    compressed = []
    channel_names = ["Y", "Cr", "Cb"] if use_color else ["Gray"]
    for idx, channel in enumerate(channels):
        print(f"Processing {channel_names[idx]} channel...")
        normalized = channel.astype(np.float32) - 128
        blocks = split_into_blocks(normalized)
        print(f"Total blocks to process: {len(blocks)}")
        encoded_blocks = []

        q_table = QUANT_TABLE if idx == 0 else np.round(QUANT_TABLE * 0.75).astype(np.int32)
        print(f"Using quantization table for {channel_names[idx]} channel:\n{q_table}")

        for block in blocks:
            dct_block = dct2d(block)
            quantized = quantize(dct_block, q_table)
            zigzag = zigzagAlgo(quantized)

            result, zero_count = [], 0
            for val in zigzag:
                if val == 0:
                    zero_count += 1
                else:
                    if zero_count > 0:
                        result.append((0, zero_count))
                        zero_count = 0
                    result.append((val, 1))
            if zero_count > 0:
                result.append((0, zero_count))

            encoded_blocks.append(result)

        compressed.append(encoded_blocks)
        print(f"Finished encoding {channel_names[idx]} channel. Encoded blocks: {len(encoded_blocks)}")

    print("Encoding completed successfully.")
    return {
        "data": compressed,
        "shape": image_rgb.shape,
        "color": use_color
    }

class ENCODER:
    def __init__(self, master):
        self.master = master
        master.title("JPeg ENCODER")
        self.widgets()

    def widgets(self):
        self.preview_label = tk.Label(self.master, text="No image loaded")
        self.preview_label.pack(pady=10)

        ttk.Button(self.master, text="Open Image", command=self.loader).pack(pady=5)

        self.use_color = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.master, text="Use Color", variable=self.use_color).pack(pady=5)

        ttk.Button(self.master, text="Compress", command=self._compress).pack(pady=5)

        self.save_button = ttk.Button(self.master, text="Save Result", command=self.SAVECOMPRESSED, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        self.status = tk.Label(self.master, text="Ready")
        self.status.pack(pady=5)

    def loader(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        self.image_path = file_path
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        self.image = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self.image, text="")
        self.status.config(text="Image loaded")
        self.save_button.config(state=tk.DISABLED)
        print("Image loaded successfully.")

    def _compress(self):
        self.compressed_data = encode(self.image_path, self.use_color.get())
        compressed_size = sum(len(b) for ch in self.compressed_data["data"] for b in ch)
        print(f"Approximate compressed size: {compressed_size} units")
        self.status.config(text=f"Compressed size: ~{compressed_size} units")
        self.save_button.config(state=tk.NORMAL)
        self.SAVEBINARY(self.compressed_data)

        preview_img = decompress_preview(self.compressed_data)
        preview_pil = Image.fromarray(preview_img)
        preview_pil.thumbnail((300, 300))
        self.preview_image = ImageTk.PhotoImage(preview_pil)
        self.preview_label.config(image=self.preview_image, text="")

    def SAVEBINARY(self, data):
        path = filedialog.asksaveasfilename(defaultextension=".bin", filetypes=[("Binary Files", "*.bin")])
        with open(path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Compressed data saved to {path}")

    def SAVECOMPRESSED(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
        image = cv2.imread(self.image_path)
        cv2.imwrite(save_path, image, [cv2.IMWRITE_JPEG_QUALITY, 30])
        print(f"Original image saved with compression quality 30 to {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ENCODER(root)
    root.mainloop()
