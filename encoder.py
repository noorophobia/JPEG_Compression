import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
from scipy.fftpack import dct
import os
import pickle


# Author: NIMRA
# Purpose: JPEG ENCODER FOR DIP PRJETC




def split_into_blocks(channel, block_size=8):
    #  2D / non-overlapping blocks
    h, w = channel.shape
    blocks = [channel[i:i+block_size, j:j+block_size] 
            for i in range(0, h - h % block_size, block_size)
            for j in range(0, w - w % block_size, block_size)]
    return blocks

def dct2d(block):
    """Apply 2D DCT transfROM"""
 #return dct(dct(block.T, norm='ortho').T, norm='ortho')
    dct_matrix = DCTMatrix()   
    return np.dot(np.dot(dct_matrix, block), dct_matrix.T)

def DCTMatrix(n = 8):
    
    mat = np.zeros((n,n))
    for u in range(n):
        for x in range(n):
            mat[u][x] = np.cos((2*x + 1)*u*np.pi/(2*n)) #DCT basis function
    return mat

def quantize(block, qtable):

    return np.round(block / qtable).astype(np.int32)

def rgbtoYCBR(image):
    
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

def zigzagAlgo(block):
    # Convert 8x8 block to 1D array using zigzag 
    coords = [(x, y) for x in range(8) for y in range(8)]
    coords.sort(key=lambda p: (p[0]+p[1], -p[1] if (p[0]+p[1]) % 2 else p[1]))
    return [block[x, y] for x, y in coords]

 

def encode(image_path, use_color=True):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if use_color:
        ycbcr_image = rgbtoYCBR(image_rgb)
        channels = cv2.split(ycbcr_image)
    else:
        grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        channels = [grayscale]

    compressed = []
    for channel in channels:
        normalized = channel.astype(np.float32) - 128
        blocks = split_into_blocks(normalized)
        encoded_blocks = []

        for block in blocks:
            dct_block = dct2d(block)
            quantized = quantize(dct_block, QUANT_TABLE)
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

    return {
        "data": compressed,
        "shape": image_rgb.shape,
        "color": use_color
    }


QUANT_TABLE = np.array([
    [17, 12, 10, 18, 25, 39, 50, 60],
    [11, 13, 14, 20, 28, 57, 62, 54],
    [14, 14, 17, 23, 38, 58, 70, 55],
    [15, 18, 22, 31, 50, 85, 81, 63],
    [19, 23, 36, 55, 67, 108, 102, 78],
    [25, 34, 54, 65, 82, 103, 114, 91],
    [48, 65, 77, 88, 104, 122, 121, 100],
    [71, 93, 96, 97, 111, 99, 102, 98]
])

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

    def _compress(self):
        
            self.compressed_data = encode(self.image_path, self.use_color.get())
            compressed_size = sum(len(b) for ch in self.compressed_data["data"] for b in ch)
            self.status.config(text=f"Compressed size: ~{compressed_size} units")
            self.save_button.config(state=tk.NORMAL)
            self.SAVEBINARY(self.compressed_data)
        

    def SAVEBINARY(self, data):
        path = filedialog.asksaveasfilename(defaultextension=".bin", filetypes=[("Binary Files", "*.bin")])
        with open(path, 'wb') as file:
                    pickle.dump(data, file)
            

    def SAVECOMPRESSED(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
        image = cv2.imread(self.image_path)
        cv2.imwrite(save_path, image, [cv2.IMWRITE_JPEG_QUALITY, 30])

    

if __name__ == "__main__":
    root = tk.Tk()
    app = ENCODER(root)
    root.mainloop()