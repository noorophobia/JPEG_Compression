import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import pickle
from scipy.fftpack import idct
import cv2  # Add OpenCV for handling image color conversion

# Author: NOOR FATIMA 
# Purpose: JPEG DECODER FOR DIP PROJECT

def inversedct(block, mat):
    #return idct(idct(block.T, norm='ortho').T, norm='ortho')
    return np.dot(np.dot(mat.T, block), mat)

def zigzagAlgo(data):
    # mapping zigzag, approx
    zigzag_coords = []
    for sum_rc in range(16):  # 0 to 15 for an 8x8 matrix
        if sum_rc % 2 == 0:
            # moving down-left
            for r in range(sum_rc + 1):
                c = sum_rc - r
                if r < 8 and c < 8:
                    zigzag_coords.append((r, c))
        else:
            # moving up-right
            for c in range(sum_rc + 1):
                r = sum_rc - c
                if r < 8 and c < 8:
                    zigzag_coords.append((r, c))

    # create empty 8x8 block
    matrix = np.zeros((8, 8), dtype=np.int32)
    
    # place data into matrix using zigzag
    for idx, (row, col) in enumerate(zigzag_coords):
        if idx >= len(data):
            break  # sometimes data might be shorter than 64
        matrix[row, col] = data[idx]
    
    return matrix

def DCTMatrix(n = 8):
    mat = np.zeros((n,n))
    for u in range(n):
        for x in range(n):
            mat[u][x] = np.cos((2*x + 1)*u*np.pi/(2*n))
    return mat

def decode(data, shape, color=True):
    h, w = shape[:2]
    rows, cols = h // 8, w // 8
    mat = DCTMatrix()
    chs = []
    for ch in data:
        blocks = []
        for rle in ch:
            arr = []
            for val, count in rle:
                if val == 0:
                    arr.extend([0]*count)
                else:
                    arr.append(val)
            blk = zigzagAlgo(arr)
            q_table = Quanttable if len(chs) == 0 else np.maximum(Quanttable // 2, 1)
            blk = blk * q_table
            blk = inversedct(blk, mat) + 128
            blocks.append(np.clip(blk, 0, 255))
        img = np.zeros((h, w), dtype=np.uint8)
        i = 0
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                img[y:y+8, x:x+8] = blocks[i]
                i += 1
        chs.append(img)

        # print(f"Y channel shape: {chs[0].shape}")
        #print(f"Cr channel shape: {chs[1].shape}")
        #print(f"Cb channel shape: {chs[2].shape}")

    if color:
            #  Y, Cr, and Cb channel
        if len(chs) == 3:
            y, cr, cb = chs[0], chs[1], chs[2]
            # print(f"Y channel shape: {y.shape}")
            #print(f"Cr channel shape: {cr.shape}")
            #print(f"Cb channel shape: {cb.shape}")
            merged_img = cv2.merge([y, cr, cb])  
            return cv2.cvtColor(merged_img, cv2.COLOR_YCrCb2RGB)
    return chs[0]
Quanttable = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99]
])


class Decoder:
    def __init__(self, master):
        self.master = master
        self.master.title("JPEG Viewer")
        self._build()
        self.img = None

    def _build(self):
        self.lbl = tk.Label(self.master, text="Load an image")
        self.lbl.pack(pady=10)

        ttk.Button(self.master, text="Open .bin File", command=self.load).pack(pady=5)
        ttk.Button(self.master, text="Save Image", command=self.save_image).pack(pady=5)

    def load(self):
        fp = filedialog.askopenfilename(filetypes=[("BIN Files", "*.bin")])
        
        with open(fp, "rb") as f:
            pkg = pickle.load(f)
            print("Loaded compressed data")
            dataStuff = decode(pkg["data"], pkg["shape"], pkg["color"])
            self.img = dataStuff
            imgg = Image.fromarray(dataStuff)
            imgg.thumbnail((300, 300))
            self.tk_img = ImageTk.PhotoImage(imgg)
            self.lbl.config(image=self.tk_img, text="Image loaded!")

    def save_image(self):
        if self.img is None:
            messagebox.showerror("Error", "No image to save!")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")])
        
        if file_path:
            img_to_save = Image.fromarray(self.img)
            img_to_save.save(file_path)
            messagebox.showinfo("Saved", f"Image saved as {file_path}")

if __name__ == '__main__':
    root = tk.Tk()
    app = Decoder(root)
    root.mainloop()
