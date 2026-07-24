import os
import cv2
import numpy as np
from numba import njit

@njit
def solve_ootheca(Sx, Sy, max=10, tol=1e-4):
    cx, cy = 128.0, 128.0
    for _ in range(max):
        Ex = (Sx + cx) / 3.0
        Ey = (Sy + cy) / 3.0
        if abs(Ex) < 1e-4 or abs(Ey) < 1e-4:
            return 1
        f1 = cx - (Ex**2) / Ey
        f2 = cy - (Ey**2) / Ex
        
        j0 = 1.0 - (2.0 * Ex) / (3.0 * Ey)
        j1 = (Ex**2) / (3.0 * Ey**2)
        j2 = (Ey**2) / (3.0 * Ex**2)
        j3 = 1.0 - (2.0 * Ey) / (3.0 * Ex)
        
        det = j0 * j3 - j1 * j2
        if abs(det) < 1e-6:
            cx += 1.0
            cy += 1.0
            continue
            
        dcx = (f1 * j3 - f2 * j1) / det
        dcy = (j0 * f2 - f1 * j2) / det
        cx -= dcx
        cy -= dcy
        
        if abs(f1) < tol and abs(f2) < tol:
            return int(abs(cx)) % 256
    return int(abs(cx)) % 256

@njit
def encrypt_channel(pixels, epsilon=0.02):
    out_pixels = np.zeros_like(pixels)
    Sx = pixels[0] + 1.0
    Sy = pixels[1] + 1.0
    out_pixels[0] = pixels[0]
    out_pixels[1] = pixels[1]
    
    for i in range(2, len(pixels)):
        modifier = solve_ootheca(Sx, Sy)

        step = int(modifier * epsilon) % 256
        
        cipher_p = (pixels[i] + step) % 256
        out_pixels[i] = cipher_p
        
        Sx = float(cipher_p) + 1.0
        Sy = float(pixels[i-1]) + 1.0
    return out_pixels

@njit
def decrypt_channel(cipher_pixels, epsilon=0.02):
    out_pixels = np.zeros_like(cipher_pixels)
    Sx = cipher_pixels[0] + 1.0
    Sy = cipher_pixels[1] + 1.0
    out_pixels[0] = cipher_pixels[0]
    out_pixels[1] = cipher_pixels[1]
    
    for i in range(2, len(cipher_pixels)):
        modifier = solve_ootheca(Sx, Sy)
        
        step = int(modifier * epsilon) % 256
        
        plain_p = (cipher_pixels[i] - step) % 256
        out_pixels[i] = plain_p
        
        Sx = float(cipher_pixels[i]) + 1.0
        Sy = float(out_pixels[i-1]) + 1.0
    return out_pixels

def process_image(image_path, mode=0, epsilon=0.02):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Image could not be loaded :(")
    height, width, _ = img.shape
    b_chan, g_chan, r_chan = cv2.split(img)
    
    if mode == 0:
        b_out = encrypt_channel(b_chan.flatten().astype(np.int64), epsilon).reshape((height, width))
        g_out = encrypt_channel(g_chan.flatten().astype(np.int64), epsilon).reshape((height, width))
        r_out = encrypt_channel(r_chan.flatten().astype(np.int64), epsilon).reshape((height, width))
    else:
        b_out = decrypt_channel(b_chan.flatten().astype(np.int64), epsilon).reshape((height, width))
        g_out = decrypt_channel(g_chan.flatten().astype(np.int64), epsilon).reshape((height, width))
        r_out = decrypt_channel(r_chan.flatten().astype(np.int64), epsilon).reshape((height, width))
        
    return cv2.merge([b_out, g_out, r_out]).astype(np.uint8)

def main():
    print("OOTHECA")
    input_path = input("Enter image file path: ").strip('"\'')
    if not os.path.exists(input_path):
        print("Error: Path does not exist :(")
        return
        
    mode_selection = input("[E]ncrypt or [D]ecrypt? ").strip().lower()
    if mode_selection not in ['e', 'd']:
        print("Invalid choice :(")
        return
    mode_bit = 0 if mode_selection == 'e' else 1
        
    try:
        epsilon_val = input("Enter epsilon (default 0.2): ").strip()
        epsilon = float(epsilon_val) if epsilon_val else 0.2
    except ValueError:
        print("Invalid float :(")
        return

    current_project_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    _, file_name = os.path.split(input_path)
    base, _ = os.path.splitext(file_name)
    suffix = "_oe" if mode_bit == 0 else "_od"
    
    output_path = os.path.join(current_project_dir, base + suffix + ".png")
    
    print("\nProcessing file...")
    try:
        final_img = process_image(input_path, mode_bit, epsilon)
        cv2.imwrite(output_path, final_img)
        print(f"Success! Saved file at:\n--> {output_path}")
        if mode_bit == 0:
            print("\nEncrypted image via Ootheca :)")
        else:
            print("\nDecrypted image via Ootheca :)")
    except Exception as e:
        print(f"An unexpected failure occurred: {e}")

if __name__ == "__main__":
    main()