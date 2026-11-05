"""
Generate sample synthetic road images with potholes and YOLO label annotations.
This allows the project to be self-contained and immediately testable even without an external internet download.
"""
import struct
import os
import random

def create_bmp(filepath, width, height, potholes):
    """
    Creates an uncompressed 24-bit BMP image resembling an asphalt road with potholes.
    """
    # BMP Header
    row_padding = (4 - (width * 3) % 4) % 4
    image_size = (width * 3 + row_padding) * height
    file_size = 54 + image_size
    
    # BMP File Header (14 bytes)
    header = bytearray([
        0x42, 0x4D,             # 'BM'
        file_size & 0xFF, (file_size >> 8) & 0xFF, (file_size >> 16) & 0xFF, (file_size >> 24) & 0xFF,
        0, 0, 0, 0,             # Reserved
        54, 0, 0, 0             # Pixel data offset
    ])
    
    # DIB Header (BITMAPINFOHEADER - 40 bytes)
    dib = bytearray([
        40, 0, 0, 0,            # Header size
        width & 0xFF, (width >> 8) & 0xFF, (width >> 16) & 0xFF, (width >> 24) & 0xFF,
        height & 0xFF, (height >> 8) & 0xFF, (height >> 16) & 0xFF, (height >> 24) & 0xFF,
        1, 0,                   # Planes
        24, 0,                  # Bits per pixel (24 bit RGB)
        0, 0, 0, 0,             # Compression (none)
        image_size & 0xFF, (image_size >> 8) & 0xFF, (image_size >> 16) & 0xFF, (image_size >> 24) & 0xFF,
        0x12, 0x0B, 0, 0,       # Horizontal resolution (~72 dpi)
        0x12, 0x0B, 0, 0,       # Vertical resolution (~72 dpi)
        0, 0, 0, 0,             # Colors in color table
        0, 0, 0, 0              # Important color count
    ])
    
    # Generate pixel rows (bottom to top for BMP)
    pixels = bytearray()
    random.seed(42)
    
    for y in range(height):
        row = bytearray()
        actual_y = height - 1 - y  # flip for BMP
        for x in range(width):
            # Base road texture (gray asphalt with subtle variations)
            noise = random.randint(-12, 12)
            b = min(255, max(0, 75 + noise))
            g = min(255, max(0, 78 + noise))
            r = min(255, max(0, 80 + noise))
            
            # Road markings (yellow center line or white shoulder)
            if abs(x - width // 2) < 6 and (actual_y % 60 < 35):
                b = 20
                g = 200
                r = 230  # Yellow dash
            elif abs(x - 40) < 4 or abs(x - (width - 40)) < 4:
                b = 210
                g = 210
                r = 210  # White lane marker
                
            # Check if (x, actual_y) is inside any pothole ellipse
            for (px, py, rx, ry) in potholes:
                dx = (x - px) / max(1, rx)
                dy = (actual_y - py) / max(1, ry)
                dist_sq = dx*dx + dy*dy
                if dist_sq <= 1.0:
                    # Inside pothole - deep dark gray / cavity depth shadow
                    depth_factor = 1.0 - dist_sq * 0.4
                    b = int(25 * depth_factor + random.randint(0, 8))
                    g = int(25 * depth_factor + random.randint(0, 8))
                    r = int(28 * depth_factor + random.randint(0, 8))
                elif dist_sq <= 1.25:
                    # Pothole edge rim / cracked jagged border
                    b = int(45 + random.randint(-5, 10))
                    g = int(45 + random.randint(-5, 10))
                    r = int(48 + random.randint(-5, 10))

            row.extend([b, g, r])
            
        row.extend([0] * row_padding)
        pixels.extend(row)
        
    with open(filepath, 'wb') as f:
        f.write(header)
        f.write(dib)
        f.write(pixels)

def main():
    base_dir = r"C:\Users\Abhi\.gemini\antigravity\scratch\Pothole-Detection-AIDL"
    dataset_dir = os.path.join(base_dir, "dataset")
    os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "test"), exist_ok=True)
    
    # Generate 10 sample images across splits with corresponding YOLO txt files
    splits = [
        ("train", 6),
        ("val", 2),
        ("test", 2)
    ]
    
    img_w, img_h = 640, 480
    
    sample_potholes = [
        [(240, 310, 45, 25), (420, 360, 60, 35)],
        [(320, 280, 50, 30)],
        [(180, 350, 65, 40), (360, 310, 40, 22), (500, 390, 55, 30)],
        [(300, 340, 70, 42)],
        [(220, 260, 35, 20), (410, 330, 55, 32)],
        [(350, 380, 80, 48)],
        [(270, 320, 48, 28)],
        [(390, 300, 42, 25), (200, 370, 62, 38)],
        [(310, 350, 58, 34)],
        [(250, 290, 40, 24), (430, 340, 50, 30)]
    ]
    
    idx = 0
    for split, count in splits:
        for i in range(count):
            img_name = f"road_pothole_{split}_{i+1}.bmp"
            txt_name = f"road_pothole_{split}_{i+1}.txt"
            
            img_path = os.path.join(dataset_dir, "images", split, img_name)
            txt_path = os.path.join(dataset_dir, "labels", split, txt_name)
            
            potholes = sample_potholes[idx % len(sample_potholes)]
            idx += 1
            
            create_bmp(img_path, img_w, img_h, potholes)
            
            # Write YOLO format: class_id x_center y_center width height (normalized)
            with open(txt_path, 'w') as f:
                for (px, py, rx, ry) in potholes:
                    # Bounding box around ellipse
                    x_center = px / img_w
                    y_center = py / img_h
                    w = (rx * 2 + 10) / img_w
                    h = (ry * 2 + 8) / img_h
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                    
    print(f"Successfully generated sample images and YOLO labels in {dataset_dir}")

if __name__ == "__main__":
    main()
