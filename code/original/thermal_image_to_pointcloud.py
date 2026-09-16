import cv2
import numpy as np

img = cv2.imread("frame_0.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("image not found")
    exit()

stride = 2
z_scale = 3000.0

img_f = img.astype(np.float32) / 255.0
h, w = img.shape

points = []
colors = []

for y in range(0, h, stride):
    for x in range(0, w, stride):
        z = img_f[y, x] * z_scale
        points.append([x, y, z])

        c = int(img[y, x])
        colors.append([c, c, c])

points = np.array(points, dtype=np.float32)
colors = np.array(colors, dtype=np.uint8)

print("point count:", len(points))

with open("thermal_pointcloud_big.ply", "w") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(points)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")

    for p, c in zip(points, colors):
        f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

print("Saved: thermal_pointcloud_big.ply")
