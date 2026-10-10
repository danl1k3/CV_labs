import matplotlib
from typing import Optional
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- Affine Transform ---

def apply_affine_transform(
    image: np.ndarray, M: np.ndarray, output_shape: Optional[tuple] = None
) -> np.ndarray:
    h_in, w_in = image.shape[:2]
    h_out, w_out = output_shape if output_shape else (h_in, w_in)

    M_hom = np.vstack([M, [0, 0, 1]])
    M_inv = np.linalg.inv(M_hom)

    y_out, x_out = np.indices((h_out, w_out))
    ones = np.ones_like(x_out)
    coords_out = np.stack([x_out.ravel(), y_out.ravel(), ones.ravel()])

    coords_in = M_inv @ coords_out
    x_in = np.round(coords_in[0]).astype(int)
    y_in = np.round(coords_in[1]).astype(int)

    valid_mask = (x_in >= 0) & (x_in < w_in) & (y_in >= 0) & (y_in < h_in)

    transformed = np.full(
        (h_out, w_out, image.shape[2]), 255, dtype=image.dtype
    )
    transformed.reshape(-1, image.shape[2])[valid_mask] = image[
        y_in[valid_mask], x_in[valid_mask]
    ]

    return transformed

# --- Find Matrix ---

def find_affine_matrix(
    src_pts: np.ndarray, dst_pts: np.ndarray
) -> np.ndarray:
    A, B = [], []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.extend([xp, yp])

    params = np.linalg.solve(np.array(A, float), np.array(B, float))
    return params.reshape((2, 3))

# --- Image Normalization ---

def normalize_image(
    distorted_img: np.ndarray, M_estimated: np.ndarray, target_shape: tuple
) -> np.ndarray:
    M_hom = np.vstack([M_estimated, [0, 0, 1]])
    M_inv = np.linalg.inv(M_hom)[:2, :]
    return apply_affine_transform(
        distorted_img, M_inv, output_shape=target_shape
    )

# --- GUI Part ---

def select_points(img: np.ndarray, title: str) -> np.ndarray:
    """Відкриває окреме вікно і зчитує 3 кліки миші."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.set_title(
        f"{title}\nКлікніть 3 вершини: 1. Верхню -> 2. Ліву нижню -> 3. Праву нижню",
        fontsize=10,
    )
    plt.draw()
    pts = plt.ginput(3, timeout=0)
    plt.close(fig)

    return np.array(pts, dtype=np.float64)


if __name__ == "__main__":
    source_img = np.array(Image.open("labs/lab1_files/Source.png").convert("RGB"))
    trans1_img = np.array(Image.open("labs/lab1_files/Transform1.png").convert("RGB"))
    trans2_img = np.array(Image.open("labs/lab1_files/Transform2.png").convert("RGB"))

    h, w = source_img.shape[:2]

    print("Вкажіть 3 точки для Source...")
    src_pts = select_points(source_img, "Source")

    print("Вкажіть 3 точки для Transform 1...")
    dst_pts1 = select_points(trans1_img, "Transform 1")

    print("Вкажіть 3 точки для Transform 2...")
    dst_pts2 = select_points(trans2_img, "Transform 2")

    # Find matrixes
    M1_est = find_affine_matrix(src_pts, dst_pts1)
    M2_est = find_affine_matrix(src_pts, dst_pts2)

    # Normalization
    norm1_img = normalize_image(trans1_img, M1_est, target_shape=(h, w))
    norm2_img = normalize_image(trans2_img, M2_est, target_shape=(h, w))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(source_img)
    axes[0, 0].set_title("Source")
    axes[0, 1].imshow(trans1_img)
    axes[0, 1].set_title("Transform 1")
    axes[0, 2].imshow(norm1_img)
    axes[0, 2].set_title("Normalized 1")

    axes[1, 0].imshow(source_img)
    axes[1, 0].set_title("Source")
    axes[1, 1].imshow(trans2_img)
    axes[1, 1].set_title("Transform 2")
    axes[1, 2].imshow(norm2_img)
    axes[1, 2].set_title("Normalized 2")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()