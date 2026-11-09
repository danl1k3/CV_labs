import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import skimage
from sklearn.datasets import load_sample_image

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(7)


# --- 1. Базові операції над зображенням ---

# читає картинку у float масив (H, W, 3)
def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64)


# переводить RGB у градації сірого
def to_gray(rgb):
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


# пропорційно зменшує зображення (найближчий сусід) для швидшого Хафа
def resize_nearest(img, max_side):
    h, w = img.shape[:2]
    scale = max(h, w) / max_side
    if scale <= 1:
        return img
    ys = (np.arange(int(h / scale)) * scale).astype(int)
    xs = (np.arange(int(w / scale)) * scale).astype(int)
    return img[np.ix_(ys, xs)]


# 2D згортка з відбиттям країв (векторизована через зсуви)
def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    flipped = kernel[::-1, ::-1]
    out = np.zeros_like(img, dtype=np.float64)
    H, W = img.shape
    for i in range(kh):
        for j in range(kw):
            out += flipped[i, j] * padded[i:i + H, j:j + W]
    return out


# генерує нормоване ядро Гаусса
def gaussian_kernel(sigma):
    size = int(2 * np.ceil(3 * sigma) + 1)
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


# розмиття по Гауссу
def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    return convolve2d(img, gaussian_kernel(sigma))


# --- 2. Виділення країв (Канні) ---

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


# рахує величину та напрямок градієнта оператором Собеля
def sobel(img):
    gx = convolve2d(img, SOBEL_X)
    gy = convolve2d(img, SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)
    return magnitude, angle


# залишає тільки локальні максимуми вздовж градієнта (потоншення країв)
def non_max_suppression(mag, ang):
    angle_deg = np.rad2deg(ang) % 180
    quant = np.zeros_like(angle_deg, dtype=int)
    quant[(angle_deg >= 22.5) & (angle_deg < 67.5)] = 1
    quant[(angle_deg >= 67.5) & (angle_deg < 112.5)] = 2
    quant[(angle_deg >= 112.5) & (angle_deg < 157.5)] = 3

    padded = np.pad(mag, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    neighbors = {
        0: (padded[1:-1, 2:], padded[1:-1, :-2]),    # горизонталь
        1: (padded[2:, 2:], padded[:-2, :-2]),      # діагональ /
        2: (padded[:-2, 1:-1], padded[2:, 1:-1]),   # вертикаль
        3: (padded[:-2, 2:], padded[2:, :-2]),      # діагональ \
    }
    out = np.zeros_like(mag)
    for direction in range(4):
        n1, n2 = neighbors[direction]
        keep = (quant == direction) & (center >= n1) & (center >= n2)
        out[keep] = center[keep]
    return out


# подвійний поріг: добираємо слабкі краї, якщо вони торкаються сильних
def hysteresis(nms, low, high):
    H, W = nms.shape
    strong = nms >= high
    weak = (nms >= low) & ~strong
    edges = strong.copy()
    stack = list(zip(*np.nonzero(strong)))
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and weak[yy, xx] and not edges[yy, xx]:
                    edges[yy, xx] = True
                    stack.append((yy, xx))
    return edges


# повний цикл детектора країв Канні
def detect_edges(gray, sigma=1.4, low_k=0.08, high_k=0.2):
    smooth = gaussian_blur(gray, sigma)
    mag, ang = sobel(smooth)
    nms = non_max_suppression(mag, ang)
    max_grad = mag.max()
    edges = hysteresis(nms, low_k * max_grad, high_k * max_grad)
    return edges, mag


# --- 3. Перетворення Хафа для прямих ---

# будує акумулятор Хафа для прямих: rho = x*cos(theta) + y*sin(theta)
def hough_lines(edges, n_theta=180):
    H, W = edges.shape
    diag = int(np.ceil(np.hypot(H, W)))
    thetas = np.deg2rad(np.arange(n_theta) * 180.0 / n_theta)
    rhos = np.arange(-diag, diag + 1)
    acc = np.zeros((len(rhos), n_theta), dtype=np.int64)

    ys, xs = np.nonzero(edges)
    for t_idx, th in enumerate(thetas):
        r = np.round(xs * np.cos(th) + ys * np.sin(th)).astype(int) + diag
        acc[:, t_idx] = np.bincount(r, minlength=len(rhos))
    return acc, thetas, rhos


# зануляє околицю знайденого піка (з урахуванням періодичності theta)
def _suppress_line_peak(acc, r0, t0, rw, tw):
    n_rho, n_theta = acc.shape
    for dt in range(-tw, tw + 1):
        t = t0 + dt
        mirror = False
        if t < 0:
            t += n_theta
            mirror = True
        elif t >= n_theta:
            t -= n_theta
            mirror = True
        rc = (n_rho - 1 - r0) if mirror else r0
        lo, hi = max(0, rc - rw), min(n_rho, rc + rw + 1)
        acc[lo:hi, t] = 0


# жадібний пошук піків в акумуляторі прямих
def find_line_peaks(acc, thetas, rhos, threshold_k=0.5, max_lines=10, rw=8, tw=8):
    work = acc.copy()
    thr = threshold_k * work.max()
    found = []
    for _ in range(max_lines):
        idx = np.argmax(work)
        r_idx, t_idx = np.unravel_index(idx, work.shape)
        votes = work[r_idx, t_idx]
        if votes < thr or votes == 0:
            break
        found.append((rhos[r_idx], thetas[t_idx], int(votes)))
        _suppress_line_peak(work, r_idx, t_idx, rw, tw)
    return found


# малює знайдені прямі поверх зображення
def draw_lines(ax, lines, shape, color="lime"):
    H, W = shape
    for rho, th, _ in lines:
        c, s = np.cos(th), np.sin(th)
        pts = []
        if abs(s) > 1e-6:
            for x in (0, W - 1):
                y = (rho - x * c) / s
                if 0 <= y <= H - 1:
                    pts.append((x, y))
        if abs(c) > 1e-6:
            for y in (0, H - 1):
                x = (rho - y * s) / c
                if 0 <= x <= W - 1:
                    pts.append((x, y))
        if len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[-1]
            ax.plot([x1, x2], [y1, y2], color=color, lw=1.4)
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)


# --- 4. Перетворення Хафа для кіл ---

# будує 3D акумулятор (радіус, y, x) для пошуку кіл
def hough_circles(edges, r_min, r_max, r_step=1):
    H, W = edges.shape
    radii = np.arange(r_min, r_max + 1, r_step)
    acc = np.zeros((len(radii), H, W), dtype=np.float64)
    ys, xs = np.nonzero(edges)

    for k, r in enumerate(radii):
        n_ang = max(16, int(2 * np.pi * r))
        angles = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        a = np.round(xs[:, None] - r * cos_a[None, :]).astype(int).ravel()
        b = np.round(ys[:, None] - r * sin_a[None, :]).astype(int).ravel()
        ok = (a >= 0) & (a < W) & (b >= 0) & (b < H)

        flat = b[ok] * W + a[ok]
        layer = np.bincount(flat, minlength=H * W).reshape(H, W).astype(np.float64)
        layer = convolve2d(layer, np.ones((3, 3)))
        acc[k] = layer / (2 * np.pi * r)
    return acc, radii


# шукає локальні максимуми у 3D акумуляторі кіл
def find_circle_peaks(acc, radii, threshold=0.6, max_circles=10, xy_w=None):
    work = acc.copy()
    top = work.max()
    found = []
    for _ in range(max_circles):
        idx = np.argmax(work)
        k, y, x = np.unravel_index(idx, work.shape)
        val = work[k, y, x]
        if val < threshold * top:
            break
        r = radii[k]
        found.append((x, y, r, float(val)))

        w = xy_w if xy_w else max(5, int(r * 0.8))
        kr = int(0.4 * r) + 3
        work[max(0, k - kr):k + kr + 1,
             max(0, y - w):y + w + 1,
             max(0, x - w):x + w + 1] = 0
    return found


# малює знайдені кола та їхні центри
def draw_circles(ax, circles, shape, color="lime"):
    H, W = shape
    t = np.linspace(0, 2 * np.pi, 200)
    for x, y, r, _ in circles:
        ax.plot(x + r * np.cos(t), y + r * np.sin(t), color=color, lw=1.6)
        ax.plot([x], [y], "+", color="red", ms=6)
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)


# --- 5. Синтетика та шум ---

# генерує фіксований набір прямих на темному фоні
def synth_lines(size=300, n_lines=5):
    img = np.full((size, size), 30.0)
    yy, xx = np.mgrid[0:size, 0:size]
    truth = []
    thetas_deg = [10, 45, 80, 120, 150][:n_lines]
    rhos = [60, 110, 150, -20, 40]
    for th_d, rho in zip(thetas_deg, rhos):
        th = np.deg2rad(th_d)
        dist = np.abs(xx * np.cos(th) + yy * np.sin(th) - rho)
        img[dist < 1.5] = 230.0
        truth.append((rho, th))
    return img, truth


# генерує випадкові прямі, які гарантовано не зливаються в акумуляторі
def random_lines(size=300, n_lines=5, seed=None, min_dtheta=15, min_drho=30):
    local_rng = np.random.default_rng(seed)
    img = np.full((size, size), 30.0)
    yy, xx = np.mgrid[0:size, 0:size]
    truth = []
    while len(truth) < n_lines:
        x0, y0 = local_rng.uniform(0.1 * size, 0.9 * size, 2)
        th = np.deg2rad(local_rng.uniform(0, 180))
        rho = x0 * np.cos(th) + y0 * np.sin(th)
        ok = True
        for rho_t, th_t in truth:
            dth = np.rad2deg(abs(th - th_t))
            dth = min(dth, 180 - dth)
            if dth < min_dtheta and abs(rho - rho_t) < min_drho:
                ok = False
                break
        if ok:
            truth.append((rho, th))

    for rho, th in truth:
        dist = np.abs(xx * np.cos(th) + yy * np.sin(th) - rho)
        img[dist < 1.5] = 230.0
    return img, truth


# генерує кілька кіл на темному фоні
def synth_circles(size=300):
    img = np.full((size, size), 30.0)
    yy, xx = np.mgrid[0:size, 0:size]
    truth = [(80, 80, 40), (200, 90, 30), (150, 200, 55), (60, 230, 22)]
    for cx, cy, r in truth:
        dist = np.hypot(xx - cx, yy - cy)
        img[np.abs(dist - r) < 1.5] = 230.0
    return img, truth


# додає гаусівський шум до зображення
def add_gaussian_noise(img, sigma):
    noisy = img + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255)


# --- 6. Експерименти ---

# запускає пошук прямих і зберігає візуалізацію етапів
def run_lines(gray, title, fname, color=None, sigma=1.4, low_k=0.08, high_k=0.2,
              thr_k=0.5, max_lines=8):
    edges, _ = detect_edges(gray, sigma, low_k, high_k)
    acc, thetas, rhos = hough_lines(edges)
    lines = find_line_peaks(acc, thetas, rhos, thr_k, max_lines)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    if color is not None:
        axs[0].imshow(color.astype(np.uint8))
        axs[0].set_title("Кольорове")
    else:
        axs[0].imshow(gray, cmap="gray")
        axs[0].set_title("Напівтонове (із шумом)" if title.endswith("шум") else "Напівтонове")

    axs[1].imshow(edges, cmap="gray")
    axs[1].set_title(f"Краї ({int(edges.sum())} точок)")

    axs[2].imshow(acc, cmap="hot", aspect="auto", extent=[0, 180, rhos[-1], rhos[0]])
    axs[2].set_title("Акумулятор Хафа")
    axs[2].set_xlabel("theta, °")
    axs[2].set_ylabel("rho")

    axs[3].imshow(gray, cmap="gray")
    draw_lines(axs[3], lines, gray.shape)
    axs[3].set_title(f"Знайдено прямих: {len(lines)}")

    for a in (axs[0], axs[1], axs[3]):
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=110)
    plt.close(fig)

    return len(lines), int(edges.sum())


# запускає пошук кіл і зберігає візуалізацію етапів
def run_circles(gray, title, fname, r_range, color=None, sigma=1.6, low_k=0.08,
                high_k=0.2, thr=0.4, max_circles=8):
    edges, _ = detect_edges(gray, sigma, low_k, high_k)
    acc, radii = hough_circles(edges, *r_range)
    circles = find_circle_peaks(acc, radii, thr, max_circles)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    if color is not None:
        axs[0].imshow(color.astype(np.uint8))
        axs[0].set_title("Кольорове")
    else:
        axs[0].imshow(gray, cmap="gray")
        axs[0].set_title("Напівтонове (із шумом)" if title.endswith("шум") else "Напівтонове")

    axs[1].imshow(edges, cmap="gray")
    axs[1].set_title(f"Краї ({int(edges.sum())} точок)")

    axs[2].imshow(acc.max(axis=0), cmap="hot")
    axs[2].set_title("Акумулятор (max по r)")

    axs[3].imshow(gray, cmap="gray")
    draw_circles(axs[3], circles, gray.shape)
    axs[3].set_title(f"Знайдено кіл: {len(circles)}")

    for a in axs:
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=110)
    plt.close(fig)

    return len(circles), int(edges.sum())


# форматований вивід результатів в консоль
def print_row(name, n_found, n_edges):
    print(f"{name:<34} краї: {n_edges:>6}   знайдено: {n_found}")


def main():
    data_dir = os.path.join(os.path.dirname(skimage.__file__), "data")
    noise_sigma = 25
    results = {}

    # 1. Випадкові прямі (чисте зображення та з шумом)
    clean, _ = random_lines(300, 6, seed=11)
    noisy = add_gaussian_noise(clean, 40)

    for tag, img, sigma, label in [("clean", clean, 1.0, "без шуму"),
                                   ("noise", noisy, 2.0, "шум")]:
        n_found, n_edges = run_lines(
            img, f"Випадкові прямі, {label}", f"lines_synth_{tag}.png",
            sigma=sigma, thr_k=0.4, max_lines=9
        )
        print_row(f"synth / {tag}", n_found, n_edges)
        results[f"synth_{tag}"] = dict(lines=n_found, edges=n_edges)

    # 2. Кольорові фото для пошуку прямих
    photos = {
        "rocket": load_image(os.path.join(data_dir, "rocket.jpg")),
        "china": np.asarray(load_sample_image("china.jpg"), dtype=np.float64),
    }
    for base, rgb in photos.items():
        rgb = resize_nearest(rgb, 320)
        gray = to_gray(rgb)
        for tag, img, sigma, color_img, label in [
            ("clean", gray, 1.4, rgb, "без шуму"),
            ("noise", add_gaussian_noise(gray, noise_sigma), 2.0, None, "шум"),
        ]:
            n_found, n_edges = run_lines(
                img, f"{base}, {label}", f"lines_{base}_{tag}.png",
                color=color_img, sigma=sigma, thr_k=0.55, max_lines=20
            )
            print_row(f"{base} / {tag}", n_found, n_edges)
            results[f"{base}_{tag}"] = dict(lines=n_found, edges=n_edges)

    # 3. Пошук кіл на фото монет
    coins_rgb = resize_nearest(load_image(os.path.join(data_dir, "coins.png")), 200)
    coins_gray = to_gray(coins_rgb)
    r_max = int(min(coins_gray.shape) * 0.35)

    for tag, img, sigma, color_img, label in [
        ("clean", coins_gray, 1.6, coins_rgb, "без шуму"),
        ("noise", add_gaussian_noise(coins_gray, noise_sigma), 2.2, None, "шум"),
    ]:
        n_found, n_edges = run_circles(
            img, f"Монети, {label}", f"circles_coins_{tag}.png",
            (8, r_max), color=color_img, sigma=sigma, thr=0.7, max_circles=30
        )
        print_row(f"coins / {tag}", n_found, n_edges)
        results[f"coins_{tag}"] = dict(circles=n_found, edges=n_edges)

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("Готово, результати в", os.path.abspath(OUT_DIR))

    clean_c, _ = synth_circles(300)
    noisy_c = add_gaussian_noise(clean_c, 40)

    for tag, img, sigma, label in [
        ("clean", clean_c, 1.0, "без шуму"),
        ("noise", noisy_c, 2.0, "шум"),
    ]:
        n_found, n_edges = run_circles(
            img,
            f"Синтетичні кола, {label}",
            f"circles_synth_{tag}.png",
            r_range=(15, 65),
            sigma=sigma,
            thr=0.5,
            max_circles=8,
        )
        print_row(f"synth_circles / {tag}", n_found, n_edges)
        results[f"circles_synth_{tag}"] = dict(circles=n_found, edges=n_edges)

if __name__ == "__main__":
    main()