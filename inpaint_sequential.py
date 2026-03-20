import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from time import perf_counter
from scipy.ndimage import convolve

# Глобальні змінні
drawing = False
ix, iy = -1, -1
mask = None
radius = 5
interp_stats = {"low": 0, "ok": 0, "fail": 0}

def select_file(title="Виберіть зображення", filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*"))):
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return filepath

def load_images(image_path, mask_path):
    image = cv2.imread(image_path).astype(np.float32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) // 255
    return image, mask

def get_neighbors(channel, mask, x, y, direction, max_radius=5):
    neighbors = []
    positions = []
    dx, dy = direction
    for r in range(1, max_radius+1):
        for sign in [1, -1]:
            nx, ny = x + dx * r * sign, y + dy * r * sign
            if 0 <= nx < channel.shape[0] and 0 <= ny < channel.shape[1]:
                if mask[nx, ny] == 0:
                    neighbors.append(channel[nx, ny])
                    positions.append(r * sign)
    if len(neighbors) < 2:
        return None, None
    sorted_data = sorted(zip(positions, neighbors))
    positions, neighbors = zip(*sorted_data)
    return np.array(positions), np.array(neighbors)

def spline_predict(positions, values):
    n = len(values)
    if n < 3:
        return np.nan, "fail"
    t = 0
    h = (positions[-1] - positions[0]) / (n - 1)
    x = (2 / h) * (t * h)
    if abs(x) > 1:
        return np.nan, "fail"
    if n == 3:
        p0, p1, p2 = values
        val = (
            (1/8)*(p0 - 2*p1 + p2)*x**2 +
            (1/4)*(-p0 + p2)*x +
            (1/8)*(p0 + 6*p1 + p2)
        )
        return clamp(val), "ok"
    elif n == 4:
        p0, p1, p2, p3 = values
        val = (
            (1/48)*(-p0 + 3*p1 - 3*p2 + p3)*x**3 +
            (1/16)*(p0 - p1 - p2 + p3)*x**2 +
            (1/16)*(-p0 - 5*p1 + 5*p2 + p3)*x +
            (1/48)*(p0 + 23*p1 + 23*p2 + p3)
        )
        return clamp(val), "ok"
    elif n == 5:
        p_2, p_1, p0, p1, p2 = values
        val = (
            (1/384)*(p_2 - 4*p_1 + 6*p0 - 4*p1 + p2)*x**4 +
            (1/96)*(-p_2 + 2*p_1 - 2*p1 + p2)*x**3 +
            (1/64)*(p_2 + 4*p_1 - 10*p0 + 4*p1 + p2)*x**2 +
            (1/96)*(-p_2 - 22*p_1 + 22*p1 + p2)*x +
            (1/384)*(p_2 + 76*p_1 + 230*p0 + 76*p1 + p2)
        )
        return clamp(val), "ok"
    else:
        return np.nan, "low"

def clamp(val, min_val=0, max_val=255):
    return max(min_val, min(max_val, val))

def upscale_image_quadratic(image, expand_width=True, expand_height=True):
    h, w, c = image.shape
    new_h = h * 2 if expand_height else h
    new_w = w * 2 if expand_width else w
    result = np.zeros((new_h, new_w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            orig_pixel = image[y, x]
            result[y * (2 if expand_height else 1), x * (2 if expand_width else 1)] = orig_pixel
            if expand_width and x < w - 1:
                next_pixel = image[y, x + 1]
                new_pixel = [clamp(int((-p + 7*p + 7*n - n) / 12)) for p, n in zip(orig_pixel, next_pixel)]
                result[y * (2 if expand_height else 1), x * 2 + 1] = new_pixel
            if expand_height and y < h - 1:
                next_row_pixel = image[y + 1, x]
                new_pixel = [clamp(int((-p + 7*p + 7*n - n) / 12)) for p, n in zip(orig_pixel, next_row_pixel)]
                result[y * 2 + 1, x * (2 if expand_width else 1)] = new_pixel
            if expand_width and expand_height and x < w - 1 and y < h - 1:
                next_pixel = image[y, x + 1]
                next_row_pixel = image[y + 1, x]
                diagonal_pixel = image[y + 1, x + 1]
                new_pixel = [clamp(int((-p + 7*np + 7*nrp - dp) / 12)) for p, np, nrp, dp in zip(orig_pixel, next_pixel, next_row_pixel, diagonal_pixel)]
                result[y * 2 + 1, x * 2 + 1] = new_pixel
    return result

def interpolate_pixel(channel, mask, x, y):
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    preds = []
    local_stats = {"low": 0, "ok": 0, "fail": 0}
    for dx, dy in directions:
        pos, vals = get_neighbors(channel, mask, x, y, (dx, dy))
        if pos is not None:
            pred, stat = spline_predict(pos, vals)
            local_stats[stat] += 1
        else:
            pred = np.nan
            local_stats["fail"] += 1
        preds.append(pred)
    return fuse_predictions(preds), local_stats

def fuse_predictions(preds):
    preds = np.array(preds)
    valid = ~np.isnan(preds)
    valid_preds = preds[valid]
    if len(valid_preds) == 0:
        return 0
    elif len(valid_preds) == 1:
        return valid_preds[0]
    else:
        sorted_preds = np.sort(valid_preds)
        diffs = np.diff(sorted_preds)
        if len(diffs) >= 3:
            max_diff_idx = np.argmax(diffs)
            if max_diff_idx == 0:
                sorted_preds = sorted_preds[1:]
            elif max_diff_idx == len(diffs) - 1:
                sorted_preds = sorted_preds[:-1]
        return np.mean(sorted_preds)

def preprocess_mask(mask):
    """Видаляє ізольовані пікселі з маски, які мають <2 сусідів."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # Виключаємо центральний піксель
    neighbor_count = convolve(mask, kernel, mode='constant', cval=0)
    return np.where((mask == 1) & (neighbor_count >= 2), 1, 0).astype(np.uint8)

def inpaint_image_color(image, mask, max_iter=30):
    h, w, c = image.shape
    inpainted = image.copy()
    working_mask = preprocess_mask(mask)  # Попередня обробка маски
    stats = {"ok": 0, "low": 0, "fail": 0}
    
    for iteration in range(max_iter):
        updated = False
        current_stats = {"ok": 0, "low": 0, "fail": 0}
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if working_mask[y, x] == 1:
                    neighbors = []
                    for j in range(-1, 2):
                        for i in range(-1, 2):
                            if i == 0 and j == 0:
                                continue
                            ny, nx = y + j, x + i
                            if 0 <= ny < h and 0 <= nx < w and working_mask[ny, nx] == 0:
                                neighbors.append(inpainted[ny, nx])
                    if len(neighbors) >= 3:
                        inpainted[y, x] = np.mean(neighbors, axis=0)
                        working_mask[y, x] = 0
                        current_stats["ok"] += 1
                        updated = True
                    elif len(neighbors) == 2:
                        inpainted[y, x] = np.mean(neighbors, axis=0)
                        working_mask[y, x] = 0
                        current_stats["low"] += 1
                        updated = True
                    else:
                        current_stats["fail"] += 1
        
        for k in stats:
            stats[k] += current_stats[k]
        
        if not updated:
            break
    
    return inpainted, stats

def evaluate_reconstruction(original_undamaged, inpainted, mask):
    original_undamaged = original_undamaged.astype(np.float32)
    inpainted = inpainted.astype(np.float32)
    mask = mask.astype(np.bool_)
    orig_pixels = original_undamaged[mask]
    inpaint_pixels = inpainted[mask]
    if len(orig_pixels) == 0:
        print("⚠️ Маска порожня, оцінка неможлива.")
        return None, None
    ssim_val = ssim(orig_pixels.reshape(-1, 3), inpaint_pixels.reshape(-1, 3), channel_axis=1, data_range=255)
    psnr_val = psnr(orig_pixels.reshape(-1, 3), inpaint_pixels.reshape(-1, 3), data_range=255)
    return ssim_val, psnr_val

def draw_mask(event, x, y, flags, param):
    global drawing, ix, iy, mask, image_for_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask, (x, y), radius, 1, -1)
        cv2.circle(image_for_drawing, (x, y), radius, (0, 0, 255), -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), radius, 1, -1)
            cv2.circle(image_for_drawing, (x, y), radius, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), radius, 1, -1)
        cv2.circle(image_for_drawing, (x, y), radius, (0, 0, 255), -1)

def main():
    global mask, image_for_drawing
    print("🔍 Виберіть оригінальне зображення (до пошкодження)...")
    original_path = select_file(title="Виберіть оригінальне зображення")
    if not original_path:
        print("❌ Не обрано оригінальне зображення.")
        return
    print("🔍 Виберіть пошкоджене зображення...")
    image_path = select_file(title="Виберіть пошкоджене зображення")
    if not image_path:
        print("❌ Не обрано пошкоджене зображення.")
        return
    original_undamaged = cv2.imread(original_path).astype(np.float32)
    image = cv2.imread(image_path).astype(np.float32)
    if original_undamaged.shape != image.shape:
        print("❌ Розміри зображень не збігаються.")
        return
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    image_for_drawing = image.copy().astype(np.uint8)
    print("✏️ Малюйте пошкоджені ділянки мишею (ліва кнопка). Натисніть 'Enter' для підтвердження.")
    cv2.namedWindow("Малювання маски")
    cv2.setMouseCallback("Малювання маски", draw_mask)
    while True:
        cv2.imshow("Малювання маски", image_for_drawing)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
        elif key == 27:  # ESC
            print("❌ Скасовано.")
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()
    print("🔎 Аналіз пошкоджень у виділеній області...")
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    damage_threshold = 30
    refined_mask = np.where((mask == 1) & (gray < damage_threshold), 1, 0).astype(np.uint8)
    num_refined = np.count_nonzero(refined_mask)
    print(f"✅ Виявлено потенційно пошкоджених пікселів: {num_refined}")
    print("🛠️ Запуск інтерполяції(послідовна)...")
    t0 = perf_counter()
    inpainted, stats = inpaint_image_color(image, refined_mask)
    t1 = perf_counter()
    interp_time = t1 - t0
    print(f"⏱️ Інтерполяція завершена за {interp_time:.2f} сек")
    print(f"📊 Статистика інтерполяції: {stats}")
    print("🔍 Оцінка якості...")
    ssim_val, psnr_val = evaluate_reconstruction(original_undamaged, inpainted, refined_mask)
    if ssim_val is not None and psnr_val is not None:
        print(f"🔍 SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(original_undamaged.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axes[0].set_title("Оригінальне зображення(без пошкоджень)")
    axes[1].imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axes[1].set_title("Пошкоджене зображення")
    axes[2].imshow(refined_mask * 255, cmap='gray')
    axes[2].set_title("Маска")
    axes[3].imshow(cv2.cvtColor(inpainted.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axes[3].set_title("Відновлене зображення")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    filename = "inpainted_by_user.png"
    cv2.imwrite(filename, inpainted.astype(np.uint8))
    print(f"✅ Збережено як '{filename}'")

if __name__ == "__main__":
    main()