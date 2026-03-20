import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from time import perf_counter
from scipy.ndimage import convolve
import multiprocessing as mp
import functools

# Глобальні змінні
drawing = False
ix, iy = -1, -1
mask = None
radius = 5
interp_stats = {"ok": 0, "low": 0, "fail": 0}

def select_file(title="Виберіть зображення", filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*"))):
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return filepath

def load_images(image_path, mask_path):
    image = cv2.imread(image_path).astype(np.float32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) // 255
    return image, mask

def bicubic_weight(t):
    """Вагова функція для бікубічної інтерполяції."""
    t = np.abs(t)
    if t < 1:
        return 1.5 * t**3 - 2.5 * t**2 + 1
    elif t < 2:
        return -0.5 * t**3 + 2.5 * t**2 - 4 * t + 2
    return 0

def get_neighbors_bicubic(channel, mask, x, y, size=4):
    """Отримання сусідів у квадраті 4x4 для бікубічної інтерполяції."""
    neighbors = []
    positions = []
    h, w = channel.shape
    half_size = size // 2
    for j in range(-half_size, half_size):
        for i in range(-half_size, half_size):
            nx, ny = x + i, y + j
            if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] == 0:
                neighbors.append(channel[nx, ny])
                positions.append((i, j))
    return np.array(positions), np.array(neighbors)

def bicubic_interpolate(positions, values, x, y):
    """Бікубічна інтерполяція для одного пікселя."""
    if len(values) < 4:  # Потрібно щонайменше 4 сусіди
        return np.nan, "fail"
    sum_weights = 0
    sum_values = 0
    for (dx, dy), val in zip(positions, values):
        wx = bicubic_weight(dx - x)
        wy = bicubic_weight(dy - y)
        weight = wx * wy
        sum_weights += weight
        sum_values += weight * val
    if sum_weights == 0:
        return np.nan, "fail"
    val = sum_values / sum_weights
    return clamp(val), "ok"

def clamp(val, min_val=0, max_val=255):
    """Обмеження значення пікселя в діапазоні [0, 255]."""
    return max(min_val, min(max_val, val))

def preprocess_mask(mask):
    """Видаляє ізольовані пікселі з маски, які мають <2 сусідів."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # Виключаємо центральний піксель
    neighbor_count = convolve(mask, kernel, mode='constant', cval=0)
    return np.where((mask == 1) & (neighbor_count >= 2), 1, 0).astype(np.uint8)

def process_pixel_bicubic(args):
    """Обробка одного пікселя для багатопоточності."""
    channel, mask, x, y = args
    pos, vals = get_neighbors_bicubic(channel, mask, x, y)
    pred, stat = bicubic_interpolate(pos, vals, 0, 0)
    return x, y, pred, stat

def vectorized_process_pixels(image, mask):
    """Векторизована обробка пікселів за допомогою numpy."""
    h, w = mask.shape
    coords = np.where(mask == 1)
    y_coords, x_coords = coords[0], coords[1]
    results = []
    local_stats = {"ok": 0, "low": 0, "fail": 0}
    
    for y, x in zip(y_coords, x_coords):
        pos, vals = get_neighbors_bicubic(image[:, :, 0], mask, x, y)
        if len(vals) >= 4:
            pred_r, stat_r = bicubic_interpolate(pos, vals, 0, 0)
            local_stats[stat_r] += 1
            pred_g, _ = bicubic_interpolate(pos, get_neighbors_bicubic(image[:, :, 1], mask, x, y)[1], 0, 0)
            pred_b, _ = bicubic_interpolate(pos, get_neighbors_bicubic(image[:, :, 2], mask, x, y)[1], 0, 0)
            if not np.isnan(pred_r) and not np.isnan(pred_g) and not np.isnan(pred_b):
                results.append((x, y, np.array([pred_r, pred_g, pred_b])))
            else:
                results.append((x, y, None))
                local_stats["fail"] += 1
        else:
            results.append((x, y, None))
            local_stats["fail"] += 1
    return results, local_stats

def parallel_process_pixels(image, mask, num_processes=4):
    """Паралельна обробка пікселів за допомогою multiprocessing."""
    h, w = mask.shape
    coords = np.where(mask == 1)
    y_coords, x_coords = coords[0], coords[1]
    tasks = [(image[:, :, c], mask, x, y) for y, x in zip(y_coords, x_coords) for c in range(3)]
    results = []
    local_stats = {"ok": 0, "low": 0, "fail": 0}
    
    with mp.Pool(processes=num_processes) as pool:
        channel_results = pool.map(process_pixel_bicubic, tasks)
    
    pixel_results = {}
    for x, y, pred, stat in channel_results:
        if (x, y) not in pixel_results:
            pixel_results[(x, y)] = [None, None, None, {"ok": 0, "low": 0, "fail": 0}]
        idx = tasks.index((image[:, :, 0], mask, x, y)) % 3
        pixel_results[(x, y)][idx] = pred
        pixel_results[(x, y)][3][stat] += 1
    
    for (x, y), (r, g, b, stats) in pixel_results.items():
        for k in local_stats:
            local_stats[k] += stats[k] // 3
        if None not in [r, g, b] and not np.isnan(r) and not np.isnan(g) and not np.isnan(b):
            results.append((x, y, np.array([r, g, b])))
        else:
            results.append((x, y, None))
            local_stats["fail"] += 1
    
    return results, local_stats

def inpaint_image_color(image, mask, max_iter=30, parallel=False, num_processes=4):
    """Відновлення зображення з використанням бікубічної інтерполяції."""
    h, w, c = image.shape
    inpainted = image.copy()
    working_mask = preprocess_mask(mask)
    stats = {"ok": 0, "low": 0, "fail": 0}
    
    for iteration in range(max_iter):
        updated = False
        current_stats = {"ok": 0, "low": 0, "fail": 0}
        coords = np.where(working_mask == 1)
        if not coords[0].size:
            break
        
        if parallel:
            results, local_stats = parallel_process_pixels(inpainted, working_mask, num_processes)
        else:
            results, local_stats = vectorized_process_pixels(inpainted, working_mask)
        
        for x, y, result in results:
            if result is not None:
                inpainted[y, x] = result
                working_mask[y, x] = 0
                updated = True
            for k in current_stats:
                current_stats[k] += local_stats[k]
        
        for k in stats:
            stats[k] += current_stats[k]
        
        if not updated:
            break
    
    return inpainted, stats

def evaluate_reconstruction(original_undamaged, inpainted, mask):
    """Оцінка якості відновлення за допомогою SSIM і PSNR."""
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
    """Малювання маски пошкоджень мишею."""
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
    """Основна функція програми."""
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
    
    # Послідовна обробка
    print("🛠️ Запуск послідовної інтерполяції...")
    t0 = perf_counter()
    inpainted_seq, stats_seq = inpaint_image_color(image, refined_mask, parallel=False)
    t1 = perf_counter()
    seq_time = t1 - t0
    print(f"⏱️ Послідовна інтерполяція завершена за {seq_time:.2f} сек")
    print(f"📊 Статистика послідовної інтерполяції: {stats_seq}")
    ssim_seq, psnr_seq = evaluate_reconstruction(original_undamaged, inpainted_seq, refined_mask)
    if ssim_seq is not None and psnr_seq is not None:
        print(f"🔍 Послідовна: SSIM: {ssim_seq:.4f}, PSNR: {psnr_seq:.2f} dB")
    
    # Паралельна обробка
    print("🛠️ Запуск паралельної інтерполяції...")
    t0 = perf_counter()
    inpainted_par, stats_par = inpaint_image_color(image, refined_mask, parallel=True, num_processes=4)
    t1 = perf_counter()
    par_time = t1 - t0
    print(f"⏱️ Паралельна інтерполяція завершена за {par_time:.2f} сек")
    print(f"📊 Статистика паралельної інтерполяції: {stats_par}")
    ssim_par, psnr_par = evaluate_reconstruction(original_undamaged, inpainted_par, refined_mask)
    if ssim_par is not None and psnr_par is not None:
        print(f"🔍 Паралельна: SSIM: {ssim_par:.4f}, PSNR: {psnr_par:.2f} dB")
    
    # Порівняння прискорення
    speedup = seq_time / par_time if par_time > 0 else float('inf')
    print(f"🚀 Прискорення: {speedup:.2f}x")
    
    # Візуалізація
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(cv2.cvtColor(original_undamaged.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Оригінальне зображення")
    axes[0, 1].imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Пошкоджене зображення")
    axes[1, 0].imshow(refined_mask * 255, cmap='gray')
    axes[1, 0].set_title("Маска")
    axes[1, 1].imshow(cv2.cvtColor(inpainted_par.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Відновлене зображення (паралельне)")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    
    # Збереження результатів
    cv2.imwrite("inpainted_sequential.png", inpainted_seq.astype(np.uint8))
    cv2.imwrite("inpainted_parallel.png", inpainted_par.astype(np.uint8))
    print("✅ Зображення збережено як 'inpainted_sequential.png' та 'inpainted_parallel.png'")

if __name__ == "__main__":
    main()