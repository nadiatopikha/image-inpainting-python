import cv2
import numpy as np
import random
import os

def random_damage(image, num_lines=5, line_thickness=3, num_spots=2, spot_size=(10, 10)):
    damaged = image.copy()
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    h, w = image.shape[:2]

    # Рандомні лінії
    for _ in range(num_lines):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        color = (0, 0, 0)  # Чорна лінія
        cv2.line(damaged, (x1, y1), (x2, y2), color, thickness=line_thickness)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=line_thickness)

    # Рандомні плями
    for _ in range(num_spots):
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(*spot_size)
        cv2.circle(damaged, (cx, cy), radius, (0, 0, 0), -1)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

    return damaged, mask

# ======= Тестування =======
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tkinter import Tk, filedialog

    def select_image():
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Виберіть зображення", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        return file_path

    print("🔍 Виберіть оригінальне зображення для пошкодження...")
    img_path = select_image()
    if not img_path:
        print("❌ Зображення не вибрано.")
        exit()

    img = cv2.imread(img_path)
    damaged, mask = random_damage(img)

    # Зберегти результати
    cv2.imwrite("damaged_image.png", damaged)
    cv2.imwrite("damage_mask.png", mask)

    # Показати результат
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(damaged, cv2.COLOR_BGR2RGB))
    plt.title("Damaged")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("Damage Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    print("✅ Збережено 'damaged_image.png' та 'damage_mask.png'")
