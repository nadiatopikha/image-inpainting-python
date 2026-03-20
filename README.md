# Parallel Image Inpainting with Cubic Spline Interpolation

This project implements image restoration (inpainting) using cubic spline interpolation in Python. It includes both sequential and parallel implementations and compares their performance and quality.

---

## Overview

The goal of this project is to restore damaged images by reconstructing missing or corrupted pixels.

Two approaches are implemented:

- Sequential algorithm
- Parallel algorithm

The project evaluates both methods in terms of:
- execution time
- image quality (SSIM, PSNR)

---

## Features

- Image damage simulation (random missing regions)
- Image restoration using cubic spline interpolation
- Sequential and parallel implementations
- Performance comparison
- Image quality evaluation using:
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)
- Visualization of results

---

## Technologies Used

- Python
- NumPy
- OpenCV
- SciPy
- scikit-image
- Matplotlib
- PIL

---

## Project Structure

parallel-image-inpainting/
├── random_damage.py
├── inpaint_sequential.py
├── inpaint_parallel.py
├── requirements.txt
└── screenshots/

---

## How It Works

1. The original image is loaded  
2. Random damage is applied to simulate missing data  
3. The image is restored using:
   - sequential interpolation  
   - parallel interpolation  
4. Results are compared using SSIM and PSNR metrics  

---

## Results

- Parallel implementation provides faster execution (~3x speedup)  
- Image quality remains high for both methods  

---

## Performance Comparison

Sequential: ~0.50 sec  
Parallel: ~0.16 sec  
Speedup: ~3.13x  

---

## Visualization Examples

![Original](screenshots/original_image.jpg)  
![Damaged](screenshots/damaged_image.png)  
![Restored Parallel](screenshots/restored-parallel.png)  
![Restored Sequential](screenshots/sequential_restored_image.png)  

---

## How to Run

1. Clone the repository:  
git clone https://github.com/nadiatopikha/parallel-image-inpainting.git  

2. Install dependencies:  
pip install -r requirements.txt  

3. Run scripts:  
python random_damage.py  
python inpaint_sequential.py  
python inpaint_parallel.py  

---

## 🇺🇦 Опис українською

Цей проєкт демонструє відновлення пошкоджених зображень (inpainting) за допомогою кубічної сплайн-інтерполяції.

Реалізовано:

- генерацію пошкоджених зображень  
- послідовний алгоритм відновлення  
- паралельний алгоритм відновлення  
- порівняння швидкодії алгоритмів  
- оцінку якості відновлення за метриками SSIM та PSNR  

Проєкт показує, як використання паралельних обчислень дозволяє значно прискорити обробку зображень без втрати якості.
