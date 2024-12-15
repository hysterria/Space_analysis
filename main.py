import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label, gaussian_filter
import csv
import threading


def select_input_folder():
    global input_folder
    input_folder = filedialog.askdirectory(title="Выберите папку с изображениями")
    input_folder_label.config(text=f"Выбрана папка: {input_folder}")


def select_output_folder():
    global output_folder
    output_folder = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
    output_folder_label.config(text=f"Выбрана папка: {output_folder}")


def analyze_image_block(image_block, block_coords, threshold=100, blur_sigma=1):
    smoothed_block = gaussian_filter(image_block, sigma=blur_sigma)
    binary_block = (smoothed_block > threshold).astype(int)
    labeled_block, num_features = label(binary_block)

    objects = []
    for obj_id in range(1, num_features + 1):
        coords = np.argwhere(labeled_block == obj_id)
        if coords.size == 0:
            continue

        x_coords, y_coords = coords[:, 0] + block_coords[0], coords[:, 1] + block_coords[1]
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()

        width, height = max_y - min_y + 1, max_x - min_x + 1
        size = len(coords)
        brightness = image_block[coords[:, 0], coords[:, 1]].mean()


        objects.append({
            "x_center": (min_x + max_x) // 2,
            "y_center": (min_y + max_y) // 2,
            "width": width,
            "height": height,
            "size": size,
            "brightness": brightness
        })

    return objects


def analyze_image_parallel(image_array):
    height, width = image_array.shape
    num_blocks = cpu_count()
    block_height = height // num_blocks
    all_objects = []

    for i in range(num_blocks):
        start_row = i * block_height
        end_row = height if i == num_blocks - 1 else (i + 1) * block_height
        block = image_array[start_row:end_row, :]
        block_coords = (start_row, 0)

        objects = analyze_image_block(block, block_coords)
        all_objects.extend(objects)

    return all_objects


def process_single_image(args):
    image_path, output_folder = args
    try:
        image_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_processed.png")
        stats_file_path = os.path.join(output_folder, "statistics.csv")

        with Image.open(image_path) as img:
            img_gray = img.convert("L")
            image_array = np.array(img_gray)

        results = analyze_image_parallel(image_array)

        draw = ImageDraw.Draw(img_gray)
        for obj in results:
            x, y = obj["x_center"], obj["y_center"]
            width, height = obj["width"], obj["height"]
            obj_type = str(obj["brightness"])

            draw.rectangle([y - width // 2, x - height // 2, y + width // 2, x + height // 2], outline="red", width=2)
            draw.text((y + 5, x - 10), obj_type, fill="white")

        img_gray.save(output_image_path)

        with open(stats_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for obj in results:
                writer.writerow([image_name, obj["x_center"], obj["y_center"],
                                 obj["width"], obj["height"], obj["size"],
                                 obj["brightness"]])

        return f"Обработано: {image_name}, найдено объектов: {len(results)}"

    except Exception as e:
        return f"Ошибка при обработке {image_path}: {str(e)}"


def process_images_in_parallel(image_paths, output_folder, progress_callback):
    def update_progress(result):
        progress_callback(result)

    args_list = [(path, output_folder) for path in image_paths]

    with Pool(cpu_count()) as pool:
        for result in pool.imap_unordered(process_single_image, args_list):
            update_progress(result)


def start_processing():
    if not input_folder or not output_folder:
        messagebox.showerror("Ошибка", "Выберите обе папки для обработки!")
        return

    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                   if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

    if not image_files:
        messagebox.showinfo("Информация", "В выбранной папке нет изображений!")
        return

    stats_file_path = os.path.join(output_folder, "statistics.csv")
    with open(stats_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "X Center", "Y Center", "Width", "Height", "Size", "Brightness"])

    progress_bar["maximum"] = len(image_files)
    progress_bar["value"] = 0
    status_text.delete("1.0", tk.END)

    def worker():
        def progress_callback(result):
            status_text.insert(tk.END, result + "\n")
            status_text.see(tk.END)
            progress_bar["value"] += 1

        try:
            process_images_in_parallel(image_files, output_folder, progress_callback)
            messagebox.showinfo("Готово", "Обработка завершена!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")

    thread = threading.Thread(target=worker)
    thread.start()


if __name__ == "__main__":
    input_folder = ""
    output_folder = ""

    root = tk.Tk()
    root.title("Обработка изображений")

    frame = tk.Frame(root)
    frame.pack(pady=20)

    input_folder_button = tk.Button(frame, text="Выбрать папку с изображениями", command=select_input_folder)
    input_folder_button.grid(row=0, column=0, padx=10, pady=5)

    output_folder_button = tk.Button(frame, text="Выбрать папку для сохранения", command=select_output_folder)
    output_folder_button.grid(row=1, column=0, padx=10, pady=5)

    input_folder_label = tk.Label(frame, text="Папка не выбрана")
    input_folder_label.grid(row=0, column=1, padx=10, pady=5)

    output_folder_label = tk.Label(frame, text="Папка не выбрана")
    output_folder_label.grid(row=1, column=1, padx=10, pady=5)

    process_button = tk.Button(root, text="Запустить обработку", command=start_processing)
    process_button.pack(pady=10)

    progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=400)
    progress_bar.pack(pady=10)

    status_text = tk.Text(root, height=10, width=60)
    status_text.pack(pady=10)

    root.mainloop()

