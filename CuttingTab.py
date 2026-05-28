import ttkbootstrap as ttk
import tkinter.filedialog as filedialog
from tkinter import messagebox
import rasterio
import geopandas as gpd
from PIL import Image
import os
import numpy as np
import logging
import threading
import tkinter as tk
from matplotlib.path import Path

try:
    from skimage.measure import label as sk_label, regionprops  # type: ignore[reportMissingImports]
except ModuleNotFoundError:
    sk_label = None

    class _SimpleRegion:
        def __init__(self, coords):
            self.area = len(coords)
            rows = [coord[0] for coord in coords]
            cols = [coord[1] for coord in coords]
            self.centroid = (float(np.mean(rows)), float(np.mean(cols)))

    def regionprops(labeled):
        regions = []
        for region_id in np.unique(labeled):
            if region_id == 0:
                continue
            coords = np.argwhere(labeled == region_id)
            if coords.size > 0:
                regions.append(_SimpleRegion(coords))
        return regions

    def sk_label(mask):
        """Fallback connected-component labeling when scikit-image is unavailable."""
        binary = np.asarray(mask).astype(bool)
        labeled = np.zeros(binary.shape, dtype=np.int32)
        current_label = 0
        height, width = binary.shape

        for start_y in range(height):
            for start_x in range(width):
                if not binary[start_y, start_x] or labeled[start_y, start_x] != 0:
                    continue

                current_label += 1
                stack = [(start_y, start_x)]
                labeled[start_y, start_x] = current_label

                while stack:
                    y, x = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if (
                                0 <= ny < height and 0 <= nx < width
                                and binary[ny, nx]
                                and labeled[ny, nx] == 0
                            ):
                                labeled[ny, nx] = current_label
                                stack.append((ny, nx))

        return labeled


class ProgressWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("处理进度")
        self.geometry("400x200")
        self.stop_event = threading.Event()
        self.detail_visible = False

        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.progress_frame = ttk.Frame(container)
        self.progress_frame.pack(fill=tk.X, pady=5)

        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='determinate',
            length=300
        )
        self.progress_bar.pack(fill=tk.X)

        self.stats_frame = ttk.Frame(container)
        self.stats_frame.pack(fill=tk.X, pady=5)

        self.percentage_label = ttk.Label(
            self.stats_frame,
            text="0.0%",
            font=("微软雅黑", 10, "bold"),
            foreground="#2c3e50"
        )
        self.percentage_label.pack(side=tk.LEFT)

        self.count_label = ttk.Label(
            self.stats_frame,
            text="已处理: 0/0",
            font=("微软雅黑", 9),
            foreground="#7f8c8d"
        )
        self.count_label.pack(side=tk.RIGHT)

        self.btn_frame = ttk.Frame(container)
        self.btn_frame.pack(fill=tk.X, pady=10)

        self.stop_btn = ttk.Button(
            self.btn_frame,
            text="强制停止",
            command=self.on_stop,
            style='danger.TButton',
            width=12
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.detail_btn = ttk.Button(
            self.btn_frame,
            text="查看详情",
            command=self.toggle_detail,
            style='info.Outline.TButton',
            width=12
        )
        self.detail_btn.pack(side=tk.RIGHT, padx=5)

        self.detail_text = tk.Text(
            container,
            height=6,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.center_window(parent)

    def center_window(self, parent):
        self.update_idletasks()
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()

        x = parent_x + (parent_width // 2 - self.winfo_width() // 2)
        y = parent_y + (parent_height // 2 - self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def on_stop(self):
        self.stop_event.set()
        self.stop_btn.config(state='disabled')
        self.append_detail("用户请求停止处理...")

    def toggle_detail(self):
        self.detail_visible = not self.detail_visible
        if self.detail_visible:
            self.detail_text.pack(fill=tk.BOTH, expand=True)
            self.geometry("400x400")
            self.detail_btn.config(text="隐藏详情")
        else:
            self.detail_text.pack_forget()
            self.geometry("400x200")
            self.detail_btn.config(text="查看详情")

    def append_detail(self, message):
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.insert(tk.END, f"• {message}\n")
        self.detail_text.see(tk.END)
        self.detail_text.config(state=tk.DISABLED)

    def update_progress(self, current, total):
        percent = current / total * 100 if total > 0 else 0
        self.progress_bar['value'] = percent
        self.percentage_label.config(
            text=f"{percent:.1f}%",
            foreground="#2980b9" if percent < 100 else "#27ae60"
        )
        self.count_label.config(
            text=f"已处理: {current}/{total}",
            foreground="#27ae60" if percent == 100 else "#2c3e50"
        )
        if self.detail_visible:
            self.detail_text.see(tk.END)


class CuttingTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.tiff_path = tk.StringVar()
        self.shp_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.color_map = {}
        self.create_widgets()

    def create_widgets(self):
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        file_frame = tk.LabelFrame(left_frame, text="文件设置")
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Label(file_frame, text="TIFF文件：").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(file_frame, textvariable=self.tiff_path).grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(file_frame, text="浏览", command=self.select_tiff).grid(row=0, column=2, padx=5)
        ttk.Label(file_frame, text="SHP文件：").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(file_frame, textvariable=self.shp_path).grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(file_frame, text="浏览", command=self.select_shp).grid(row=1, column=2, padx=5)
        ttk.Label(file_frame, text="输出目录：").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(file_frame, textvariable=self.output_dir).grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(file_frame, text="浏览", command=self.select_output).grid(row=2, column=2, padx=5)

        param_frame = tk.LabelFrame(left_frame, text="处理参数")
        param_frame.pack(fill=tk.X, pady=5)
        ttk.Label(param_frame, text="起始编号:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.start_number_entry = ttk.Entry(param_frame, width=10)
        self.start_number_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.start_number_entry.insert(0, "0")

        ttk.Label(param_frame, text="颜色数量:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.color_count_entry = ttk.Entry(param_frame, width=10)
        self.color_count_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.color_count_entry.insert(0, "20")

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 5))

        self.process_btn = ttk.Button(btn_frame,
                                    text="开始裁剪 ※",
                                    style="success.TButton",
                                    command=self.start_processing)
        self.process_btn.pack(fill=tk.X, ipady=10)

        ttk.Label(btn_frame,
                  text="快捷键：Ctrl + J",
                  font=("微软雅黑", 9)).pack(pady=(5, 0))

        self.bind_all('<Control-j>', lambda e: self.start_processing())

        folder_frame = tk.LabelFrame(right_frame, text="输出预览")
        folder_frame.pack(fill=tk.BOTH, expand=True)

        self.folders_text = tk.Text(folder_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(folder_frame, command=self.folders_text.yview)
        self.folders_text.configure(yscrollcommand=scrollbar.set)

        self.folders_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def select_tiff(self):
        path = filedialog.askopenfilename(
            title="选择TIFF文件",
            filetypes=[("TIFF files", "*.tiff *.tif")]
        )
        if path:
            self.tiff_path.set(path)

    def select_shp(self):
        path = filedialog.askopenfilename(
            title="选择SHP文件",
            filetypes=[("SHP files", "*.shp")]
        )
        if path:
            self.shp_path.set(path)

    def select_output(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_dir.set(path)

    def validate_inputs(self):
        required = {
            "TIFF文件": self.tiff_path.get(),
            "SHP文件": self.shp_path.get(),
            "输出目录": self.output_dir.get()
        }

        for name, value in required.items():
            if not value:
                messagebox.showerror("错误", f"请选择{name}")
                return False

        try:
            int(self.start_number_entry.get())
            color_count = int(self.color_count_entry.get())
            if not 1 <= color_count <= 255:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "起始编号必须为整数，颜色数量为1-255之间的整数")
            return False

        return True

    def start_processing(self):
        if self.validate_inputs():
            self.progress_window = ProgressWindow(self)
            self.progress_window.progress_bar.start()

            threading.Thread(
                target=self.process_files_wrapper,
                args=(self.progress_window.stop_event,),daemon=True).start()

    def process_files_wrapper(self, stop_event):
        try:
            start_number = int(self.start_number_entry.get())
            color_count = int(self.color_count_entry.get())
            self.process_btn.config(state="disabled")
            self.process_files(
                self.tiff_path.get(),
                self.shp_path.get(),
                self.output_dir.get(),
                start_number,
                color_count,
                stop_event
            )
        except Exception as e:
            self.progress_window.append_detail(f"处理错误: {str(e)}")
        finally:
            self.after(0, lambda: [
                self.process_btn.config(state="normal"),
                self.progress_window.progress_bar.stop(),
                self.progress_window.destroy()
            ])

    def shp_to_mask(self, gdf, out_shape, name_to_id, offset, transform):
        mask = np.zeros(out_shape, dtype=np.uint8)
        x_off, y_off = offset
        width, height = out_shape

        x_min, y_max = rasterio.transform.xy(transform, y_off, x_off)
        x_max, y_min = rasterio.transform.xy(transform, y_off + height, x_off + width)

        gdf_slice = gdf.cx[x_min:x_max, y_min:y_max]

        for idx, feature in gdf_slice.iterrows():
            geom = feature.geometry
            class_name = feature['类别'] if '类别' in feature else 'class_1'
            class_id = name_to_id.get(class_name, 0)

            if geom.geom_type == 'Polygon':
                self._rasterize_polygon(geom, mask, class_id, offset, transform)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    self._rasterize_polygon(poly, mask, class_id, offset, transform)
        return mask

    def _rasterize_polygon(self, polygon, mask, class_id, offset, transform):
        x_off, y_off = offset
        coords = np.array(polygon.exterior.coords)
        rows, cols = rasterio.transform.rowcol(transform, coords[:, 0], coords[:, 1])
        rows = rows - y_off
        cols = cols - x_off

        if not (np.any((rows >= 0) & (rows < mask.shape[0])) and
                np.any((cols >= 0) & (cols < mask.shape[1]))):
            return

        path = Path(np.column_stack((cols, rows)))
        y, x = np.mgrid[:mask.shape[0], :mask.shape[1]]
        points = np.vstack((x.ravel(), y.ravel())).T
        mask_points = path.contains_points(points)
        mask_coords = np.unravel_index(np.where(mask_points)[0], mask.shape)
        mask[mask_coords] = class_id

    def process_files(self, tiff_path, shp_path, output_dir, start_number, color_count, stop_event, tile_size=256,
                      overlap=128):
        with rasterio.open(tiff_path) as src:
            width, height = src.width, src.height
            num_bands = src.count
            transform = src.transform

            # 读取影像数据
            if num_bands >= 3:
                tiff_data = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                self.progress_window.append_detail(f"读取{num_bands}波段TIFF文件")
            else:
                tiff_data = np.expand_dims(src.read(1), axis=2)
                self.progress_window.append_detail("读取单波段TIFF文件")

            # 处理无效值
            nodata = src.nodata
            if nodata is not None:
                if num_bands >= 3:
                    for band in range(3):
                        tiff_data[:, :, band] = np.where(tiff_data[:, :, band] == nodata, np.nan, tiff_data[:, :, band])
                else:
                    tiff_data = np.where(tiff_data == nodata, np.nan, tiff_data)

            # 百分比拉伸增强
            def normalize_percentile(band_data, lower=2, upper=98):
                valid_data = band_data[~np.isnan(band_data)]
                if valid_data.size == 0:
                    return np.zeros_like(band_data)
                low_val = np.percentile(valid_data, lower)
                high_val = np.percentile(valid_data, upper)
                return np.clip((band_data - low_val) / (high_val - low_val + 1e-8), 0, 1)

            if tiff_data.dtype in [np.float32, np.float64]:
                if num_bands >= 3:
                    for band in range(3):
                        tiff_data[:, :, band] = normalize_percentile(tiff_data[:, :, band])
                    tiff_data = (np.nan_to_num(tiff_data) * 255).astype(np.uint8)
                else:
                    tiff_data = (normalize_percentile(tiff_data) * 255).astype(np.uint8)

            # 读取矢量数据
            gdf = gpd.read_file(shp_path)
            if '类别' not in gdf.columns:
                self.progress_window.append_detail("警告: 使用默认类别class_1")
                gdf['类别'] = 'class_1'

            # 获取关键波段用于特征计算（仅用于目标热力图，不影响标签）
            try:
                red_band = src.read(gdf['红波段索引'].iloc[0])
                nir_band = src.read(gdf['近红外波段索引'].iloc[0])
                red_edge1 = src.read(gdf['红边1波段索引'].iloc[0])
                red_edge2 = src.read(gdf['红边2波段索引'].iloc[0])
            except Exception:
                # 默认波段映射
                red_band = src.read(3) if src.count >= 3 else src.read(1)
                nir_band = src.read(4) if src.count >= 4 else src.read(1)
                red_edge1 = src.read(5) if src.count >= 5 else src.read(1)
                red_edge2 = src.read(6) if src.count >= 6 else src.read(1)

            # 计算特征指数（用于补充采样）
            ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
            slope = (red_edge2 - red_edge1) / 40  # 假设波段间隔40nm
            target_map = (ndvi > 0.6) & (slope > 0.15)

            # === 固定类别 → 灰度值映射（支持特殊类别）===
            all_classes = set(gdf['类别'].dropna().unique())

            # 预设特殊类别
            name_to_id = {}
            used_ids = set()

            # 背景类 "0" → 0
            if "0" in all_classes:
                name_to_id["0"] = 0
                used_ids.add(0)

            # "加黄" → 浅灰色
            if "加黄" in all_classes:
                name_to_id["加黄"] = 200
                used_ids.add(200)

            # 其余类别：排序后分配 1~255 中未使用的最小可用ID
            remaining_classes = sorted([c for c in all_classes if c not in name_to_id])
            next_id = 1
            for cls in remaining_classes:
                while next_id in used_ids:
                    next_id += 15
                if next_id >= 256:
                    self.progress_window.append_detail("警告: 类别过多，无法分配唯一灰度值（上限256）")
                    break
                name_to_id[cls] = next_id
                used_ids.add(next_id)
                next_id += 1

            self.progress_window.append_detail(f"类别映射: {name_to_id}")

            # 坐标系转换
            try:
                if gdf.crs != src.crs:
                    self.progress_window.append_detail("转换坐标系...")
                    gdf = gdf.to_crs(src.crs)
            except Exception as e:
                self.progress_window.append_detail(f"坐标系错误: {str(e)}")

            # 创建输出目录
            images_dir = os.path.join(output_dir, 'images')
            masks_dir = os.path.join(output_dir, 'masks')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            # 智能采样点集合
            positions = []

            # 基础网格采样（50%重叠）
            step = tile_size - overlap
            for y in range(0, height - overlap, step):
                for x in range(0, width - overlap, step):
                    positions.append((x, y))

            if np.any(target_map):
                labeled = sk_label(target_map)
                properties = regionprops(labeled)

                for prop in properties:
                    if prop.area >= 25:
                        cy, cx = prop.centroid
                        cx, cy = int(cx), int(cy)
                        x = max(0, min(cx - tile_size // 2, width - tile_size))
                        y = max(0, min(cy - tile_size // 2, height - tile_size))
                        if (x, y) not in positions:
                            positions.append((x, y))
                            self.progress_window.append_detail(f"添加补充采样点 @ ({x},{y})")

            total_slices = len(positions)
            base_grid_count = ((height - overlap + step - 1) // step) * ((width - overlap + step - 1) // step)
            supplement_count = total_slices - base_grid_count
            self.progress_window.append_detail(
                f"预计生成 {total_slices} 个切片（含 {max(0, supplement_count)} 个补充点）"
            )

            # 处理所有采样点
            slice_count = start_number
            for pos in positions:
                x, y = pos
                if stop_event.is_set():
                    self.progress_window.append_detail("用户终止处理")
                    return

                if num_bands >= 3:
                    cropped_tiff = tiff_data[y:y + tile_size, x:x + tile_size]
                else:
                    cropped_tiff = tiff_data[y:y + tile_size, x:x + tile_size, 0]

                # 跳过不完整切片
                if cropped_tiff.shape[0] < tile_size or cropped_tiff.shape[1] < tile_size:
                    continue

                # 生成标注掩膜（uint8，0=背景，1~255=类别）
                mask = self.shp_to_mask(gdf, (tile_size, tile_size), name_to_id, (x, y), transform)

                try:
                    # 保存影像切片
                    img_mode = 'RGB' if num_bands >= 3 else 'L'
                    img = Image.fromarray(cropped_tiff, mode=img_mode)
                    img.save(os.path.join(images_dir, f'image_{slice_count}.jpg'))

                    # 保存灰度标注掩膜（关键修改：直接保存 mask 为灰度图）
                    Image.fromarray(mask, mode='L').save(os.path.join(masks_dir, f'image_{slice_count}_lab.png'))

                    # 更新进度
                    current = slice_count - start_number + 1
                    if current % 10 == 0 or current == total_slices:
                        self.progress_window.update_progress(current, total_slices)
                        self.update_output_preview(f"已保存切片 {slice_count}")

                    slice_count += 1

                except Exception as e:
                    self.progress_window.append_detail(f"切片保存失败: {str(e)}")
                    continue

            self.progress_window.append_detail(f"处理完成，共生成 {slice_count - start_number} 个切片")

    def update_output_preview(self, message):
        self.folders_text.insert(tk.END, message + "\n")
        self.folders_text.see(tk.END)