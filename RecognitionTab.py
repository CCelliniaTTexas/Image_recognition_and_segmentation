import os
import json
import torch
import torch.nn as nn
import csv
import ttkbootstrap as ttk
import threading
import logging.handlers
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, StringVar
from datetime import datetime
from torchvision import models
from config import RecentFiles
from RecognitionPredictor import BatchRecognizer,ParallelProcessor
from RecognitionChart import ResultsViewer
from _init_ import device



class RecognitionTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._stop_event = threading.Event()
        self.recent_files = RecentFiles(max_items=1)
        self.selected_folders = []
        self.processor = ParallelProcessor()
        self.setup_ui()



    def setup_ui(self):
        # 创建左右分栏
        frame_left = ttk.Frame(self)
        frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        frame_right = ttk.Frame(self)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 模型文件选择
        model_frame = tk.LabelFrame(frame_left, text="模型设置")
        model_frame.pack(fill=tk.X, pady=5)

        self.model_path = StringVar()
        ttk.Label(model_frame, text="模型文件：").pack(anchor=tk.W, padx=5)
        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Entry(model_path_frame, textvariable=self.model_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="浏览",
                   command=self.browse_model).pack(side=tk.RIGHT, padx=5)

        # 类别文件选择
        class_frame = tk.LabelFrame(frame_left, text="类别设置")
        class_frame.pack(fill=tk.X, pady=5)

        self.class_names_path = StringVar()
        ttk.Label(class_frame, text="类别文件：").pack(anchor=tk.W, padx=5)
        class_path_frame = ttk.Frame(class_frame)
        class_path_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Entry(class_path_frame, textvariable=self.class_names_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(class_path_frame, text="浏览",
                   command=self.browse_class_file).pack(side=tk.RIGHT, padx=5)

        # 最近使用的类别文件
        recent_class_files = self.recent_files.get_recent_files('class_files')
        if recent_class_files:
            recent_class_frame = ttk.Frame(class_frame)
            recent_class_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(recent_class_frame, text="最近使用：").pack(side=tk.LEFT)
            ttk.Button(recent_class_frame,
                       text=os.path.basename(recent_class_files[0]),
                       command=lambda: self.class_names_path.set(recent_class_files[0])).pack(side=tk.LEFT, padx=5)

        # 右侧：文件夹选择和预览
        folder_frame = tk.LabelFrame(frame_right, text="识别文件夹")
        folder_frame.pack(fill=tk.BOTH, expand=True)

        # 文件夹列表
        self.folders_text = tk.Text(folder_frame, height=5, width=40)
        self.folders_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 按钮区域
        button_frame = ttk.Frame(folder_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="添加文件夹",
                   command=self.select_folders).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除选择",
                   command=self.clear_selected_folders).pack(side=tk.LEFT, padx=5)

        # 开始识别按钮
        start_frame = ttk.Frame(frame_left)
        start_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 5))

        start_button = ttk.Button(start_frame,
                                  text="开始识别 ※",
                                  style='success.TButton',
                                  command=self.start_recognition)
        start_button.pack(fill=tk.X, ipady=10)

        # 添加快捷键提示
        ttk.Label(start_frame,
                  text="快捷键：Ctrl + R",
                  font=("微软雅黑", 9)).pack(pady=(5, 0))

        # 添加快捷键绑定
        self.bind_all('<Control-r>', lambda e: self.start_recognition())

    def browse_model(self):
        """选择模型文件"""
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)
            self.recent_files.add_recent_file(path, 'models')

    def browse_class_file(self):
        """选择类别文件"""
        path = filedialog.askopenfilename(
            title="选择类别文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.class_names_path.set(path)
            self.recent_files.add_recent_file(path, 'class_files')

    def select_folders(self):
        """选择要识别的文件夹"""
        folders = filedialog.askdirectory(title="选择要识别的文件夹")
        if folders:
            self.selected_folders.append(folders)
            self.recent_files.add_recent_file(folders, 'folders')
            self.update_folders_display()

    def clear_selected_folders(self):
        """清除选择的文件夹"""
        self.selected_folders.clear()
        self.update_folders_display()

    def update_folders_display(self):
        """更新文件夹显示"""
        self.folders_text.delete(1.0, tk.END)
        for folder in self.selected_folders:
            self.folders_text.insert(tk.END, f"{folder}\n")

    def load_class_names(self, class_file):
        """加载类别名称"""
        try:
            with open(class_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"加载类别文件失败: {str(e)}")

    def export_results(self, results, export_dir):
        """导出识别结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(export_dir, f"recognition_results_{timestamp}.csv")

        try:
            with open(export_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["文件路径", "预测类别"])
                for path, class_name in results:
                    writer.writerow([path, class_name])
            return export_path
        except Exception as e:
            raise Exception(f"导出结果失败: {str(e)}")

    def evaluation_thread(self):
        """识别处理线程"""
        try:
            # 加载模型和类别名称
            class_names = self.load_class_names(self.class_names_path.get())
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(class_names))
            recognizer = BatchRecognizer(model, class_names, device)
            recognizer._stop_event = self._stop_event
            results = recognizer.recognize_all(image_paths, self.update_progress)
            model.load_state_dict(torch.load(self.model_path.get(), map_location=device))
            model = model.to(device)
            model.eval()

            # 收集所有图片路径
            image_paths = []
            for folder in self.selected_folders:
                for file in os.listdir(folder):
                    if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                        image_paths.append(os.path.join(folder, file))

            if not image_paths:
                raise Exception("没有找到任何图片")

            # 创建批量识别器
            recognizer = BatchRecognizer(model, class_names, device)

            # 开始识别
            results = recognizer.recognize_all(image_paths, self.update_progress)

            if not results:
                raise Exception("处理过程被取消或出现错误")

            # 展平结果列表
            flat_results = [item for sublist in results if sublist for item in sublist]

            return flat_results

        except Exception as e:
            raise Exception(f"识别过程出错: {str(e)}")

    def start_recognition(self):
        if not self.selected_folders or not self.model_path.get() or not self.class_names_path.get():
            messagebox.showerror("错误", "请确保已选择所有必要的文件和文件夹")
            return

        try:
            logging.info("开始识别过程")
            logging.info(f"模型路径: {self.model_path.get()}")
            logging.info(f"类别文件: {self.class_names_path.get()}")
            logging.info(f"选择的文件夹: {', '.join(self.selected_folders)}")

            self._stop_event.clear()

            progress_window = tk.Toplevel(self)
            progress_window.title("处理进度")
            progress_window.geometry("400x200")
            self._progress_window = progress_window

            progress_var = tk.IntVar()
            status_label = ttk.Label(progress_window, text="正在处理...", font=("微软雅黑", 12))
            status_label.pack(pady=10)

            progress_bar = ttk.Progressbar(progress_window, variable=progress_var,
                                           maximum=100, length=300)
            progress_bar.pack(pady=10)

            info_label = ttk.Label(progress_window, text="", font=("微软雅黑", 10))
            info_label.pack(pady=5)

            cancel_button = ttk.Button(progress_window, text="取消",
                                       style='danger.TButton',
                                       command=self.stop_recognition)
            cancel_button.pack(pady=10)

            def update_progress(progress, status_text=None):
                if self._stop_event.is_set():
                    return
                self.after(0, lambda: self._update_progress_ui(
                    progress_var, status_label, info_label, progress, status_text))

            def evaluation_thread():
                try:
                    logging.info("正在加载模型和类别文件...")
                    class_names = self.load_class_names(self.class_names_path.get())

                    model = models.resnet18(weights=None)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Sequential(
                        nn.BatchNorm1d(num_ftrs),
                        nn.Dropout(0.5),
                        nn.Linear(num_ftrs, len(class_names))
                    )

                    state_dict = torch.load(self.model_path.get(), map_location=device)
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()

                    logging.info(f"类别数量: {len(class_names)}")

                    logging.info("正在收集图片...")
                    image_paths = []
                    for folder in self.selected_folders:
                        folder_images = [os.path.join(folder, file)
                                         for file in os.listdir(folder)
                                         if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
                        image_paths.extend(folder_images)
                        logging.info(f"文件夹 {folder} 中找到 {len(folder_images)} 张图片")

                    if not image_paths:
                        raise Exception("没有找到任何图片")

                    logging.info(f"共找到 {len(image_paths)} 张图片")

                    self.recognizer = BatchRecognizer(model, class_names, device)
                    self.recognizer._stop_event = self._stop_event

                    logging.info("开始批量识别...")
                    results = self.recognizer.recognize_all(image_paths, update_progress)

                    if self._stop_event.is_set():
                        logging.info("识别已被用户取消")
                        self.after(0, lambda: self._close_progress_window())
                        return

                    if not results:
                        raise Exception("处理过程出现错误")

                    logging.info(f"识别完成，成功处理 {len(results)} 张图片")
                    success_rate = len(results) / len(image_paths) * 100
                    logging.info(f"识别成功率: {success_rate:.2f}%")

                    self.after(0, lambda: [self._close_progress_window(),
                                           ResultsViewer(self, results)])

                except Exception as e:
                    logging.error(f"识别过程出错: {str(e)}")
                    error_msg = str(e)
                    self.after(0, lambda msg=error_msg: [
                        self._close_progress_window(),
                        messagebox.showerror("错误", msg)])

            self.recognition_thread = threading.Thread(target=evaluation_thread, daemon=True)
            self.recognition_thread.start()

        except Exception as e:
            logging.error(f"启动识别失败: {str(e)}")
            messagebox.showerror("错误", str(e))

    def _update_progress_ui(self, progress_var, status_label, info_label, progress, status_text):
        try:
            progress_var.set(int(progress))
            if status_text:
                status_label.config(text=status_text)
            info_label.config(text=f"已处理: {int(progress)}%")
        except tk.TclError:
            pass

    def _close_progress_window(self):
        if hasattr(self, '_progress_window') and self._progress_window.winfo_exists():
            self._progress_window.destroy()

    def stop_recognition(self):
        """停止识别过程"""
        self._stop_event.set()
        if hasattr(self, 'recognizer') and self.recognizer is not None:
            self.recognizer.stop()
        self._close_progress_window()
        logging.info("识别已停止")

    def cleanup_resources(self):
        """清理GPU和模型资源"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()