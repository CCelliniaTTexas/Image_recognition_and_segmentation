import json
import os
import torch
import multiprocessing
import psutil
import tkinter as tk
import ttkbootstrap as ttk
import logging.handlers
import shutil
import logging
from io import StringIO
from tkinter import filedialog, messagebox, StringVar, ttk, BooleanVar
from pathlib import Path
from datetime import datetime
from config import config, RecentFiles
from HelpDialog import HelpDialog
from TrainingHistory import TrainingHistoryStore
from TrainingHistoryViewer import TrainingHistoryViewer


# 添加 CacheManager 类定义
class CacheManager:
    """缓存管理器"""

    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = Path.home() / '.adl_tool' / 'cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_size(self):
        """获取缓存大小"""
        total_size = 0
        for path in self.cache_dir.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024 * 1024)  # 转换为 MB

    def clear_cache(self):
        """清理缓存"""
        try:
            for path in self.cache_dir.rglob('*'):
                if path.is_file():
                    path.unlink()
            logging.info("缓存已清理")
        except Exception as e:
            logging.error(f"清理缓存失败: {str(e)}")

    def cache_file(self, file_path, category):
        """缓存文件"""
        try:
            dest_dir = self.cache_dir / category
            dest_dir.mkdir(parents=True, exist_ok=True)

            file_path = Path(file_path)
            dest_path = dest_dir / file_path.name

            # 如果文件已存在，先删除
            if dest_path.exists():
                dest_path.unlink()

            # 复制文件到缓存目录
            shutil.copy2(file_path, dest_path)
            return dest_path

        except Exception as e:
            logging.error(f"缓存文件失败: {str(e)}")
            return None

    def get_cached_file(self, filename, category):
        """获取缓存的文件"""
        cache_path = self.cache_dir / category / filename
        return cache_path if cache_path.exists() else None

    def clean_old_cache(self, days=7):
        """清理指定天数前的缓存"""
        try:
            now = datetime.now()
            for path in self.cache_dir.rglob('*'):
                if path.is_file():
                    mtime = datetime.fromtimestamp(path.stat().st_mtime)
                    if (now - mtime).days > days:
                        path.unlink()
            logging.info(f"已清理 {days} 天前的缓存")
        except Exception as e:
            logging.error(f"清理旧缓存失败: {str(e)}")


# 添加 MemoryMonitor 类定义
class MemoryMonitor:
    """内存监控器"""

    @staticmethod
    def get_memory_usage():
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'gpu_memory': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }

    @staticmethod
    def log_memory_usage():
        mem = MemoryMonitor.get_memory_usage()
        logging.info(f"内存使用 - RSS: {mem['rss']:.1f}MB, VMS: {mem['vms']:.1f}MB, "
                     f"GPU: {mem['gpu_memory']:.1f}MB")


class SettingsTab(ttk.Frame):
    def __init__(self, parent, log_auto_scroll_var=None):
        super().__init__(parent)
        self.log_auto_scroll_var = log_auto_scroll_var
        self.setup_ui()
        self.update_gpu_status()
        #self.device_status_label = None

    def setup_ui(self):
        # 创建设置类别
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 常规设置
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="常规")
        self.setup_general_settings(general_frame)

        # 外观设置
        appearance_frame = ttk.Frame(notebook)
        notebook.add(appearance_frame, text="外观")
        self.setup_appearance_settings(appearance_frame)

        # 性能设置
        performance_frame = ttk.Frame(notebook)
        notebook.add(performance_frame, text="性能")
        self.setup_performance_settings(performance_frame)

        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="数据")
        self.setup_data_management(data_frame)

        # 添加关于和帮助按钮
        about_frame = tk.LabelFrame(self, text="关于")
        about_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(about_frame, text="使用说明",
                   command=lambda: HelpDialog.show(self.winfo_toplevel())).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(about_frame, text="版本检查",
                   command=self.check_updates).pack(fill=tk.X, padx=5, pady=2)

        version_label = ttk.Label(about_frame, text="当前版本：V2.0.1")
        version_label.pack(padx=5, pady=2)

        # CPU限制
        cpu_frame = ttk.Frame(performance_frame)
        cpu_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(cpu_frame, text="CPU使用率限制:").pack(side=tk.LEFT, padx=5)
        self.cpu_limit = tk.IntVar(value=int(config.get('cpu_limit', 50)))
        self.cpu_display = tk.StringVar(value=str(self.cpu_limit.get()))

        cpu_scale = ttk.Scale(cpu_frame, from_=10, to=90,
                              variable=self.cpu_limit,
                              orient=tk.HORIZONTAL,
                              command=self.update_cpu_display)
        cpu_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(cpu_frame, textvariable=self.cpu_display).pack(side=tk.LEFT, padx=5)
        ttk.Label(cpu_frame, text="%").pack(side=tk.LEFT)

        # 工作线程数显示
        thread_frame = ttk.Frame(performance_frame)
        thread_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(thread_frame, text="当前工作线程数:").pack(side=tk.LEFT, padx=5)
        self.thread_count = tk.StringVar()
        ttk.Label(thread_frame, textvariable=self.thread_count).pack(side=tk.LEFT, padx=5)

        # 定期更新线程数显示
        self.update_thread_count()

    def setup_general_settings(self, parent):
        """常规设置"""
        # 自动保存设置
        autosave_frame = tk.LabelFrame(parent, text="自动保存")
        autosave_frame.pack(fill=tk.X, padx=5, pady=5)

        self.autosave_enabled = BooleanVar(value=config.get('autosave_enabled', True))
        ttk.Checkbutton(autosave_frame, text="启用自动保存",
                        variable=self.autosave_enabled,
                        command=lambda: config.set('autosave_enabled',
                                                   self.autosave_enabled.get())).pack(padx=5, pady=2)

        self.auto_save_training_curves = BooleanVar(value=config.get('auto_save_training_curves', True))
        ttk.Checkbutton(
            autosave_frame,
            text="训练结束后自动保存分割训练曲线",
            variable=self.auto_save_training_curves,
            command=lambda: config.set('auto_save_training_curves', self.auto_save_training_curves.get())
        ).pack(anchor=tk.W, padx=5, pady=2)

        # 日志显示设置
        log_frame = tk.LabelFrame(parent, text="日志显示")
        log_frame.pack(fill=tk.X, padx=5, pady=5)

        if self.log_auto_scroll_var is None:
            self.log_auto_scroll_var = BooleanVar(value=config.get('log_auto_scroll', True))

        ttk.Checkbutton(
            log_frame,
            text="自动滚动到最新消息",
            variable=self.log_auto_scroll_var,
            command=lambda: config.set('log_auto_scroll', self.log_auto_scroll_var.get())
        ).pack(anchor=tk.W, padx=5, pady=2)

        # 关闭行为设置
        close_frame = tk.LabelFrame(parent, text="关闭行为")
        close_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(close_frame, text="点击关闭按钮时：").pack(anchor=tk.W, padx=5, pady=(2, 0))
        self.close_action_var = StringVar(value=config.get('close_action_mode', 'ask'))
        close_action_map = {
            "每次询问": "ask",
            "最小化到托盘（不退出）": "minimize",
            "直接退出程序": "exit",
        }
        close_action_reverse_map = {v: k for k, v in close_action_map.items()}
        self.close_action_display_var = StringVar(
            value=close_action_reverse_map.get(self.close_action_var.get(), "每次询问")
        )

        def update_close_action(_event=None):
            selected = self.close_action_display_var.get()
            mode = close_action_map.get(selected, "ask")
            self.close_action_var.set(mode)
            config.set('close_action_mode', mode)

        close_combo = ttk.Combobox(
            close_frame,
            textvariable=self.close_action_display_var,
            values=list(close_action_map.keys()),
            state="readonly",
            width=24
        )
        close_combo.pack(anchor=tk.W, padx=5, pady=2)
        close_combo.bind("<<ComboboxSelected>>", update_close_action)

    def setup_appearance_settings(self, parent):
        """外观设置"""
        # 主题设置
        theme_frame = tk.LabelFrame(parent, text="主题设置")
        theme_frame.pack(fill=tk.X, padx=5, pady=5)

        self.theme_var = StringVar(value=config.get('theme', 'cosmo'))
        themes = [
            ('白昼模式', 'cosmo'),
            ('黑夜模式', 'darkly')
        ]

        for theme_name, theme_value in themes:
            ttk.Radiobutton(theme_frame,
                            text=theme_name,
                            value=theme_value,
                            variable=self.theme_var,
                            command=self.apply_theme).pack(anchor=tk.W, padx=5, pady=2)

    def setup_performance_settings(self, parent):
        """性能设置"""
        # GPU设置框架
        gpu_frame = tk.LabelFrame(parent, text="计算设备")
        gpu_frame.pack(fill=tk.X, padx=5, pady=5)
        status_frame = ttk.Frame(gpu_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        self.device_status_label = ttk.Label(
            status_frame,
            text=f"当前计算设备: {self.get_current_device_status()}"
        )
        self.device_status_label.pack(side=tk.LEFT)

        ttk.Button(
            status_frame,
            text="刷新状态",
            command=self.update_device_status_display).pack(side=tk.RIGHT)

        self.gpu_info_label = ttk.Label(gpu_frame, text="")
        self.gpu_info_label.pack(anchor=tk.W, padx=5)
        self.use_gpu_var = BooleanVar()
        self.gpu_checkbutton = ttk.Checkbutton(gpu_frame,
                                               text="启用GPU加速",
                                               variable=self.use_gpu_var,
                                               command=self.toggle_device)
        self.gpu_checkbutton.pack(anchor=tk.W, padx=5)

        gpu_mem_frame = ttk.Frame(gpu_frame)
        gpu_mem_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(gpu_mem_frame, text="GPU显存占用限制:").pack(side=tk.LEFT, padx=5)
        self.gpu_mem_fraction = tk.DoubleVar(value=config.get('gpu_memory_fraction', 0.8))
        self.gpu_mem_display = tk.StringVar(value=f"{int(self.gpu_mem_fraction.get() * 100)}%")
        gpu_mem_scale = ttk.Scale(gpu_mem_frame, from_=0.3, to=1.0,
                                  variable=self.gpu_mem_fraction,
                                  orient=tk.HORIZONTAL,
                                  command=self.update_gpu_mem_display)
        gpu_mem_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(gpu_mem_frame, textvariable=self.gpu_mem_display).pack(side=tk.LEFT, padx=5)
        # 训练优化
        optim_frame = tk.LabelFrame(parent, text="训练优化")
        optim_frame.pack(fill=tk.X, padx=5, pady=5)

        self.amp_var = BooleanVar(value=config.get('amp_enabled', True))
        ttk.Checkbutton(optim_frame, text="启用混合精度训练 (AMP，可减少约40%显存)",
                        variable=self.amp_var,
                        command=lambda: config.set('amp_enabled', self.amp_var.get())
                        ).pack(anchor=tk.W, padx=5, pady=2)

        accum_frame = ttk.Frame(optim_frame)
        accum_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(accum_frame, text="梯度累积步数:").pack(side=tk.LEFT)
        self.accum_steps_var = StringVar(value=str(config.get('gradient_accumulation_steps', 1)))
        accum_spin = ttk.Spinbox(accum_frame, from_=1, to=16,
                                 textvariable=self.accum_steps_var,
                                 command=self.update_accum_steps, width=5)
        accum_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(accum_frame, text="(1=不累积, 4=等效batch×4)").pack(side=tk.LEFT)

        parallel_frame = tk.LabelFrame(parent, text="并行处理")
        parallel_frame.pack(fill=tk.X, padx=5, pady=5)
        cpu_frame = ttk.Frame(parallel_frame)
        cpu_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(cpu_frame, text=f"CPU核心数: {multiprocessing.cpu_count()}").pack(side=tk.LEFT)
        worker_frame = ttk.Frame(parallel_frame)
        worker_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(worker_frame, text="工作线程数:").pack(side=tk.LEFT)
        self.num_workers_var = StringVar(value=config.get('num_workers',max(1, multiprocessing.cpu_count() - 1)))

        workers = ttk.Spinbox(worker_frame, from_=1,
                              to=multiprocessing.cpu_count(),
                              textvariable=self.num_workers_var,
                              command=self.update_workers)
        workers.pack(side=tk.LEFT, padx=5)

    def get_current_device_status(self):
        if torch.cuda.is_available() and config.get('use_gpu', False):
            return f"GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"

    def update_device_status_display(self):
        self.device_status_label.config(
            text=f"当前计算设备: {self.get_current_device_status()}"
        )
        logging.info(f"设备状态已更新: {self.get_current_device_status()}")

    def toggle_device(self):
        """切换设备（CPU/GPU）"""
        use_gpu = self.use_gpu_var.get()

        if use_gpu and not torch.cuda.is_available():
            self.use_gpu_var.set(False)
            messagebox.showwarning("警告", "当前设备不支持GPU，已自动切换至CPU")

        config.set('use_gpu', self.use_gpu_var.get())
        self.update_global_device()
        self.update_device_status_display()

        device_type = "GPU" if self.use_gpu_var.get() else "CPU"
        messagebox.showinfo("提示", f"已切换至{device_type}计算")

    def update_global_device(self):
        """更新全局设备状态"""
        global device
        if config.get('use_gpu', False) and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        if device.type == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info(f"全局计算设备已更新: {device}")

    def update_gpu_status(self):
        """更新GPU状态显示"""
        # 检查当前GPU可用性
        gpu_available = torch.cuda.is_available()

        # 获取配置中的GPU设置
        config_use_gpu = config.get('use_gpu', False)

        # 更新UI状态
        if gpu_available:
            self.gpu_info_label.config(text=f"可用GPU：{torch.cuda.get_device_name(0)}")
            self.gpu_checkbutton.config(state="enabled")

            # 如果配置中启用GPU且GPU可用，则选中复选框
            self.use_gpu_var.set(config_use_gpu)
        else:
            self.gpu_info_label.config(text="无可用GPU", foreground="red")
            self.gpu_checkbutton.config(state="disabled")
            self.use_gpu_var.set(False)

        # 更新全局设备状态
        self.update_global_device()

    def setup_data_management(self, parent):
        """数据管理设置"""
        history_frame = tk.LabelFrame(parent, text="历史训练数据")
        history_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(history_frame, text="查看语义分割训练的历史概览、参数、结果和曲线").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Button(history_frame, text="查看训练历史",
                   command=self.open_training_history).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(history_frame, text="打开历史目录",
                   command=self.open_training_history_dir).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(history_frame, text="清空训练历史",
                   command=self.clear_training_history).pack(fill=tk.X, padx=5, pady=2)

        # 缓存管理
        cache_frame = tk.LabelFrame(parent, text="缓存管理")
        cache_frame.pack(fill=tk.X, padx=5, pady=5)

        cache_size = self.get_cache_size()
        ttk.Label(cache_frame, text=f"当前缓存大小: {cache_size}").pack(padx=5, pady=2)
        ttk.Button(cache_frame, text="清理缓存",
                   command=self.clear_cache).pack(padx=5, pady=2)

        # 最近文件管理
        recent_frame = tk.LabelFrame(parent, text="最近文件")
        recent_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(recent_frame, text="清除最近使用的文件记录",
                   command=self.clear_recent_files).pack(padx=5, pady=2)

        # 数据备份
        backup_frame = tk.LabelFrame(parent, text="数据备份")
        backup_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(backup_frame, text="导出设置",
                   command=self.export_settings).pack(padx=5, pady=2)
        ttk.Button(backup_frame, text="导入设置",
                   command=self.import_settings).pack(padx=5, pady=2)

    # 功能实现方法
    def open_training_history(self):
        TrainingHistoryViewer.show(self.winfo_toplevel())

    def open_training_history_dir(self):
        store = TrainingHistoryStore()
        try:
            os.startfile(store.history_dir)
        except Exception as e:
            messagebox.showerror("错误", f"打开历史目录失败：{str(e)}")

    def clear_training_history(self):
        if not messagebox.askyesno("确认", "是否清空所有训练历史记录？该操作不会删除已保存的模型权重和统计输出文件。"):
            return
        try:
            TrainingHistoryStore().clear()
            messagebox.showinfo("成功", "训练历史记录已清空")
        except Exception as e:
            messagebox.showerror("错误", f"清空训练历史失败：{str(e)}")

    def browse_default_model_path(self):
        path = filedialog.askdirectory(title="选择默认模型保存路径")
        if path:
            self.default_model_path.set(path)
            config.set('default_model_path', path)

    def browse_default_dataset_path(self):
        path = filedialog.askdirectory(title="选择默认数据集路径")
        if path:
            self.default_dataset_path.set(path)
            config.set('default_dataset_path', path)

    def apply_theme(self):
        theme = self.theme_var.get()
        # 使用 root 对象的 apply_theme 方法
        self.winfo_toplevel().apply_theme(theme)

    def toggle_device(self):
        """切换设备（CPU/GPU）"""
        use_gpu = self.use_gpu_var.get()

        if use_gpu:
            # 用户选择使用GPU
            if torch.cuda.is_available():
                # GPU可用，更新设备
                self.update_global_device()
                messagebox.showinfo("提示", "已切换至GPU加速")
            else:
                # GPU不可用，显示警告并禁用复选框
                messagebox.showwarning("警告", "当前设备不支持GPU，已自动切换至CPU")
                self.use_gpu_var.set(False)
                self.update_global_device()
        else:
            # 用户选择使用CPU
            self.update_global_device()
            messagebox.showinfo("提示", "已切换至CPU计算")

        # 保存配置
        config.set('use_gpu', self.use_gpu_var.get())

    def update_global_device(self):
        """更新全局设备状态"""
        global device
        if self.use_gpu_var.get() and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        # 记录设备变更
        logging.info(f"当前计算设备: {device}")

    def update_workers(self):
        try:
            workers = int(self.num_workers_var.get())
            config.set('num_workers', workers)
        except ValueError:
            pass

    def get_cache_size(self):
        """获取缓存大小"""
        cache_dir = Path.home() / '.adl_tool' / 'cache'
        total_size = 0
        if cache_dir.exists():
            for path in cache_dir.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
        return f"{total_size / (1024 * 1024):.2f} MB"

    def clear_cache(self):
        """清理缓存"""
        if messagebox.askyesno("确认", "是否清理所有缓存文件？"):
            cache_dir = Path.home() / '.adl_tool' / 'cache'
            if cache_dir.exists():
                for path in cache_dir.rglob('*'):
                    if path.is_file():
                        path.unlink()
                messagebox.showinfo("成功", "缓存已清理")

    def clear_recent_files(self):
        """清理最近文件记录"""
        if messagebox.askyesno("确认", "是否清除所有最近文件记录？"):
            recent_files = RecentFiles()
            recent_files.clear_recent_files()
            messagebox.showinfo("成功", "最近文件记录已清除")

    def export_settings(self):
        """导出设置"""
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                initialfile="settings_backup.json"
            )
            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config.config, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("成功", f"设置已导出到：\n{path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出设置失败：{str(e)}")

    def import_settings(self):
        """导入设置"""
        try:
            path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json")]
            )
            if path:
                with open(path, 'r', encoding='utf-8') as f:
                    new_config = json.load(f)
                config.config.update(new_config)
                config.save_config()
                messagebox.showinfo("成功", "设置已导入，部分设置可能需要重启程序才能生效")
        except Exception as e:
            messagebox.showerror("错误", f"导入设置失败：{str(e)}")

    def check_updates(self):
        """检查更新"""
        messagebox.showinfo("版本信息", """当前版本：V2.0.1 (2026-5)

主要功能：
• 分类训练（ResNet18 + AdamW + 早停）
• 分类预测（批量识别 + 结果导出）
• 分割训练（DeepLabV3 / FCN + 交叉注意力）
• 分割预测（自动识别模型架构和类别数）
• 影像裁剪（TIFF + SHP → 训练数据集）
• GPU 显存占用限制
• CPU 使用率限制与工作线程数配置

联系邮箱：1346636108@qq.com""")

    def update_accum_steps(self):
        try:
            steps = int(self.accum_steps_var.get())
            config.set('gradient_accumulation_steps', max(1, steps))
        except ValueError:
            pass

    def update_gpu_mem_display(self, value):
        """GPU显存限制显示"""
        fraction = round(float(value), 2)
        self.gpu_mem_display.set(f"{int(fraction * 100)}%")
        config.set('gpu_memory_fraction', fraction)
        if torch.cuda.is_available() and config.get('use_gpu', False):
            try:
                torch.cuda.set_per_process_memory_fraction(fraction)
            except Exception as e:
                logging.warning(f"设置GPU显存限制失败: {e}")

    def update_cpu_display(self, value):
        """CPU使用率显示"""
        int_value = round(float(value))
        self.cpu_display.set(str(int_value))
        self.update_cpu_limit(int_value)
        # 立即更新线程数显示
        self.update_thread_count()

    def update_cpu_limit(self, value):
        """CPU限制设置"""
        config.set('cpu_limit', int(value))

    def update_thread_count(self):
        """更新工作线程数显示"""
        try:
            cpu_count = multiprocessing.cpu_count()
            cpu_limit = self.cpu_limit.get()
            thread_count = max(1, int(cpu_count * (cpu_limit / 100)))
            self.thread_count.set(str(thread_count))
        except Exception as e:
            logging.error(f"更新线程数显示失败: {str(e)}")
        finally:
            # 每秒更新一次
            self.after(1000, self.update_thread_count)