import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import os
import csv
import time
import ttkbootstrap as ttk
import tkinter as tk
import matplotlib.pyplot as plt
import logging
import queue
import logging.handlers
import multiprocessing
import numpy as np
from openpyxl import Workbook
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from torchvision import transforms
from torch.utils.data import DataLoader
from tkinter import filedialog, messagebox, StringVar
from SegmentationDataset import SegmentationDataset
from config import config
from SegmentationModelRegistry import (
    get_architectures,
    architecture_requires_backbone,
    get_backbones,
    get_default_backbone,
    get_architecture_param_schema,
    normalize_architecture_params,
    compose_model_name,
    build_model,
)
from SegmentationAttentionRegistry import (
    apply_attention_to_model,
    get_attention_display_name,
    get_attention_options,
    get_default_attention_type,
    normalize_attention_type,
)
from TrainingHistory import TrainingHistoryStore


class DiceLoss(nn.Module):
    """多分类语义分割 Dice Loss，输入 logits，标签为 (N,H,W) 类别索引。"""

    def __init__(self, smooth=1.0, eps=1e-6):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits.float(), dim=1)
        targets = targets.long()

        valid = (targets >= 0) & (targets < num_classes)
        safe_targets = targets.clamp(min=0, max=num_classes - 1)
        target_one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2)
        target_one_hot = target_one_hot.to(dtype=probs.dtype, device=probs.device)
        valid = valid.unsqueeze(1).to(dtype=probs.dtype, device=probs.device)

        probs = probs * valid
        target_one_hot = target_one_hot * valid

        dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice.mean()


class SegmentationTrainingTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        # 修改数据存储结构，分别存储训练集和验证集的数据
        self.train_data = {
            'batch_indices': [],
            'losses': [],
            'accuracies': [],
            'oa': [],
            'f1': [],
            'recall': []
        }
        self.val_data = {
            'batch_indices': [],
            'losses': [],
            'accuracies': [],
            'oa': [],
            'f1': [],
            'recall': []
        }
        self.parent = parent
        self.queue = queue.Queue()
        self.training_history = TrainingHistoryStore()
        self.setup_ui()
        self.stop_training = False
        self.bind_all('<Control-t>', lambda e: self.start_training())
        # 绑定主题变化事件
        self.bind('<<ThemeChanged>>', self.on_theme_changed)

    def setup_ui(self):
        # 创建左右分栏
        frame_left = ttk.Frame(self)
        frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        frame_right = ttk.Frame(self)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：参数区（可滚动）+ 固定底部按钮区
        left_content = ttk.Frame(frame_left)
        left_content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_canvas = tk.Canvas(left_content, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_content, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        scrollbar_visible = True
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollable_left = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=scrollable_left, anchor="nw")

        def _update_scrollbar_visibility():
            nonlocal scrollbar_visible
            bbox = left_canvas.bbox("all")
            if not bbox:
                return
            content_height = bbox[3] - bbox[1]
            canvas_height = left_canvas.winfo_height()
            need_scrollbar = content_height > canvas_height + 2

            if need_scrollbar and not scrollbar_visible:
                left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                scrollbar_visible = True
            elif not need_scrollbar and scrollbar_visible:
                left_scrollbar.pack_forget()
                scrollbar_visible = False

        def _sync_scroll_region(_event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            left_canvas.itemconfigure(left_window, width=left_canvas.winfo_width())
            _update_scrollbar_visibility()

        scrollable_left.bind("<Configure>", _sync_scroll_region)
        left_canvas.bind("<Configure>", _sync_scroll_region)

        def _on_mousewheel(event):
            if scrollbar_visible:
                left_canvas.yview_scroll(int(-event.delta / 120), "units")

        left_canvas.bind("<Enter>", lambda _e: left_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        left_canvas.bind("<Leave>", lambda _e: left_canvas.unbind_all("<MouseWheel>"))

        # 左侧：模型和参数设置
        # 数据集路径选择
        dataset_frame = tk.LabelFrame(scrollable_left, text="数据集设置")
        dataset_frame.pack(fill=tk.X, pady=5)

        self.dataset_path = StringVar()
        ttk.Label(dataset_frame, text="数据集路径：").pack(anchor=tk.W, padx=5)
        dataset_path_frame = ttk.Frame(dataset_frame)
        dataset_path_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Entry(dataset_path_frame, textvariable=self.dataset_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dataset_path_frame, text="浏览",
                   command=lambda: self.browse_dir(self.dataset_path)).pack(side=tk.RIGHT, padx=5)

        # 模型选择部分
        model_frame = tk.LabelFrame(scrollable_left, text="模型选择")
        model_frame.pack(fill=tk.X, pady=5)

        # 模型架构选择
        ttk.Label(model_frame, text="模型架构:").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        self.model_arch_var = StringVar(value="DeepLabV3")
        model_combobox = ttk.Combobox(
            model_frame,
            textvariable=self.model_arch_var,
            values=get_architectures(),
            state="readonly",
            width=20
        )
        model_combobox.grid(row=0, column=1, padx=5, pady=2, sticky='w')

        self.backbone_label = ttk.Label(model_frame, text="主干网络:")
        self.backbone_label.grid(row=1, column=0, padx=5, pady=2, sticky='e')
        self.backbone_var = StringVar(value=get_default_backbone(self.model_arch_var.get()) or "")
        self.backbone_combobox = ttk.Combobox(
            model_frame,
            textvariable=self.backbone_var,
            values=get_backbones(self.model_arch_var.get()),
            state="readonly",
            width=20
        )
        self.backbone_combobox.grid(row=1, column=1, padx=5, pady=2, sticky='w')

        def _on_arch_change(_event=None):
            arch = self.model_arch_var.get()
            need_backbone = architecture_requires_backbone(arch)
            candidates = get_backbones(arch)
            self.backbone_combobox.configure(values=candidates)

            if need_backbone:
                self.backbone_label.grid()
                self.backbone_combobox.grid()
                if self.backbone_var.get() not in candidates:
                    self.backbone_var.set(candidates[0] if candidates else "")
            else:
                self.backbone_var.set("")
                self.backbone_label.grid_remove()
                self.backbone_combobox.grid_remove()
            if hasattr(self, "arch_param_grid"):
                self._refresh_arch_param_controls()

        model_combobox.bind("<<ComboboxSelected>>", _on_arch_change)
        _on_arch_change()

        # 架构参数卡（按模型架构动态显示）
        arch_param_frame = tk.LabelFrame(scrollable_left, text="架构参数")
        arch_param_frame.pack(fill=tk.X, pady=5)
        self.arch_param_grid = ttk.Frame(arch_param_frame)
        self.arch_param_grid.pack(fill=tk.X, padx=5, pady=5)
        self.arch_param_vars = {}

        # 预训练权重选项
        self.pretrained_var = tk.BooleanVar(value=True)
        pretrained_check = ttk.Checkbutton(
            model_frame,
            text="使用预训练权重",
            variable=self.pretrained_var
        )
        pretrained_check.grid(row=2, column=1, padx=5, pady=2, sticky='w')

        # 注意力机制选择
        ttk.Label(model_frame, text="注意力机制:").grid(row=3, column=0, padx=5, pady=2, sticky='e')
        self.attention_type_var = StringVar(value=get_default_attention_type())
        attention_combobox = ttk.Combobox(
            model_frame,
            textvariable=self.attention_type_var,
            values=get_attention_options(),
            state="readonly",
            width=20
        )
        attention_combobox.grid(row=3, column=1, padx=5, pady=2, sticky='w')

        # 右侧：训练监控
        monitor_frame = tk.LabelFrame(frame_right, text="训练监控")
        monitor_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_notebook = ttk.Notebook(monitor_frame)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.plot_specs = self._create_plot_pages()

        # 初始化图表
        self.init_plots()

        # 权重保存路径
        save_frame = tk.LabelFrame(scrollable_left, text="保存设置")
        save_frame.pack(fill=tk.X, pady=5)

        self.weights_save_path = StringVar()
        ttk.Label(save_frame, text="权重保存路径：").pack(anchor=tk.W, padx=5)
        save_path_frame = ttk.Frame(save_frame)
        save_path_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Entry(save_path_frame, textvariable=self.weights_save_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(save_path_frame, text="浏览",
                   command=self.browse_save_path).pack(side=tk.RIGHT, padx=5)

        self.metrics_output_path = StringVar()
        ttk.Label(save_frame, text="统计输出路径：").pack(anchor=tk.W, padx=5)
        metrics_path_frame = ttk.Frame(save_frame)
        metrics_path_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Entry(metrics_path_frame, textvariable=self.metrics_output_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(metrics_path_frame, text="浏览",
                   command=lambda: self.browse_dir(self.metrics_output_path)).pack(side=tk.RIGHT, padx=5)

        # 训练参数设置
        param_frame = tk.LabelFrame(scrollable_left, text="训练参数")
        param_frame.pack(fill=tk.X, pady=5)

        # 使用网格布局来对齐参数
        params_grid = ttk.Frame(param_frame)
        params_grid.pack(padx=5, pady=5)

        # 类别数量
        ttk.Label(params_grid, text="类别数量:").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        self.num_classes = StringVar(value="2")
        ttk.Entry(params_grid, textvariable=self.num_classes, width=10).grid(row=0, column=1, padx=5, pady=2)

        # 批次大小
        ttk.Label(params_grid, text="批次大小:").grid(row=1, column=0, padx=5, pady=2, sticky='e')
        self.batch_size = StringVar(value="4")
        ttk.Entry(params_grid, textvariable=self.batch_size, width=10).grid(row=1, column=1, padx=5, pady=2)

        # 学习率
        ttk.Label(params_grid, text="学习率:").grid(row=2, column=0, padx=5, pady=2, sticky='e')
        self.learning_rate = StringVar(value="0.001")
        ttk.Entry(params_grid, textvariable=self.learning_rate, width=10).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(params_grid, text="训练轮数:").grid(row=3, column=0, padx=5, pady=2, sticky='e')
        self.epochs = StringVar(value="10")
        ttk.Entry(params_grid, textvariable=self.epochs, width=10).grid(row=3, column=1, padx=5, pady=2)

        ttk.Label(params_grid, text="图像尺寸:").grid(row=4, column=0, padx=5, pady=2, sticky='e')
        self.img_size = StringVar(value="256")
        ttk.Combobox(params_grid, textvariable=self.img_size,
                     values=['128', '256', '384', '512'], state="readonly",
                     width=8).grid(row=4, column=1, padx=5, pady=2)

        ttk.Label(params_grid, text="L2权重衰减:").grid(row=5, column=0, padx=5, pady=2, sticky='e')
        self.weight_decay = StringVar(value="0.0005")
        ttk.Entry(params_grid, textvariable=self.weight_decay, width=10).grid(row=5, column=1, padx=5, pady=2)

        ttk.Label(params_grid, text="解码器随机失活:").grid(row=6, column=0, padx=5, pady=2, sticky='e')
        self.decoder_dropout = StringVar(value="0.2")
        ttk.Entry(params_grid, textvariable=self.decoder_dropout, width=10).grid(row=6, column=1, padx=5, pady=2)

        self.strong_aug_var = tk.BooleanVar(value=True)
        strong_aug_check = ttk.Checkbutton(
            params_grid,
            text="启用强数据增强",
            variable=self.strong_aug_var
        )
        strong_aug_check.grid(row=7, column=0, columnspan=2, padx=5, pady=2, sticky='w')

        self._refresh_arch_param_controls()

        # 开始训练按钮
        start_frame = ttk.Frame(frame_left)
        start_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 5))

        start_button = ttk.Button(start_frame,
                                  text="开始训练 ※",
                                  style='success.TButton',
                                  command=self.start_training)
        start_button.pack(fill=tk.X, ipady=10)

        # 添加快捷键提示
        ttk.Label(start_frame,
                  text="快捷键：Ctrl + T",
                  font=("微软雅黑", 9)).pack(pady=(5, 0))

    def _create_plot_pages(self):
        specs = {
            'loss': {'tab': 'Loss', 'title': 'Loss Curve', 'ylabel': 'Loss', 'train_key': 'losses',
                     'val_key': 'losses', 'train_label': 'Training Loss', 'val_label': 'Validation Loss',
                     'ylim': (0, 1), 'filename': 'loss'},
            'miou': {'tab': 'mIoU', 'title': 'mIoU Curve', 'ylabel': 'mIoU', 'train_key': 'accuracies',
                     'val_key': 'accuracies', 'train_label': 'Training mIoU', 'val_label': 'Validaiton mIoU',
                     'ylim': (0, 0.1), 'filename': 'miou'},
            'oa': {'tab': 'OA', 'title': 'OA Curve (Overall Accuracy)', 'ylabel': 'OA (%)', 'train_key': 'oa',
                   'val_key': 'oa', 'train_label': 'Training OA', 'val_label': 'Validation OA',
                   'ylim': (0, 100), 'filename': 'oa'},
            'f1': {'tab': 'F1-score', 'title': 'F1-score Curve', 'ylabel': 'F1-score', 'train_key': 'f1',
                   'val_key': 'f1', 'train_label': 'Training F1-score', 'val_label': 'Validation F1-score',
                   'ylim': (0, 1), 'filename': 'f1_score'},
            'recall': {'tab': 'Recall', 'title': 'Recall Curve', 'ylabel': 'Recall Rate', 'train_key': 'recall',
                       'val_key': 'recall', 'train_label': 'Training Recall Rate', 'val_label': 'Validation Recall Rate',
                       'ylim': (0, 1), 'filename': 'recall'},
        }

        for metric_key, spec in specs.items():
            page = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(page, text=spec['tab'])

            toolbar = ttk.Frame(page)
            toolbar.pack(fill=tk.X, padx=5, pady=(5, 0))
            ttk.Button(
                toolbar,
                text="导出当前曲线",
                command=lambda key=metric_key: self.export_curve(key)
            ).pack(side=tk.RIGHT)

            figure = Figure(figsize=(6, 5), dpi=100)
            canvas = FigureCanvasTkAgg(figure, master=page)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=5)
            spec['figure'] = figure
            spec['canvas'] = canvas
            spec['ax'] = figure.add_subplot(111)

        self.plot_notebook.update_idletasks()

        return specs


    def init_plots(self):
        """初始化图表"""
        current_theme = self.winfo_toplevel().style.theme_use()
        is_dark_theme = current_theme == 'darkly'

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        bg_color = '#2B3E50' if is_dark_theme else 'white'
        text_color = 'white' if is_dark_theme else 'black'
        grid_color = '#486581' if is_dark_theme else '#CCCCCC'

        for spec in self.plot_specs.values():
            figure = spec['figure']
            ax = spec['ax']
            ax.clear()
            figure.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)

            spec['train_line'], = ax.plot([], [], 'r.-', label=spec['train_label'], markersize=3)
            spec['val_line'], = ax.plot([], [], 'b.-', label=spec['val_label'], markersize=3)
            ax.set_title(spec['title'], color=text_color)
            ax.set_xlabel('Epoch', color=text_color)
            ax.set_ylabel(spec['ylabel'], color=text_color)
            ax.grid(True, linestyle='--', alpha=0.7, color=grid_color)
            ax.tick_params(colors=text_color)
            ax.set_xlim(0, 1)
            ax.set_ylim(*spec['ylim'])

            legend = ax.legend(loc='best', fontsize=9)
            frame = legend.get_frame()
            frame.set_facecolor(bg_color)
            frame.set_edgecolor(grid_color)
            for text in legend.get_texts():
                text.set_color(text_color)

            for spine in ax.spines.values():
                spine.set_color(grid_color)

            figure.tight_layout(pad=2.0)
            spec['canvas'].draw()

    def _refresh_plot_data(self):
        all_epochs = self.train_data['batch_indices'] + self.val_data['batch_indices']
        if not all_epochs:
            for spec in self.plot_specs.values():
                spec['canvas'].draw()
            return

        def _ema(values, alpha=0.12):
            if len(values) < 3:
                return values
            smoothed = []
            prev = values[0]
            for value in values:
                prev = alpha * value + (1.0 - alpha) * prev
                smoothed.append(prev)
            return smoothed

        max_epoch = max(all_epochs)
        for spec in self.plot_specs.values():
            ax = spec['ax']
            train_values = self.train_data[spec['train_key']]
            val_values = self.val_data[spec['val_key']]
            if spec['filename'] == 'loss':
                train_values = _ema(train_values)
                val_values = _ema(val_values)
            all_values = train_values + val_values

            spec['train_line'].set_data(self.train_data['batch_indices'], train_values)
            spec['val_line'].set_data(self.val_data['batch_indices'], val_values)
            ax.set_xlim(0, max(max_epoch * 1.05, 1))

            if all_values:
                max_value = max(all_values)
                if spec['filename'] == 'oa':
                    ax.set_ylim(0, min(max(max_value * 1.1, 1), 105))
                elif spec['filename'] == 'loss':
                    ax.set_ylim(0, max(max_value * 1.1, 1e-6))
                else:
                    ax.set_ylim(0, min(max(max_value * 1.1, 0.01), 1.05))

            spec['canvas'].draw()

    def update_plots(self, epoch_progress, train_loss=None, train_miou=None, train_oa=None,
                     train_f1=None, train_recall=None, val_loss=None, val_miou=None,
                     val_oa=None, val_f1=None, val_recall=None):
        """更新图表，epoch_progress 为浮点 epoch 值（如 1.0, 1.5, 2.0）"""

        def _update():
            try:
                current_theme = self.winfo_toplevel().style.theme_use()
                is_dark_theme = current_theme == 'darkly'
                text_color = 'white' if is_dark_theme else 'black'

                if train_loss is not None and train_miou is not None:
                    self.train_data['batch_indices'].append(epoch_progress)
                    self.train_data['losses'].append(train_loss)
                    self.train_data['accuracies'].append(train_miou)
                    self.train_data['oa'].append(train_oa if train_oa is not None else 0)
                    self.train_data['f1'].append(train_f1 if train_f1 is not None else 0)
                    self.train_data['recall'].append(train_recall if train_recall is not None else 0)

                if val_loss is not None and val_miou is not None:
                    self.val_data['batch_indices'].append(epoch_progress)
                    self.val_data['losses'].append(val_loss)
                    self.val_data['accuracies'].append(val_miou)
                    self.val_data['oa'].append(val_oa if val_oa is not None else 0)
                    self.val_data['f1'].append(val_f1 if val_f1 is not None else 0)
                    self.val_data['recall'].append(val_recall if val_recall is not None else 0)

                for spec in self.plot_specs.values():
                    ax = spec['ax']
                    ax.set_title(ax.get_title(), color=text_color)
                    ax.set_xlabel(ax.get_xlabel(), color=text_color)
                    ax.set_ylabel(ax.get_ylabel(), color=text_color)
                    ax.tick_params(colors=text_color)
                self._refresh_plot_data()

            except Exception as e:
                logging.error(f"更新图表失败: {str(e)}")

        self.after(0, _update)

    def on_theme_changed(self, event=None):
        """主题变化时更新图表"""
        self.init_plots()
        self._refresh_plot_data()

    def export_curve(self, metric_key, output_path=None):
        """导出单张训练曲线。"""
        spec = self.plot_specs.get(metric_key)
        if not spec:
            return None

        if not (self.train_data['batch_indices'] or self.val_data['batch_indices']):
            messagebox.showinfo("提示", "当前没有可导出的曲线数据")
            return None

        if output_path is None:
            output_path = filedialog.asksaveasfilename(
                title=f"导出{spec['title']}",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("All files", "*.*")]
            )
            if not output_path:
                return None

        spec['figure'].savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"{spec['title']}已导出: {output_path}")
        return output_path

    def _save_training_plot(self, save_path):
        """训练结束后按设置自动保存所有曲线图到权重同目录。"""
        if not config.get('auto_save_training_curves', True):
            logging.info("已关闭训练结束自动保存曲线")
            return []

        has_curve_data = bool(self.train_data['batch_indices'] or self.val_data['batch_indices'])
        if not has_curve_data:
            return []

        weights_dir = os.path.dirname(save_path) or os.getcwd()
        weight_name = os.path.splitext(os.path.basename(save_path))[0] or "model"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []

        try:
            os.makedirs(weights_dir, exist_ok=True)
            for metric_key, spec in self.plot_specs.items():
                plot_filename = f"{weight_name}_{spec['filename']}_curve_{timestamp}.png"
                plot_path = os.path.join(weights_dir, plot_filename)
                exported_path = self.export_curve(metric_key, plot_path)
                if exported_path:
                    saved_paths.append(exported_path)
        except Exception as e:
            logging.error(f"保存训练曲线失败: {str(e)}")
        return saved_paths

    def _save_and_record_training_plot(self, save_path, history_run_id=None):
        curve_paths = self._save_training_plot(save_path)
        if history_run_id and curve_paths:
            self.training_history.update_artifacts(history_run_id, {"curve_paths": curve_paths})

    def _reset_plots_for_new_training(self):
        """开始新训练前清空图表数据。"""

        self.train_data = {
            'batch_indices': [],
            'losses': [],
            'accuracies': [],
            'oa': [],
            'f1': [],
            'recall': []
        }
        self.val_data = {
            'batch_indices': [],
            'losses': [],
            'accuracies': [],
            'oa': [],
            'f1': [],
            'recall': []
        }

        # 清空残留进度，避免上一轮消息影响新训练显示
        self.queue = queue.Queue()
        self.init_plots()

    def browse_dir(self, string_var):
        path = filedialog.askdirectory(title="选择文件夹")
        if path:
            string_var.set(path)

    def browse_save_path(self):
        path = filedialog.asksaveasfilename(
            title="选择保存路径",
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if path:
            self.weights_save_path.set(path)

    @staticmethod
    def _compose_model_name(architecture, backbone):
        return compose_model_name(architecture, backbone)

    def _refresh_arch_param_controls(self):
        for widget in self.arch_param_grid.winfo_children():
            widget.destroy()

        self.arch_param_vars = {}
        architecture = self.model_arch_var.get()
        schema = get_architecture_param_schema(architecture)

        if not schema:
            ttk.Label(self.arch_param_grid, text="当前架构无额外可调参数").grid(row=0, column=0, sticky='w', padx=5, pady=2)
            return

        row = 0
        for item in schema:
            key = item["key"]
            label = item["label"]
            default = item["default"]
            ptype = item["type"]
            options = item.get("options")

            if ptype == "bool":
                var = tk.BooleanVar(value=bool(default))
                check = ttk.Checkbutton(self.arch_param_grid, text=label, variable=var)
                check.grid(row=row, column=0, columnspan=2, padx=5, pady=2, sticky='w')
            else:
                ttk.Label(self.arch_param_grid, text=f"{label}:").grid(row=row, column=0, padx=5, pady=2, sticky='e')
                var = StringVar(value=str(default))
                if options:
                    widget = ttk.Combobox(
                        self.arch_param_grid,
                        textvariable=var,
                        values=[str(x) for x in options],
                        state="readonly",
                        width=12
                    )
                else:
                    widget = ttk.Entry(self.arch_param_grid, textvariable=var, width=12)
                widget.grid(row=row, column=1, padx=5, pady=2, sticky='w')
            self.arch_param_vars[key] = var
            row += 1

    def _collect_architecture_params(self):
        raw = {}
        for key, var in self.arch_param_vars.items():
            raw[key] = var.get()
        return normalize_architecture_params(self.model_arch_var.get(), raw)

    def _inject_decoder_dropout(self, model, decoder_dropout):
        """在解码器输出层前注入Dropout2d以抑制过拟合。"""
        if decoder_dropout <= 0:
            return

        classifier = getattr(model, 'classifier', None)
        if classifier is None:
            return

        if isinstance(classifier, nn.Sequential):
            for idx in range(len(classifier) - 1, -1, -1):
                if isinstance(classifier[idx], nn.Conv2d):
                    classifier[idx] = nn.Sequential(
                        nn.Dropout2d(p=decoder_dropout),
                        classifier[idx]
                    )
                    return
        elif isinstance(classifier, nn.Conv2d):
            model.classifier = nn.Sequential(
                nn.Dropout2d(p=decoder_dropout),
                classifier
            )
        elif hasattr(classifier, 'classifier') and isinstance(classifier.classifier, nn.Conv2d):
            # 兼容 DeepLabV3+ 自定义头：head.classifier 为最终 1x1 分类卷积
            classifier.classifier = nn.Sequential(
                nn.Dropout2d(p=decoder_dropout),
                classifier.classifier
            )

    def create_model(self, model_name, num_classes, use_pretrained=True, attention_type="cross",
                     decoder_dropout=0.2, architecture_params=None):
        """根据选择的名称创建模型，并按注意力类型进行可选增强。"""
        try:
            model, family, feature_dim = build_model(
                model_name,
                num_classes,
                use_pretrained=use_pretrained,
                aux_loss=False,
                architecture_params=architecture_params or {}
            )
            self._inject_decoder_dropout(model, decoder_dropout)

            model = apply_attention_to_model(model, model_name, family, feature_dim, attention_type)
            return model, model_name

        except Exception as e:
            logging.error(f"创建模型失败: {str(e)}")
            raise

    @staticmethod
    def _resolve_metrics_dir(metrics_output_path, save_path):
        output_dir = metrics_output_path or os.path.dirname(save_path) or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def _update_confusion_matrix(confusion_matrix, outputs, targets, num_classes):
        preds = outputs.argmax(1).detach().view(-1).cpu()
        targets = targets.detach().view(-1).cpu()
        valid = (targets >= 0) & (targets < num_classes)
        bins = torch.bincount(
            num_classes * targets[valid] + preds[valid],
            minlength=num_classes ** 2
        )
        confusion_matrix += bins.reshape(num_classes, num_classes).numpy()

    @staticmethod
    def _metrics_from_confusion(confusion_matrix):
        cm = confusion_matrix.astype(np.float64)
        tp = np.diag(cm)
        pred_sum = cm.sum(axis=0)
        target_sum = cm.sum(axis=1)
        union = target_sum + pred_sum - tp
        class_iou = np.divide(tp, union, out=np.zeros_like(tp, dtype=np.float64), where=union > 0)
        class_precision = np.divide(tp, pred_sum, out=np.zeros_like(tp, dtype=np.float64), where=pred_sum > 0)
        class_recall = np.divide(tp, target_sum, out=np.zeros_like(tp, dtype=np.float64), where=target_sum > 0)
        class_f1 = np.divide(
            2 * class_precision * class_recall,
            class_precision + class_recall,
            out=np.zeros_like(tp, dtype=np.float64),
            where=(class_precision + class_recall) > 0
        )
        miou = float(class_iou.mean()) if class_iou.size else 0.0
        macro_f1 = float(class_f1.mean()) if class_f1.size else 0.0
        macro_recall = float(class_recall.mean()) if class_recall.size else 0.0
        total = cm.sum()
        oa = float(tp.sum() / total * 100.0) if total > 0 else 0.0
        return class_iou, miou, oa, class_precision, class_recall, class_f1, macro_f1, macro_recall

    @staticmethod
    def _save_class_iou_table(output_dir, run_name, records):
        table_path = os.path.join(output_dir, f"{run_name}_class_iou.csv")
        with open(table_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "phase", "class_id", "iou", "precision", "recall", "f1_score"])
            for record in records:
                for class_id, iou in enumerate(record["class_iou"]):
                    writer.writerow([
                        record["epoch"],
                        record["phase"],
                        class_id,
                        f"{iou:.6f}",
                        f"{record['class_precision'][class_id]:.6f}",
                        f"{record['class_recall'][class_id]:.6f}",
                        f"{record['class_f1'][class_id]:.6f}",
                    ])
        logging.info(f"类别IoU表格已保存: {table_path}")
        return table_path

    @staticmethod
    def _save_metrics_excel(output_dir, run_name, batch_records, class_records):
        excel_path = os.path.join(output_dir, f"{run_name}_training_metrics.xlsx")
        wb = Workbook()

        ws1 = wb.active
        ws1.title = "Sheet1"
        ws1.append(["epoch", "batch", "phase", "loss", "miou", "oa", "f1_score", "recall", "elapsed_seconds", "elapsed_time"])
        for record in batch_records:
            ws1.append([
                record["epoch"],
                record["batch"],
                record["phase"],
                record["loss"],
                record["miou"],
                record["oa"],
                record["f1_score"],
                record["recall"],
                record["elapsed_seconds"],
                record["elapsed_time"],
            ])

        ws2 = wb.create_sheet("Sheet2")
        ws2.append(["epoch", "phase", "class_id", "precision", "recall", "f1_score", "miou"])
        for record in class_records:
            for class_id, precision in enumerate(record["class_precision"]):
                ws2.append([
                    record["epoch"],
                    record["phase"],
                    class_id,
                    float(precision),
                    float(record["class_recall"][class_id]),
                    float(record["class_f1"][class_id]),
                    float(record["class_iou"][class_id]),
                ])

        for worksheet in (ws1, ws2):
            for column_cells in worksheet.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 24)

        wb.save(excel_path)
        logging.info(f"训练指标Excel已保存: {excel_path}")
        return excel_path

    @staticmethod
    def _save_confusion_matrix_heatmap(output_dir, run_name, confusion_matrix):
        cm = confusion_matrix.astype(np.float64)
        row_sum = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
        heatmap_path = os.path.join(output_dir, f"{run_name}_normalized_confusion_matrix.png")

        fig_size = max(6, min(16, normalized.shape[0] * 0.7))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=150)
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.arange(normalized.shape[0])
        labels = [f"Class{i}" for i in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title("Normalized Confusion Matrix")

        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                value = normalized[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        color="white" if value > 0.5 else "black", fontsize=8)

        fig.tight_layout()
        fig.savefig(heatmap_path, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"归一化混淆矩阵热力图已保存: {heatmap_path}")
        return heatmap_path

    @staticmethod
    def _format_duration(seconds):
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _measure_model_complexity(model, device, img_size):
        params = sum(p.numel() for p in model.parameters())
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        was_training = model.training
        model.eval()
        flops_text = "N/A（未安装 thop 或 fvcore，或当前模型不支持自动统计）"

        try:
            from thop import profile
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
            flops_text = f"{flops / 1e9:.3f} GFLOPs"
        except Exception:
            try:
                from fvcore.nn import FlopCountAnalysis
                flops = FlopCountAnalysis(model, dummy).total()
                flops_text = f"{flops / 1e9:.3f} GFLOPs"
            except Exception as e:
                logging.warning(f"FLOPs统计失败: {e}")

        if was_training:
            model.train()
        return params, flops_text

    @staticmethod
    def _benchmark_inference_time(model, device, img_size):
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        was_training = model.training
        model.eval()
        warmup, repeat = 5, 20

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeat):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        if was_training:
            model.train()
        return elapsed / repeat * 1000.0

    @staticmethod
    def _save_training_summary(output_dir, run_name, summary):
        summary_path = os.path.join(output_dir, f"{run_name}_training_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("语义分割训练统计\n")
            f.write("=" * 30 + "\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        logging.info(f"训练统计文件已保存: {summary_path}")
        return summary_path

    def train_model(self, dataset_path, save_path, num_classes, batch_size, learning_rate, epochs, update_progress,
                    model_name, use_pretrained, attention_type="cross", weight_decay=5e-4, decoder_dropout=0.2,
                    strong_augment=True, img_size=256, architecture_params=None, metrics_output_path=None,
                    history_run_id=None):
        """训练分割模型"""
        history_finished = False
        try:
            training_start_time = time.perf_counter()
            device = torch.device('cuda' if config.get('use_gpu', True) and torch.cuda.is_available() else 'cpu')
            logging.info(f"Using device: {device}")
            metrics_output_dir = self._resolve_metrics_dir(metrics_output_path, save_path)
            run_name = os.path.splitext(os.path.basename(save_path))[0] or datetime.now().strftime("segmentation_%Y%m%d_%H%M%S")
            if history_run_id:
                self.training_history.update_metadata(history_run_id, {
                    "runtime": {
                        "device": str(device),
                        "metrics_output_dir": metrics_output_dir,
                        "run_name": run_name,
                    }
                })

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            logging.info(f"训练图像尺寸: {img_size}x{img_size}")
            train_dataset = SegmentationDataset(
                dataset_path, 'train', transform, img_size=img_size, enable_augmentation=strong_augment
            )
            valid_dataset = SegmentationDataset(
                dataset_path, 'valid', transform, img_size=img_size, enable_augmentation=False
            )

            actual_classes = train_dataset.num_classes
            if actual_classes != num_classes:
                logging.warning(f"用户设置类别数 {num_classes}，数据集检测到 {actual_classes} 个类别，已自动修正")
                num_classes = actual_classes
            if history_run_id:
                self.training_history.update_metadata(history_run_id, {"parameters": {"actual_num_classes": num_classes}})

            model, model_name = self.create_model(
                model_name, num_classes, use_pretrained, attention_type, decoder_dropout, architecture_params
            )

            if device.type == 'cuda' and hasattr(model, 'base_model'):
                from torch.utils.checkpoint import checkpoint

                backbone = model.base_model.backbone
                for name in ('layer2', 'layer3', 'layer4'):
                    layer = getattr(backbone, name, None)
                    if layer is not None:
                        orig_forward = layer.forward

                        def make_ckpt_forward(fn):
                            def ckpt_forward(x):
                                return checkpoint(fn, x, use_reentrant=False)
                            return ckpt_forward

                        layer.forward = make_ckpt_forward(orig_forward)
                logging.info("已启用梯度检查点 (layer2, layer3, layer4)")

            model = model.to(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            logging.info(f"使用模型: {model_name}, 类别数: {num_classes}")
            logging.info(
                f"训练正则化参数 - 解码器随机失活: {decoder_dropout:.3f}, L2权重衰减: {weight_decay:.6f}, "
                f"强增强: {strong_augment}"
            )

            use_amp = config.get('amp_enabled', True) and device.type == 'cuda'
            accum_steps = max(1, config.get('gradient_accumulation_steps', 1))
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
            logging.info(f"混合精度(AMP): {'开启' if use_amp else '关闭'}, 梯度累积步数: {accum_steps}")
            grad_clip_norm = float(config.get('grad_clip_norm', 1.0))
            loss_smoothing_alpha = float(config.get('loss_smoothing_alpha', 0.12))
            loss_smoothing_alpha = min(max(loss_smoothing_alpha, 0.0), 1.0)

            num_workers = config.get('num_workers', max(1, multiprocessing.cpu_count() - 1))
            if history_run_id:
                self.training_history.update_metadata(history_run_id, {
                    "runtime": {
                        "amp_enabled": use_amp,
                        "gradient_accumulation_steps": accum_steps,
                        "num_workers": num_workers,
                        "loss_function": "DiceLoss",
                        "grad_clip_norm": grad_clip_norm,
                        "loss_smoothing_alpha": loss_smoothing_alpha,
                    }
                })
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
            if len(train_loader) == 0 or len(valid_loader) == 0:
                raise ValueError("训练集或验证集为空，请检查数据集路径和掩码文件")

            criterion = DiceLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            logging.info(
                f"损失函数: Dice Loss, 梯度裁剪阈值: {grad_clip_norm:.2f}, "
                f"曲线EMA平滑系数: {loss_smoothing_alpha:.2f}"
            )

            best_miou = -1.0
            best_metrics = {}
            final_metrics = {}
            class_iou_records = []
            batch_metric_records = []
            final_val_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

            for epoch in range(1, epochs + 1):
                if self.stop_training:
                    logging.info("训练被用户停止")
                    break

                total_loss = 0
                smoothed_train_loss = None
                train_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
                model.train()
                optimizer.zero_grad()

                for batch_idx, (images, masks) in enumerate(train_loader, 1):
                    if self.stop_training:
                        logging.info("训练被用户停止")
                        return

                    images = images.to(device)
                    masks = masks.to(device)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model(images)['out']
                        if outputs.shape[-2:] != masks.shape[-2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                        raw_loss = criterion(outputs, masks)
                        loss = raw_loss / accum_steps

                    scaler.scale(loss).backward()

                    if batch_idx % accum_steps == 0 or batch_idx == len(train_loader):
                        if grad_clip_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    self._update_confusion_matrix(train_confusion, outputs.float(), masks, num_classes)
                    _, miou, oa, _, _, _, f1_score, recall = self._metrics_from_confusion(train_confusion)

                    batch_loss = raw_loss.item()
                    total_loss += batch_loss

                    avg_loss = total_loss / batch_idx
                    if smoothed_train_loss is None:
                        smoothed_train_loss = batch_loss
                    else:
                        smoothed_train_loss = (
                            loss_smoothing_alpha * batch_loss
                            + (1.0 - loss_smoothing_alpha) * smoothed_train_loss
                        )
                    display_loss = smoothed_train_loss
                    avg_miou = miou
                    avg_oa = oa
                    avg_f1 = f1_score
                    avg_recall = recall
                    elapsed_seconds = time.perf_counter() - training_start_time
                    batch_metric_records.append({
                        "epoch": epoch,
                        "batch": batch_idx,
                        "total_batches": len(train_loader),
                        "phase": "train",
                        "loss": float(display_loss),
                        "miou": float(avg_miou),
                        "oa": float(avg_oa),
                        "f1_score": float(avg_f1),
                        "recall": float(avg_recall),
                        "elapsed_seconds": float(elapsed_seconds),
                        "elapsed_time": self._format_duration(elapsed_seconds),
                    })
                    if history_run_id:
                        self.training_history.append_metric(history_run_id, batch_metric_records[-1])
                    update_progress(epoch, batch_idx, len(train_loader), display_loss, avg_miou, avg_oa, avg_f1, avg_recall)

                    if batch_idx % 10 == 0 and self.stop_training:
                        logging.info("训练被用户停止")
                        return

                (train_class_iou, avg_miou, avg_oa, train_class_precision,
                 train_class_recall, train_class_f1, avg_f1, avg_recall) = self._metrics_from_confusion(train_confusion)
                class_iou_records.append({
                    "epoch": epoch,
                    "phase": "train",
                    "class_iou": train_class_iou,
                    "class_precision": train_class_precision,
                    "class_recall": train_class_recall,
                    "class_f1": train_class_f1,
                })
                logging.info(
                    "训练集各类别IoU: " +
                    ", ".join([f"类别{idx}={iou:.4f}" for idx, iou in enumerate(train_class_iou)])
                )

                model.eval()
                val_loss = 0
                val_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

                with torch.no_grad():
                    for val_images, val_masks in valid_loader:
                        val_images = val_images.to(device)
                        val_masks = val_masks.to(device)

                        with torch.amp.autocast('cuda', enabled=use_amp):
                            val_outputs = model(val_images)['out']
                            if val_outputs.shape[-2:] != val_masks.shape[-2:]:
                                val_outputs = F.interpolate(val_outputs, size=val_masks.shape[-2:], mode='bilinear',
                                                            align_corners=False)
                            val_loss += criterion(val_outputs, val_masks).item()

                        self._update_confusion_matrix(val_confusion, val_outputs.float(), val_masks, num_classes)

                avg_val_loss = val_loss / len(valid_loader)
                (val_class_iou, avg_val_miou, avg_val_oa, val_class_precision,
                 val_class_recall, val_class_f1, avg_val_f1,
                 avg_val_recall) = self._metrics_from_confusion(val_confusion)
                final_val_confusion = val_confusion.copy()
                class_iou_records.append({
                    "epoch": epoch,
                    "phase": "valid",
                    "class_iou": val_class_iou,
                    "class_precision": val_class_precision,
                    "class_recall": val_class_recall,
                    "class_f1": val_class_f1,
                })
                logging.info(
                    "验证集各类别IoU: " +
                    ", ".join([f"类别{idx}={iou:.4f}" for idx, iou in enumerate(val_class_iou)])
                )

                logging.info(f'Epoch {epoch}/{epochs}:')
                logging.info(
                    f'Training Loss: {avg_loss:.4f}, mIoU: {avg_miou:.4f}, OA: {avg_oa:.2f}%, '
                    f'F1-score: {avg_f1:.4f}, Recall: {avg_recall:.4f}'
                )
                logging.info(
                    f'Validation Loss: {avg_val_loss:.4f}, mIoU: {avg_val_miou:.4f}, OA: {avg_val_oa:.2f}%, '
                    f'F1-score: {avg_val_f1:.4f}, Recall: {avg_val_recall:.4f}'
                )

                update_progress(epoch, len(train_loader), len(train_loader),
                                val_loss=avg_val_loss, val_miou=avg_val_miou, val_oa=avg_val_oa,
                                val_f1=avg_val_f1, val_recall=avg_val_recall)
                elapsed_seconds = time.perf_counter() - training_start_time
                batch_metric_records.append({
                    "epoch": epoch,
                    "batch": "valid",
                    "total_batches": len(train_loader),
                    "phase": "valid",
                    "loss": float(avg_val_loss),
                    "miou": float(avg_val_miou),
                    "oa": float(avg_val_oa),
                    "f1_score": float(avg_val_f1),
                    "recall": float(avg_val_recall),
                    "elapsed_seconds": float(elapsed_seconds),
                    "elapsed_time": self._format_duration(elapsed_seconds),
                })
                if history_run_id:
                    self.training_history.append_metric(history_run_id, batch_metric_records[-1])

                if avg_val_miou > best_miou:
                    best_miou = avg_val_miou
                    best_metrics = {
                        "epoch": epoch,
                        "loss": float(avg_val_loss),
                        "miou": float(avg_val_miou),
                        "oa": float(avg_val_oa),
                        "f1_score": float(avg_val_f1),
                        "recall": float(avg_val_recall),
                        "class_iou": [float(v) for v in val_class_iou],
                        "class_precision": [float(v) for v in val_class_precision],
                        "class_recall": [float(v) for v in val_class_recall],
                        "class_f1": [float(v) for v in val_class_f1],
                    }
                    torch.save(model.state_dict(), save_path)
                    logging.info(f"保存最佳模型, mIoU: {best_miou:.4f}")
                final_metrics = {
                    "epoch": epoch,
                    "loss": float(avg_val_loss),
                    "miou": float(avg_val_miou),
                    "oa": float(avg_val_oa),
                    "f1_score": float(avg_val_f1),
                    "recall": float(avg_val_recall),
                    "class_iou": [float(v) for v in val_class_iou],
                    "class_precision": [float(v) for v in val_class_precision],
                    "class_recall": [float(v) for v in val_class_recall],
                    "class_f1": [float(v) for v in val_class_f1],
                }

            if not self.stop_training:
                training_seconds = time.perf_counter() - training_start_time
                peak_memory_mb = (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    if device.type == 'cuda' else 0.0
                )
                class_iou_path = self._save_class_iou_table(metrics_output_dir, run_name, class_iou_records)
                excel_path = self._save_metrics_excel(metrics_output_dir, run_name, batch_metric_records, class_iou_records)
                confusion_path = self._save_confusion_matrix_heatmap(metrics_output_dir, run_name, final_val_confusion)

                params, flops_text = self._measure_model_complexity(model, device, img_size)
                try:
                    inference_ms = self._benchmark_inference_time(model, device, img_size)
                    inference_text = f"{inference_ms:.3f} ms/张"
                except Exception as e:
                    logging.warning(f"推理耗时统计失败: {e}")
                    inference_text = "N/A"

                summary_data = {
                    "训练耗时": f"{self._format_duration(training_seconds)} ({training_seconds:.2f} 秒)",
                    "参数量": f"{params:,}",
                    "浮点运算量(FLOPs)": flops_text,
                    "单张图像推理时间": inference_text,
                    "峰值显存占用": f"{peak_memory_mb:.2f} MB" if device.type == 'cuda' else "0.00 MB（CPU训练）",
                    "最佳验证mIoU": f"{best_miou:.6f}",
                    "最佳验证各类别IoU": ", ".join(
                        [f"类别{idx}={iou:.6f}" for idx, iou in enumerate(best_metrics.get("class_iou", []))]
                    ),
                    "最终验证各类别IoU": ", ".join(
                        [f"类别{idx}={iou:.6f}" for idx, iou in enumerate(final_metrics.get("class_iou", []))]
                    ),
                    "训练指标Excel": excel_path,
                    "类别IoU表格": class_iou_path,
                    "归一化混淆矩阵热力图": confusion_path,
                }
                summary_path = self._save_training_summary(metrics_output_dir, run_name, summary_data)
                if history_run_id:
                    results = {
                        "training_seconds": float(training_seconds),
                        "training_time": self._format_duration(training_seconds),
                        "parameters_count": int(params),
                        "flops": flops_text,
                        "inference_time": inference_text,
                        "peak_memory_mb": float(peak_memory_mb),
                        "best_metrics": best_metrics,
                        "final_metrics": final_metrics,
                        "best_miou": float(best_miou),
                    }
                    self.training_history.update_artifacts(history_run_id, {
                        "weight_path": save_path,
                        "class_iou_path": class_iou_path,
                        "excel_path": excel_path,
                        "confusion_matrix_path": confusion_path,
                        "summary_path": summary_path,
                    })
                    self.training_history.finish_run(history_run_id, "completed", results=results)
                    history_finished = True
                try:
                    self.after(0, lambda: messagebox.showinfo("完成", "训练完成！"))
                except RuntimeError:
                    pass

        except Exception as e:
            try:
                logging.error(f"训练失败: {str(e)}")
            except RuntimeError:
                print(f"训练失败: {str(e)}")
            if history_run_id:
                self.training_history.finish_run(history_run_id, "failed", error=str(e))
                history_finished = True
            error_msg = str(e)
            try:
                self.after(0, lambda msg=error_msg: messagebox.showerror("训练失败", msg))
            except RuntimeError:
                pass
        finally:
            if history_run_id and not history_finished:
                status = "stopped" if self.stop_training else "finished_without_summary"
                self.training_history.finish_run(history_run_id, status)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                self.after(0, lambda p=save_path, rid=history_run_id: self._save_and_record_training_plot(p, rid))
                self.after(0, self._close_progress_window)
            except RuntimeError:
                pass

    def start_training(self):
        """开始训练"""
        try:
            # 验证输入
            if not all([self.dataset_path.get(), self.weights_save_path.get()]):
                messagebox.showerror("错误", "请填写所有必要参数！")
                return

            # 获取参数
            dataset_path = self.dataset_path.get()
            save_path = self.weights_save_path.get()
            metrics_output_path = self.metrics_output_path.get().strip()
            num_classes = int(self.num_classes.get())
            batch_size = int(self.batch_size.get())
            learning_rate = float(self.learning_rate.get())
            epochs = int(self.epochs.get())
            if batch_size <= 0 or epochs <= 0:
                raise ValueError("批次大小和训练轮数必须大于0")
            arch = self.model_arch_var.get()
            backbone = self.backbone_var.get() if architecture_requires_backbone(arch) else None
            model_name = self._compose_model_name(arch, backbone)
            arch_params = self._collect_architecture_params()
            use_pretrained = self.pretrained_var.get()
            attention_type = normalize_attention_type(self.attention_type_var.get())
            attention_label = get_attention_display_name(attention_type)
            weight_decay = float(self.weight_decay.get())
            decoder_dropout = float(self.decoder_dropout.get())
            strong_augment = self.strong_aug_var.get()
            decoder_dropout = min(max(decoder_dropout, 0.0), 0.5)
            weight_decay = max(weight_decay, 0.0)
            img_size = int(self.img_size.get())
            ablation_config = {
                "architecture": arch,
                "backbone": backbone,
                "model_name": model_name,
                "architecture_params": arch_params,
                "use_pretrained": use_pretrained,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "img_size": img_size,
                "weight_decay": weight_decay,
                "decoder_dropout": decoder_dropout,
                "strong_augment": strong_augment,
                "loss_function": "DiceLoss",
            }
            history_run_id = self.training_history.create_run({
                "parameters": {
                    "task": "semantic_segmentation",
                    "dataset_path": dataset_path,
                    "save_path": save_path,
                    "metrics_output_path": metrics_output_path,
                    "architecture": arch,
                    "backbone": backbone,
                    "model_name": model_name,
                    "architecture_params": arch_params,
                    "use_pretrained": use_pretrained,
                    "attention_type": attention_type,
                    "attention_label": attention_label,
                    "num_classes": num_classes,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "img_size": img_size,
                    "weight_decay": weight_decay,
                    "decoder_dropout": decoder_dropout,
                    "strong_augment": strong_augment,
                    "loss_function": "DiceLoss",
                    "ablation_config": ablation_config,
                },
                "runtime": {
                    "configured_use_gpu": config.get('use_gpu', True),
                    "configured_amp_enabled": config.get('amp_enabled', True),
                    "configured_gradient_accumulation_steps": config.get('gradient_accumulation_steps', 1),
                    "configured_num_workers": config.get('num_workers', max(1, multiprocessing.cpu_count() - 1)),
                }
            })

            # 新训练前仅清空图表
            self._reset_plots_for_new_training()

            # 创建进度窗口
            self.progress_window = tk.Toplevel()
            self.progress_window.title("训练进度")
            self.progress_window.geometry("400x400")
            self.progress_window.transient(self.winfo_toplevel())  # 设置为主窗口的子窗口

            # 进度信息
            info_frame = ttk.Frame(self.progress_window)
            info_frame.pack(fill=tk.X, padx=10, pady=5)

            # 添加模型信息显示
            attention_text = "" if attention_type == "none" else f" + {attention_label}"
            model_info = f"模型: {model_name}{' (预训练)' if use_pretrained else ' (随机初始化)'}{attention_text}"
            self.model_label = ttk.Label(info_frame, text=model_info)
            self.model_label.pack(anchor=tk.W)

            self.epoch_label = ttk.Label(info_frame, text="当前轮次: 0/{}".format(epochs))
            self.epoch_label.pack(anchor=tk.W)

            self.batch_label = ttk.Label(info_frame, text="批次进度: 0/0")
            self.batch_label.pack(anchor=tk.W)

            self.loss_label = ttk.Label(info_frame, text="当前损失: -")
            self.loss_label.pack(anchor=tk.W)

            self.miou_label = ttk.Label(info_frame, text="当前mIoU: -")
            self.miou_label.pack(anchor=tk.W)

            self.oa_label = ttk.Label(info_frame, text="当前OA: -")
            self.oa_label.pack(anchor=tk.W)

            self.f1_label = ttk.Label(info_frame, text="当前F1-score: -")
            self.f1_label.pack(anchor=tk.W)

            self.recall_label = ttk.Label(info_frame, text="当前召回率: -")
            self.recall_label.pack(anchor=tk.W)

            # 进度条
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(self.progress_window,
                                                variable=self.progress_var,
                                                maximum=100,
                                                length=300)
            self.progress_bar.pack(pady=10)

            # 停止按钮
            self.stop_btn = ttk.Button(self.progress_window,
                                       text="停止训练",
                                       style='danger.TButton',
                                       command=self.stop_training_safely)
            self.stop_btn.pack(pady=10)

            # 记录训练开始
            logging.info(f"开始训练 - 数据集: {dataset_path}")
            logging.info(
                f"训练参数 - 类别数: {num_classes}, 批次大小: {batch_size}, 学习率: {learning_rate}, "
                f"L2权重衰减: {weight_decay}, 解码器随机失活: {decoder_dropout}, 强增强: {strong_augment}"
            )
            logging.info(f"使用模型: {model_name}, 预训练: {use_pretrained}, 注意力机制: {attention_label}")
            logging.info(f"架构参数: {arch_params}")

            def update_progress(epoch, batch, total_batches, train_loss=None, train_miou=None, train_oa=None,
                                train_f1=None, train_recall=None, val_loss=None, val_miou=None,
                                val_oa=None, val_f1=None, val_recall=None):
                metric_type = 'val' if val_loss is not None else 'train'
                payload = {
                    'epoch': epoch,
                    'batch': batch,
                    'total_batches': total_batches,
                    'type': metric_type,
                    'train_loss': train_loss,
                    'train_miou': train_miou,
                    'train_oa': train_oa,
                    'train_f1': train_f1,
                    'train_recall': train_recall,
                    'val_loss': val_loss,
                    'val_miou': val_miou,
                    'val_oa': val_oa,
                    'val_f1': val_f1,
                    'val_recall': val_recall,
                }
                self.queue.put(payload)

            self.progress_bar = ttk.Progressbar(
                info_frame, variable=self.progress_var, maximum=100)
            self.progress_bar.pack(fill=tk.X, pady=5)

            # 创建训练线程
            self.stop_training = False
            self.training_thread = threading.Thread(
                target=self.train_model,
                args=(dataset_path, save_path, num_classes, batch_size, learning_rate, epochs,
                      update_progress, model_name, use_pretrained, attention_type,
                      weight_decay, decoder_dropout, strong_augment, img_size, arch_params, metrics_output_path,
                      history_run_id)
            )
            self.training_thread.start()
            # 训练线程启动后再轮询队列，避免首次检查时线程尚未创建导致轮询提前结束
            self.after(100, self.check_progress)

            # 绑定窗口关闭事件
            self.progress_window.protocol("WM_DELETE_WINDOW", self.stop_training_safely)

        except Exception as e:
            logging.error(f"启动训练失败: {str(e)}")
            messagebox.showerror("错误", str(e))

    def check_progress(self):
        """检查进度队列"""
        try:
            while True:
                # 非阻塞方式获取消息
                msg = self.queue.get_nowait()
                if isinstance(msg, dict):
                    epoch = msg['epoch']
                    batch = msg['batch']
                    total_batches = max(1, msg['total_batches'])
                    metric_type = msg.get('type', 'train')
                    train_loss = msg.get('train_loss')
                    train_miou = msg.get('train_miou')
                    train_oa = msg.get('train_oa')
                    train_f1 = msg.get('train_f1')
                    train_recall = msg.get('train_recall')
                    val_loss = msg.get('val_loss')
                    val_miou = msg.get('val_miou')
                    val_oa = msg.get('val_oa')
                    val_f1 = msg.get('val_f1')
                    val_recall = msg.get('val_recall')
                else:
                    # 兼容旧格式队列消息
                    epoch, batch, total_batches, train_loss, train_miou, train_oa = msg
                    total_batches = max(1, total_batches)
                    metric_type = 'train'
                    train_f1 = None
                    train_recall = None
                    val_loss = None
                    val_miou = None
                    val_oa = None
                    val_f1 = None
                    val_recall = None

                try:
                    if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
                        self.epoch_label.config(text=f"当前轮次: {epoch}/{self.epochs.get()}")
                        self.batch_label.config(text=f"批次进度: {batch}/{total_batches}")

                        if metric_type == 'train' and train_loss is not None:
                            self.loss_label.config(text=f"当前损失: {train_loss:.4f}")
                            self.miou_label.config(text=f"当前mIoU: {train_miou:.4f}")
                            self.oa_label.config(text=f"当前OA: {train_oa:.2f}%")
                            self.f1_label.config(text=f"当前F1-score: {(train_f1 or 0):.4f}")
                            self.recall_label.config(text=f"当前召回率: {(train_recall or 0):.4f}")
                        elif metric_type == 'val' and val_loss is not None:
                            self.loss_label.config(text=f"验证损失: {val_loss:.4f}")
                            self.miou_label.config(text=f"验证mIoU: {val_miou:.4f}")
                            self.oa_label.config(text=f"验证OA: {val_oa:.2f}%")
                            self.f1_label.config(text=f"验证F1-score: {(val_f1 or 0):.4f}")
                            self.recall_label.config(text=f"验证召回率: {(val_recall or 0):.4f}")

                        progress = (epoch - 1 + batch / total_batches) / int(self.epochs.get()) * 100
                        self.progress_var.set(progress)

                    epoch_progress = epoch - 1 + batch / total_batches
                    self.update_plots(
                        epoch_progress,
                        train_loss=train_loss,
                        train_miou=train_miou,
                        train_oa=train_oa,
                        train_f1=train_f1,
                        train_recall=train_recall,
                        val_loss=val_loss,
                        val_miou=val_miou,
                        val_oa=val_oa,
                        val_f1=val_f1,
                        val_recall=val_recall
                    )
                except RuntimeError:
                    # 主线程退出或窗口销毁时静默退出，避免线程级崩溃
                    break

        except queue.Empty:
            pass
        finally:
            # 如果训练还在继续，则继续检查
            should_continue = False
            if hasattr(self, 'training_thread') and self.training_thread.is_alive():
                should_continue = True
            elif not self.queue.empty():
                should_continue = True

            if should_continue:
                try:
                    self.after(100, self.check_progress)
                except RuntimeError:
                    pass

    def _close_progress_window(self):
        if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
            self.progress_window.destroy()

    def stop_training_safely(self):
        """安全停止训练"""
        if messagebox.askyesno("确认", "确定要停止训练吗？"):
            logging.info("正在安全停止训练...")
            self.stop_training = True
            if hasattr(self, 'progress_window'):
                self.progress_window.destroy()

    def validate_inputs(self):
        """验证输入是否有效"""
        if not self.dataset_path.get():
            messagebox.showerror("错误", "请选择数据集路径")
            return False
        if not self.weights_save_path.get():
            messagebox.showerror("错误", "请选择模型保存路径")
            return False
        if not self.num_classes.get():
            messagebox.showerror("错误", "请输入类别数量")
            return False
        if not self.batch_size.get():
            messagebox.showerror("错误", "请输入批次大小")
            return False
        if not self.learning_rate.get():
            messagebox.showerror("错误", "请输入学习率")
            return False
        return True

    def update_dataset_info(self, dataset_path):
        # 实现更新数据集信息逻辑
        pass