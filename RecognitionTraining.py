import os
import torch
import tkinter as tk
import torch.nn as nn
import ttkbootstrap as ttk
import threading
import sys
import time
import logging.handlers
import matplotlib.pyplot as plt
import logging
import multiprocessing
from config import config
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from SegmentationModelRegistry import _allow_torchvision_weights, _build_with_pretrained_fallback



class TrainingTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.stop_event = threading.Event()
        self.processor = None
        self._last_progress_update_time = 0.0
        self._last_plot_refresh_time = 0.0
        self._progress_update_interval = 0.5
        self._plot_refresh_interval = 2.0
        # 修改数据存储结构，分别存储训练集和验证集的数据
        self.train_data = {
            'batch_indices': [],
            'losses': [],
            'accuracies': []
        }
        self.val_data = {
            'epochs': [],
            'losses': [],
            'accuracies': []
        }
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

        # 左侧：模型和参数设置
        # 数据集路径选择
        dataset_frame = tk.LabelFrame(frame_left, text="数据集设置")
        dataset_frame.pack(fill=tk.X, pady=5)

        self.dataset_path = StringVar()
        ttk.Label(dataset_frame, text="数据集路径：").pack(anchor=tk.W, padx=5)
        dataset_path_frame = ttk.Frame(dataset_frame)
        dataset_path_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Entry(dataset_path_frame, textvariable=self.dataset_path).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 修改这部分，删除预览按钮
        ttk.Button(dataset_path_frame, text="浏览",
                   command=self.browse_dataset_path).pack(side=tk.RIGHT, padx=5)

        # 模型保存路径
        save_frame = tk.LabelFrame(frame_left, text="模型设置")
        save_frame.pack(fill=tk.X, pady=5)

        self.save_model_path = StringVar()
        ttk.Label(save_frame, text="输出模型地址：").pack(anchor=tk.W, padx=5)
        save_path_frame = ttk.Frame(save_frame)
        save_path_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Entry(save_path_frame, textvariable=self.save_model_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(save_path_frame, text="浏览",
                   command=self.browse_save_path).pack(side=tk.RIGHT, padx=5)

        # 训练参数设置
        param_frame = tk.LabelFrame(frame_left, text="训练参数")
        param_frame.pack(fill=tk.X, pady=5)

        # 使用网格布局来对齐参数
        params_grid = ttk.Frame(param_frame)
        params_grid.pack(padx=5, pady=5)

        # 类别数量
        ttk.Label(params_grid, text="类别数量:").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        self.class_count = StringVar(value="2")
        ttk.Entry(params_grid, textvariable=self.class_count, width=10).grid(row=0, column=1, padx=5, pady=2)

        # 批次大小
        ttk.Label(params_grid, text="批次大小:").grid(row=1, column=0, padx=5, pady=2, sticky='e')
        self.batch_size = StringVar(value="32")
        ttk.Entry(params_grid, textvariable=self.batch_size, width=10).grid(row=1, column=1, padx=5, pady=2)

        # 学习率
        ttk.Label(params_grid, text="学习率:").grid(row=2, column=0, padx=5, pady=2, sticky='e')
        self.learning_rate = StringVar(value="0.001")
        ttk.Entry(params_grid, textvariable=self.learning_rate, width=10).grid(row=2, column=1, padx=5, pady=2)

        # 训练轮数
        ttk.Label(params_grid, text="训练轮数:").grid(row=3, column=0, padx=5, pady=2, sticky='e')
        self.epochs = StringVar(value="10")
        ttk.Entry(params_grid, textvariable=self.epochs, width=10).grid(row=3, column=1, padx=5, pady=2)

        # 右侧：训练监控
        monitor_frame = tk.LabelFrame(frame_right, text="训练监控")
        monitor_frame.pack(fill=tk.BOTH, expand=True)

        # 创建图表区域
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=monitor_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 添加两个子图
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)

        # 初始化图表
        self.init_plots()

        # 初始化数据
        self.train_losses = []
        self.val_accuracies = []

        # 替换为新的开始训练按钮框架（放在左侧框架底部）
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

    def init_plots(self):
        """初始化图表"""
        # 获取当前主题
        current_theme = self.winfo_toplevel().style.theme_use()
        is_dark_theme = current_theme == 'darkly'

        # 设置图表样式
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 根据主题设置颜色
        bg_color = '#2B3E50' if is_dark_theme else 'white'
        text_color = 'white' if is_dark_theme else 'black'
        grid_color = '#486581' if is_dark_theme else '#CCCCCC'

        self.ax1.clear()
        self.ax2.clear()

        # 设置图表背景和文字颜色
        self.figure.patch.set_facecolor(bg_color)
        self.ax1.set_facecolor(bg_color)
        self.ax2.set_facecolor(bg_color)

        # 训练损失曲线
        self.train_line, = self.ax1.plot([], [], 'r.-', label='训练损失', markersize=4)
        self.val_line, = self.ax1.plot([], [], 'b.-', label='验证损失', markersize=4)
        self.ax1.set_title('损失曲线', color=text_color)
        self.ax1.set_xlabel('批次', color=text_color)
        self.ax1.set_ylabel('损失', color=text_color)
        self.ax1.grid(True, linestyle='--', alpha=0.7, color=grid_color)
        self.ax1.tick_params(colors=text_color)

        # 准确率曲线
        self.train_acc_line, = self.ax2.plot([], [], 'r.-', label='训练准确率', markersize=4)
        self.val_acc_line, = self.ax2.plot([], [], 'b.-', label='验证准确率', markersize=4)
        self.ax2.set_title('准确率曲线', color=text_color)
        self.ax2.set_xlabel('批次', color=text_color)
        self.ax2.set_ylabel('准确率 (%)', color=text_color)
        self.ax2.grid(True, linestyle='--', alpha=0.7, color=grid_color)
        self.ax2.tick_params(colors=text_color)

        # 设置图例颜色
        legend1 = self.ax1.legend(loc='upper right')
        legend2 = self.ax2.legend(loc='lower right')
        for legend in [legend1, legend2]:
            frame = legend.get_frame()
            frame.set_facecolor(bg_color)
            frame.set_edgecolor(grid_color)
            for text in legend.get_texts():
                text.set_color(text_color)

        # 设置初始范围
        self.ax1.set_xlim(0, 100)
        self.ax1.set_ylim(0, 1)
        self.ax2.set_xlim(0, 100)
        self.ax2.set_ylim(0, 100)

        # 设置轴线颜色
        for ax in [self.ax1, self.ax2]:
            ax.spines['bottom'].set_color(grid_color)
            ax.spines['top'].set_color(grid_color)
            ax.spines['left'].set_color(grid_color)
            ax.spines['right'].set_color(grid_color)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def on_theme_changed(self, event=None):
        """主题变化时更新图表"""
        self.init_plots()
        # 如果有数据，重新绘制
        if hasattr(self, 'train_data') and self.train_data['batch_indices']:
            self.update_plots(
                self.train_data['batch_indices'][-1],
                self.train_data['losses'][-1],
                self.train_data['accuracies'][-1]
            )

    def update_plots(self, current_batch, train_loss, train_acc, val_loss=None, val_acc=None):
        """更新图表"""

        def _update():
            try:
                now = time.perf_counter()
                is_validation_update = val_loss is not None and val_acc is not None
                should_redraw = is_validation_update or (now - self._last_plot_refresh_time >= self._plot_refresh_interval)

                # 获取当前主题
                current_theme = self.winfo_toplevel().style.theme_use()
                is_dark_theme = current_theme == 'darkly'
                text_color = 'white' if is_dark_theme else 'black'

                # 更新训练数据
                self.train_data['batch_indices'].append(current_batch)
                self.train_data['losses'].append(train_loss)
                self.train_data['accuracies'].append(train_acc)

                # 更新训练曲线
                self.train_line.set_data(self.train_data['batch_indices'], self.train_data['losses'])
                self.train_acc_line.set_data(self.train_data['batch_indices'], self.train_data['accuracies'])

                # 如果有验证数据，则更新验证曲线
                if val_loss is not None and val_acc is not None:
                    self.val_data['epochs'].append(current_batch)
                    self.val_data['losses'].append(val_loss)
                    self.val_data['accuracies'].append(val_acc)

                    self.val_line.set_data(self.val_data['epochs'], self.val_data['losses'])
                    self.val_acc_line.set_data(self.val_data['epochs'], self.val_data['accuracies'])

                # 动态调整坐标轴范围
                if len(self.train_data['batch_indices']) > 1:
                    max_batch = max(self.train_data['batch_indices'])
                    self.ax1.set_xlim(0, max_batch * 1.1)
                    self.ax2.set_xlim(0, max_batch * 1.1)

                    max_loss = max(max(self.train_data['losses']),
                                   max(self.val_data['losses']) if self.val_data['losses'] else 0)
                    self.ax1.set_ylim(0, max_loss * 1.1)

                # 更新文字颜色
                for ax in [self.ax1, self.ax2]:
                    ax.set_title(ax.get_title(), color=text_color)
                    ax.set_xlabel(ax.get_xlabel(), color=text_color)
                    ax.set_ylabel(ax.get_ylabel(), color=text_color)
                    ax.tick_params(colors=text_color)

                # 重绘图表
                if should_redraw:
                    self.canvas.draw_idle()
                    self._last_plot_refresh_time = now

            except Exception as e:
                logging.error(f"更新图表失败: {str(e)}")

        # 在主线程中执行更新
        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.after(0, _update)

    def browse_dataset_path(self):
        path = filedialog.askdirectory(title="选择数据集目录")
        if path:
            self.dataset_path.set(path)

    def browse_save_path(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth")]
        )
        if path:
            self.save_model_path.set(path)

    def start_training(self):
        """开始训练"""
        try:
            # 验证输入
            if not self.validate_inputs():
                return

            # 获取参数
            dataset_path = self.dataset_path.get()
            save_path = self.save_model_path.get()
            num_classes = int(self.class_count.get())
            batch_size = int(self.batch_size.get())
            learning_rate = float(self.learning_rate.get())
            epochs = int(self.epochs.get())

            # 创建进度窗口
            self.progress_window = tk.Toplevel(self)
            self.progress_window.title("训练进度")
            self.progress_window.geometry("600x500")
            self.progress_window.transient(self)  # 设置为主窗口的子窗口

            # 进度信息
            info_frame = ttk.Frame(self.progress_window)
            info_frame.pack(fill=tk.X, padx=10, pady=5)

            self.epoch_label = ttk.Label(info_frame, text="当前轮次: 0/{}".format(epochs))
            self.epoch_label.pack(anchor=tk.W)

            self.batch_label = ttk.Label(info_frame, text="批次进度: 0/0")
            self.batch_label.pack(anchor=tk.W)

            self.loss_label = ttk.Label(info_frame, text="当前损失: -")
            self.loss_label.pack(anchor=tk.W)

            self.acc_label = ttk.Label(info_frame, text="当前准确率: -")
            self.acc_label.pack(anchor=tk.W)

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
            logging.info(f"训练参数 - 类别数: {num_classes}, 批次大小: {batch_size}, 学习率: {learning_rate}")
            self._last_progress_update_time = 0.0
            self._last_plot_refresh_time = 0.0

            def update_progress(epoch, batch, total_batches, loss, accuracy):
                if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
                    try:
                        self.epoch_label.config(text=f"当前轮次: {epoch}/{epochs}")
                        self.batch_label.config(text=f"批次进度: {batch}/{total_batches}")
                        self.loss_label.config(text=f"当前损失: {loss:.4f}")
                        self.acc_label.config(text=f"当前准确率: {accuracy:.2f}%")
                        progress = (epoch - 1 + batch / total_batches) / epochs * 100
                        self.progress_var.set(progress)
                    except Exception as e:
                        logging.error(f"更新进度显示失败: {str(e)}")

            # 创建训练线程
            self.stop_training = False
            self.training_thread = threading.Thread(
                target=self.train_model,
                args=(dataset_path, save_path, num_classes, batch_size, learning_rate, epochs, update_progress)
            )
            self.training_thread.daemon = True
            self.training_thread.start()

            # 绑定窗口关闭事件
            self.progress_window.protocol("WM_DELETE_WINDOW", self.stop_training_safely)

        except Exception as e:
            logging.error(f"启动训练失败: {str(e)}")
            messagebox.showerror("错误", str(e))



    def train_model(self, dataset_path, save_path, num_classes, batch_size, learning_rate, epochs, progress_callback):
        """训练模型的具体实现"""
        # 从全局配置获取设备
        device = torch.device("cuda:0" if config.get('use_gpu') and torch.cuda.is_available() else "cpu")
        try:
            # 初始化模型
            weights = _allow_torchvision_weights(models.ResNet18_Weights.DEFAULT, "ResNet18")
            model = _build_with_pretrained_fallback(
                models.resnet18,
                "ResNet18",
                weights=weights,
                progress=False,
            )

            # 修改最后一层
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.BatchNorm1d(num_ftrs),
                nn.Dropout(0.5),  # 添加dropout防止过拟合
                nn.Linear(num_ftrs, num_classes)
            )

            # 更保守的权重初始化
            nn.init.kaiming_normal_(model.fc[2].weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(model.fc[2].bias, 0)

            # 将模型移至GPU
            model = model.to(device)

            # 增强数据预处理策略
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.3),  # 添加垂直翻转
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 添加平移
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),  # 随机灰度
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2)  # 随机擦除
            ])

            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # 加载训练集和验证集
            train_path = os.path.join(dataset_path, 'train')
            valid_path = os.path.join(dataset_path, 'valid')

            # 检查数据集结构
            if not (os.path.exists(train_path) and os.path.exists(valid_path)):
                raise Exception("数据集结构不正确，需要包含 train 和 valid 子目录")

            # 加载数据集
            train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
            val_dataset = datasets.ImageFolder(valid_path, transform=val_transform)

            # 检查类别
            train_classes = train_dataset.classes
            val_classes = val_dataset.classes

            if len(train_classes) != num_classes:
                raise Exception(f"训练集类别数量 ({len(train_classes)}) 与指定类别数量 ({num_classes}) 不匹配")

            if train_classes != val_classes:
                raise Exception("训练集和验证集的类别不匹配")

            logging.info(f"类别映射: {train_dataset.class_to_idx}")

            # 创建数据加载器
            configured_workers = int(config.get('num_workers', 0))
            if sys.platform.startswith('win'):
                # Tkinter + Windows multiprocessing DataLoader can make the GUI appear frozen.
                num_workers = 0
            else:
                num_workers = max(0, configured_workers)
            logging.info(f"DataLoader工作进程数: {num_workers}")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

            # 打印数据集信息
            logging.info(f"训练集大小: {len(train_dataset)}")
            logging.info(f"验证集大小: {len(val_dataset)}")
            logging.info(f"类别数量: {len(train_classes)}")
            logging.info(f"类别列表: {train_classes}")

            # 使用更小的初始学习率
            initial_lr = learning_rate  # 不要过度降低初始学习率

            # 优化器设置
            optimizer = torch.optim.AdamW([
                {'params': model.conv1.parameters(), 'lr': initial_lr * 0.1},
                {'params': model.layer1.parameters(), 'lr': initial_lr * 0.1},
                {'params': model.layer2.parameters(), 'lr': initial_lr * 0.2},
                {'params': model.layer3.parameters(), 'lr': initial_lr * 0.5},
                {'params': model.layer4.parameters(), 'lr': initial_lr * 0.7},
                {'params': model.fc.parameters(), 'lr': initial_lr}
            ], weight_decay=0.01)

            # 使用 ReduceLROnPlateau 来根据验证集性能调整学习率
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=3
            )

            # 添加一个函数来记录学习率变化
            def log_lr_change(old_lr, new_lr):
                if old_lr != new_lr:
                    logging.info(f'Learning rate changed from {old_lr:.6f} to {new_lr:.6f}')

            # 损失函数
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

            # 重置训练数据记录
            self.train_data = {
                'batch_indices': [],
                'losses': [],
                'accuracies': []
            }

            total_batches = len(train_loader)

            use_amp = config.get('amp_enabled', True) and device.type == 'cuda'
            accum_steps = max(1, config.get('gradient_accumulation_steps', 1))
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            logging.info(f"混合精度(AMP): {'开启' if use_amp else '关闭'}, 梯度累积步数: {accum_steps}")

            best_val_acc = 0.0
            patience = 0
            max_patience = 10
            last_ui_emit_time = 0.0

            def update_progress_and_plots(epoch, batch_idx, total_batches, avg_loss, accuracy):
                nonlocal last_ui_emit_time
                if self.stop_training:
                    return
                now = time.perf_counter()
                if batch_idx < total_batches and now - last_ui_emit_time < self._progress_update_interval:
                    return
                last_ui_emit_time = now
                self.after(0, lambda: progress_callback(epoch, batch_idx, total_batches, avg_loss, accuracy))
                current_batch = (epoch - 1) * len(train_loader) + batch_idx
                self.after(0, lambda: self.update_plots(current_batch, avg_loss, accuracy))

            self.stop_event = threading.Event()

            def check_stop():
                if self.stop_training or self.stop_event.is_set():
                    logging.info("正在停止训练...")
                    return True
                return False

            for epoch in range(1, epochs + 1):
                if check_stop():
                    break

                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                current_batch = 0
                optimizer.zero_grad()

                for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
                    if check_stop():
                        break

                    try:
                        inputs, labels = inputs.to(device), labels.to(device)

                        with torch.cuda.amp.autocast(enabled=use_amp):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels) / accum_steps

                        scaler.scale(loss).backward()

                        if batch_idx % accum_steps == 0 or batch_idx == len(train_loader):
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                        train_loss += loss.item() * accum_steps
                        _, predicted = outputs.max(1)
                        train_total += labels.size(0)
                        train_correct += predicted.eq(labels).sum().item()

                        current_batch = (epoch - 1) * len(train_loader) + batch_idx

                        if batch_idx % 5 == 0 or batch_idx == len(train_loader):
                            if check_stop():
                                break

                            avg_loss = train_loss / batch_idx
                            accuracy = 100. * train_correct / train_total
                            current_lr = optimizer.param_groups[-1]['lr']

                            update_progress_and_plots(epoch, batch_idx, len(train_loader), avg_loss, accuracy)

                            logging.info(f"Epoch {epoch}, Batch {batch_idx}, LR: {current_lr:.6f}, "
                                         f"Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%")

                    except Exception as e:
                        logging.error(f"处理批次时出错: {str(e)}")
                        continue

                if check_stop():
                    break

                # 验证阶段
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        if check_stop():
                            break

                        try:
                            inputs, labels = inputs.to(device), labels.to(device)
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                            val_loss += loss.item()
                            _, predicted = outputs.max(1)
                            val_total += labels.size(0)
                            val_correct += predicted.eq(labels).sum().item()
                        except Exception as e:
                            logging.error(f"验证时出错: {str(e)}")
                            continue

                if check_stop():
                    break

                val_acc = 100. * val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)

                # 使用最后一个batch的current_batch来更新验证集数据点
                if not check_stop():  # 只在非停止状态下更新
                    self.update_plots(current_batch, avg_loss, accuracy, avg_val_loss, val_acc)

                logging.info(f"Epoch {epoch}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                # 获取当前学习率
                current_lr = optimizer.param_groups[-1]['lr']

                # 更新学习率调度器
                scheduler.step(val_acc)

                # 检查学习率是否改变并记录
                new_lr = optimizer.param_groups[-1]['lr']
                log_lr_change(current_lr, new_lr)

                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), save_path)
                    patience = 0
                else:
                    patience += 1

                # 早停
                if patience >= max_patience:
                    logging.info("Early stopping triggered")
                    break

                # 定期保存检查点
                if epoch % 5 == 0:
                    checkpoint_path = f"{save_path[:-4]}_epoch_{epoch}.pth"
                    torch.save(model.state_dict(), checkpoint_path)

            logging.info(f"Best validation accuracy: {best_val_acc:.2f}%")

        except Exception as e:
            logging.error(f"训练过程出错: {str(e)}")
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: messagebox.showerror("训练失败", msg))
        finally:
            if self.stop_training:
                logging.info("训练已手动停止")
            else:
                logging.info("训练正常完成")

            self.stop_training = False
            self.stop_event.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.after(0, self._close_progress_window)

    def validate_inputs(self):
        """验证输入是否有效"""
        if not self.dataset_path.get():
            messagebox.showerror("错误", "请选择数据集路径")
            return False
        if not self.save_model_path.get():
            messagebox.showerror("错误", "请选择模型保存路径")
            return False
        if not self.class_count.get():
            messagebox.showerror("错误", "请输入类别数量")
            return False
        if not self.batch_size.get():
            messagebox.showerror("错误", "请输入批次大小")
            return False
        if not self.learning_rate.get():
            messagebox.showerror("错误", "请输入学习率")
            return False
        return True

    def stop_recognition(self):
        """安全停止识别过程"""
        if not messagebox.askyesno("确认", "确定要停止当前识别吗？"):
            return

        logging.info("正在安全停止训练...")
        self.stop_training = True
        self.stop_event.set()

        try:
            if hasattr(self, 'processor') and self.processor is not None:
                if hasattr(self.processor, 'stop'):
                    self.processor.stop()
                    logging.info("已发送停止信号到识别处理器")
        except Exception as e:
            logging.error(f"停止识别时发生异常: {str(e)}", exc_info=True)
        finally:
            try:
                if hasattr(self, 'recognition_thread') and self.recognition_thread.is_alive():
                    self.recognition_thread.join(timeout=3)
                    if self.recognition_thread.is_alive():
                        logging.warning("识别线程未在指定时间内终止")
                    else:
                        logging.info("识别线程已安全停止")
            except (AttributeError, RuntimeError) as e:
                logging.error(f"线程操作异常: {str(e)}")

            self.processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
                self.progress_window.destroy()
    def update_dataset_info(self, dataset_path):
        # 实现更新数据集信息逻辑
        pass
    def _close_progress_window(self):
        if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
            self.progress_window.destroy()

    def stop_training_safely(self):
        """安全停止训练的统一方法"""
        if messagebox.askyesno("确认", "确定要停止训练吗？"):
            self.stop_event.set()
            logging.info("停止信号已发送")
            self._close_progress_window()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()