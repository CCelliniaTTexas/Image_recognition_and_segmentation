import os
import csv
import ttkbootstrap as ttk
import matplotlib.pyplot as plt
import logging
import tkinter as tk
import numpy as np
from io import StringIO
from tkinter import filedialog, messagebox, StringVar
from datetime import datetime
from PIL import Image, ImageTk
from collections import Counter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from config import config


class LogWindow(tk.Toplevel):
    """日志窗口"""

    def __init__(self, parent, title="运行日志"):
        super().__init__(parent)
        self.title(title)
        self.geometry("600x400")

        # 创建日志文本框
        self.log_frame = ttk.Frame(self)
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # 创建按钮框架
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, pady=5)

        # 添加清除和保存按钮
        ttk.Button(button_frame, text="清除日志",
                   command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="保存日志",
                   command=self.save_log).pack(side=tk.LEFT, padx=5)

        # 设置日志处理器
        self.log_handler = logging.StreamHandler(StringIO())
        self.log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S'))

        # 配置根日志记录器
        self.logger = logging.getLogger()
        self.logger.addHandler(self.log_handler)
        self.logger.setLevel(logging.INFO)

        # 定期更新日志显示
        self.after(100, self.update_log)

    def update_log(self):
        """更新日志显示"""
        try:
            # 获取新的日志内容
            log_stream = self.log_handler.stream
            content = log_stream.getvalue()
            if content:
                self.log_text.insert(tk.END, content)
                self.log_text.see(tk.END)  # 滚动到底部
                log_stream.truncate(0)
                log_stream.seek(0)

            # 继续定期更新
            self.after(100, self.update_log)
        except Exception as e:
            print(f"更新日志出错: {str(e)}")

    def clear_log(self):
        """清除日志内容"""
        self.log_text.delete(1.0, tk.END)

    def save_log(self):
        """保存日志到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".log",
                initialfile=f"run_log_{timestamp}.log",
                filetypes=[("Log files", "*.log"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("成功", f"日志已保存到：\n{filename}")
        except Exception as e:
            messagebox.showerror("错误", f"保存日志失败：{str(e)}")

class ResultDetailWindow(tk.Toplevel):
    """结果详细信息窗口"""

    def __init__(self, parent, file_path, class_name):
        super().__init__(parent)
        self.title("识别结果详情")
        self.geometry("800x600")

        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 信息区域
        info_frame = tk.LabelFrame(main_frame, text="文件信息")
        info_frame.pack(fill=tk.X, pady=5)

        # 文件信息
        ttk.Label(info_frame, text=f"文件名: {os.path.basename(file_path)}").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, text=f"完整路径: {file_path}").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, text=f"预测类别: {class_name}").pack(anchor=tk.W, padx=5, pady=2)

        try:
            # 获取文件信息
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size / 1024  # 转换为KB
            modify_time = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

            ttk.Label(info_frame, text=f"文件大小: {file_size:.2f} KB").pack(anchor=tk.W, padx=5, pady=2)
            ttk.Label(info_frame, text=f"修改时间: {modify_time}").pack(anchor=tk.W, padx=5, pady=2)

            # 图像信息
            img = Image.open(file_path)
            ttk.Label(info_frame, text=f"图像尺寸: {img.size[0]}x{img.size[1]}").pack(anchor=tk.W, padx=5, pady=2)
            ttk.Label(info_frame, text=f"图像模式: {img.mode}").pack(anchor=tk.W, padx=5, pady=2)

            # 图像预览
            preview_frame = tk.LabelFrame(main_frame, text="图像预览")
            preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

            # 调整图像大小以适应显示
            max_size = (600, 400)
            img.thumbnail(max_size)

            # 显示图像
            photo = ImageTk.PhotoImage(img)
            img_label = ttk.Label(preview_frame, image=photo)
            img_label.pack(pady=10)

            # 保持引用以防止垃圾回收
            self.photo = photo

        except Exception as e:
            ttk.Label(info_frame, text=f"加载图片失败: {str(e)}", foreground="red").pack(pady=5)

class ResultsViewer(tk.Toplevel):
    def __init__(self, parent, results):
        super().__init__(parent)
        self.title("识别结果")
        self.geometry("800x600")

        # 保存结果列表，确保每个结果都包含完整路径
        self.results = [(str(path), str(class_name)) for path, class_name in results]
        self.filtered_results = self.results.copy()  # 用于存储筛选后的结果

        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 添加搜索和筛选框架
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))

        # 搜索框
        search_box_frame = ttk.Frame(search_frame)
        search_box_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(search_box_frame, text="搜索:").pack(side=tk.LEFT, padx=5)
        self.search_var = StringVar()
        self.search_var.trace('w', self.on_search_change)
        search_entry = ttk.Entry(search_box_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 筛选选项
        filter_frame = ttk.Frame(search_frame)
        filter_frame.pack(side=tk.RIGHT)

        ttk.Label(filter_frame, text="类别筛选:").pack(side=tk.LEFT, padx=5)
        self.filter_var = StringVar(value="全部")
        self.class_filter = ttk.Combobox(filter_frame,
                                         textvariable=self.filter_var,
                                         state='readonly')
        self.class_filter.pack(side=tk.LEFT, padx=5)

        # 获取所有唯一的类别
        unique_classes = sorted(set(class_name for _, class_name in self.results))
        self.class_filter['values'] = ['全部'] + unique_classes

        # 绑定筛选事件
        self.class_filter.bind('<<ComboboxSelected>>', self.apply_filter)

        # 创建结果表格
        columns = ("文件", "预测类别")
        self.tree = ttk.Treeview(main_frame, columns=columns, show="headings")

        # 设置列标题
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # 放置表格和滚动条
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 更新表格显示
        self.update_table()

        # 绑定双击事件
        self.tree.bind("<Double-1>", self.on_result_double_click)

        # 添加统计信息
        self.stats_frame = tk.LabelFrame(self, text="统计信息")
        self.stats_frame.pack(fill=tk.X, padx=10, pady=5)

        # 更新统计信息
        self.update_stats()

        # 添加图表框架
        self.plot_frame = tk.LabelFrame(self, text="类别分布")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 创建初始图表
        self.create_stats_plot(Counter(class_name for _, class_name in self.filtered_results))

        # 添加导出按钮
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(button_frame, text="导出结果",
                   command=lambda: self.export_results(self.filtered_results)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="导出统计图",
                   command=self.export_plot).pack(side=tk.LEFT, padx=5)

    def on_search_change(self, *args):
        """搜索框内容变化时的处理"""
        self.apply_filter()

    def apply_filter(self, *args):
        """应用搜索和筛选"""
        search_text = self.search_var.get().lower()
        selected_class = self.filter_var.get()

        self.filtered_results = []
        for path, class_name in self.results:
            if search_text in path.lower() or search_text in class_name.lower():
                if selected_class == "全部" or selected_class == class_name:
                    self.filtered_results.append((path, class_name))

        self.update_table()
        self.update_stats()
        self.create_stats_plot(Counter(class_name for _, class_name in self.filtered_results))

    def update_table(self):
        """更新表格显示"""
        # 清除现有项目
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 添加筛选后的结果
        for path, class_name in self.filtered_results:
            self.tree.insert("", tk.END, values=(path, class_name))

    def update_stats(self):
        """更新统计信息"""
        # 清除现有统计信息
        for widget in self.stats_frame.winfo_children():
            widget.destroy()

        # 计算新的统计信息
        total_images = len(self.filtered_results)
        class_counts = Counter(class_name for _, class_name in self.filtered_results)

        # 显示统计信息
        ttk.Label(self.stats_frame,
                  text=f"显示图片数: {total_images}").pack(side=tk.LEFT, padx=10)
        ttk.Label(self.stats_frame,
                  text=f"类别数: {len(class_counts)}").pack(side=tk.LEFT, padx=10)

    def create_stats_plot(self, class_counts):
        """创建统计图表（主题感知）"""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        try:
            current_theme = self.winfo_toplevel().style.theme_use()
        except Exception:
            current_theme = 'cosmo'
        is_dark = current_theme == 'darkly'

        bg_color = '#2B3E50' if is_dark else 'white'
        text_color = 'white' if is_dark else 'black'
        grid_color = '#486581' if is_dark else '#CCCCCC'
        bar_color = '#5bc0de' if is_dark else '#337ab7'

        self.fig = Figure(figsize=(8, 4))
        self.fig.patch.set_facecolor(bg_color)
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(bg_color)

        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        if classes:
            x_pos = np.arange(len(classes))
            bars = ax.bar(x_pos, counts, color=bar_color, edgecolor=grid_color, linewidth=0.5)

            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(count), ha='center', va='bottom', fontsize=8, color=text_color)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9, color=text_color)

        ax.set_ylabel('数量', fontsize=10, color=text_color)
        ax.set_title('各类别图片数量分布', fontsize=12, color=text_color)
        ax.tick_params(colors=text_color)
        ax.grid(True, linestyle='--', alpha=0.3, color=grid_color)

        for spine in ax.spines.values():
            spine.set_color(grid_color)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_results(self, results):
        """导出识别结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile=f"recognition_results_{timestamp}.csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["文件", "预测类别"])
                    for path, class_name in results:
                        writer.writerow([path, class_name])
                messagebox.showinfo("成功", f"结果已保存到：\n{filename}")
        except Exception as e:
            messagebox.showerror("错误", f"导出结果失败：{str(e)}")

    def export_plot(self):
        """导出统计图表"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialfile=f"recognition_stats_{timestamp}.png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("成功", f"图表已保存到：\n{filename}")
        except Exception as e:
            messagebox.showerror("错误", f"导出图表失败：{str(e)}")

    def on_result_double_click(self, event):
        """双击结果项时的处理"""
        try:
            item = self.tree.selection()[0]
            values = self.tree.item(item)['values']
            if not values:
                return

            file_path, class_name = values

            # 打开详细信息窗口
            ResultDetailWindow(self, file_path, class_name)

        except Exception as e:
            messagebox.showerror("错误", f"显示详细信息失败：{str(e)}")
