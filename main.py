import sys
import ttkbootstrap as ttk
import traceback
import sqlite3
import torch
import logging
import os
import ctypes
import threading
import  tkinter as tk
import matplotlib.pyplot as plt
from CuttingTab import CuttingTab
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms
from config import config, multiprocessing  # 全局配置
from _init_ import debounce, device
from Setting import CacheManager, MemoryMonitor, SettingsTab
from tkinter import messagebox
from config import RecentFiles
from RecognitionTraining import TrainingTab
from RecognitionTab import RecognitionTab
from SegmentationTraining import SegmentationTrainingTab
from SegmentationPredictor import SegmentationPredictTab

try:
    import pystray  # type: ignore[reportMissingImports]
except Exception:
    pystray = None


def main():
    print(f"当前设备: {device}")

@debounce()
def my_function():
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 从配置中获取设备状态
device = torch.device("cuda:0" if config.get('use_gpu', torch.cuda.is_available()) else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def handle_exception(exc_type, exc_value, exc_traceback):
    """全局异常处理器"""
    logging.error("未捕获的异常:", exc_info=(exc_type, exc_value, exc_traceback))
    messagebox.showerror("错误", str(exc_value))


sys.excepthook = handle_exception

def create_gui():
    valid_themes = ['cosmo', 'darkly', 'flatly', 'journal']
    current_theme = config.get('theme', 'cosmo')
    if current_theme not in valid_themes:
        current_theme = 'cosmo'
        config.set('theme', current_theme)

    # 创建无边框窗口
    root = ttk.Window(themename="cosmo")
    root.overrideredirect(False)
    root.geometry("1600x1000")

    # 保存初始大小和位置
    root.normal_size = (1200, 800)
    root.normal_pos = (root.winfo_x(), root.winfo_y())
    root.is_maximized = False

    # 预览窗口
    class ResizePreview(tk.Toplevel):
        def __init__(self, master):
            super().__init__(master)
            self.overrideredirect(True)
            self.attributes('-alpha', 0.3)
            self.configure(bg='gray')
            self.withdraw()
            self.attributes('-topmost', True)

        def show(self, x, y, width, height):
            self.geometry(f"{width}x{height}+{x}+{y}")
            self.deiconify()

        def hide(self):
            self.withdraw()

    # 边框调整大小
    class BorderResizer:
        def __init__(self, root):
            self.root = root
            self.preview = ResizePreview(root)
            self.edges = {
                'n': {'cursor': 'sb_v_double_arrow', 'height': 5},
                's': {'cursor': 'sb_v_double_arrow', 'height': 5},
                'e': {'cursor': 'sb_h_double_arrow', 'width': 5},
                'w': {'cursor': 'sb_h_double_arrow', 'width': 5}
            }
            self.setup_borders()

        def setup_borders(self):
            for edge, props in self.edges.items():
                frame = ttk.Frame(self.root, cursor=props['cursor'])
                if 'height' in props:
                    frame.configure(height=props['height'])
                if 'width' in props:
                    frame.configure(width=props['width'])

                frame.bind('<Button-1>', lambda e, edge=edge: self.start_resize(e, edge))
                frame.bind('<B1-Motion>', lambda e, edge=edge: self.do_resize(e, edge))
                frame.bind('<ButtonRelease-1>', self.stop_resize)

                if edge == 'n':
                    frame.place(relx=0, rely=0, relwidth=1, height=5)
                elif edge == 's':
                    frame.place(relx=0, rely=1, relwidth=1, height=5, anchor='sw')
                elif edge == 'e':
                    frame.place(relx=1, rely=0, width=5, relheight=1, anchor='ne')
                elif edge == 'w':
                    frame.place(relx=0, rely=0, width=5, relheight=1)

        def start_resize(self, event, edge):
            if not self.root.is_maximized:
                self.edge = edge
                self.start_x = event.x_root
                self.start_y = event.y_root
                self.start_width = self.root.winfo_width()
                self.start_height = self.root.winfo_height()
                self.start_pos_x = self.root.winfo_x()
                self.start_pos_y = self.root.winfo_y()
                self.preview.show(self.start_pos_x, self.start_pos_y,
                                  self.start_width, self.start_height)

        def do_resize(self, event, edge):
            if not self.root.is_maximized:
                delta_x = event.x_root - self.start_x
                delta_y = event.y_root - self.start_y
                width = self.start_width
                height = self.start_height
                x = self.start_pos_x
                y = self.start_pos_y

                if edge in ['e', 'w']:
                    if edge == 'e':
                        width += delta_x
                    else:
                        width -= delta_x
                        x += delta_x

                if edge in ['n', 's']:
                    if edge == 's':
                        height += delta_y
                    else:
                        height -= delta_y
                        y += delta_y

                width = max(800, width)
                height = max(600, height)
                self.preview.show(x, y, width, height)

        def stop_resize(self, event):
            if not self.root.is_maximized:
                self.preview.hide()
                x = self.preview.winfo_x()
                y = self.preview.winfo_y()
                width = self.preview.winfo_width()
                height = self.preview.winfo_height()
                self.root.geometry(f"{width}x{height}+{x}+{y}")

    BorderResizer(root)

    config.load_config()
    root.option_add("*Font", ("微软雅黑", 12))

    style = root.style
    current_theme = config.get('theme', 'cosmo')
    style.theme_use(current_theme)
    root.title("图像识别分割系统")

    class TrayManager:
        def __init__(self, root_window):
            self.root = root_window
            self.icon = None
            self.thread = None

        @staticmethod
        def _create_icon_image():
            image = Image.new("RGBA", (64, 64), (22, 126, 251, 255))
            draw = ImageDraw.Draw(image)
            draw.rectangle((12, 12, 52, 52), fill=(17, 94, 182, 255))
            draw.rectangle((20, 20, 44, 44), fill=(255, 255, 255, 255))
            return image

        def _ensure_icon(self):
            if pystray is None:
                return False
            if self.icon is not None:
                return True

            menu = pystray.Menu(
                pystray.MenuItem("显示主窗口", lambda icon, item: self._restore()),
                pystray.MenuItem("退出程序", lambda icon, item: self._exit())
            )
            self.icon = pystray.Icon("adl_tool", self._create_icon_image(), "图像识别分割系统", menu)

            def run_icon():
                self.icon.run()

            self.thread = threading.Thread(target=run_icon, daemon=True)
            self.thread.start()
            return True

        def minimize_to_tray(self):
            if not self._ensure_icon():
                logging.warning("系统托盘不可用，已回退为任务栏最小化")
                self.root.iconify()
                return
            self.root.withdraw()

        def _restore(self):
            self.root.after(0, self.restore)

        def restore(self):
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()

        def _exit(self):
            self.root.after(0, lambda: on_closing(self.root, force_exit=True))

        def stop(self):
            if self.icon is not None:
                try:
                    self.icon.stop()
                except Exception:
                    pass
                self.icon = None

    root.tray_manager = TrayManager(root)

    def apply_windows_titlebar_effect(theme_name):
        """让 Windows 原生标题栏与软件主题同色。"""
        if sys.platform != "win32":
            return
        try:
            def _hex_to_colorref(hex_color):
                hex_color = hex_color.lstrip('#')
                if len(hex_color) != 6:
                    return 0xFFFFFF
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return (b << 16) | (g << 8) | r

            hwnd = root.winfo_id()
            user32 = ctypes.windll.user32
            top_hwnd = user32.GetParent(hwnd)
            if top_hwnd:
                hwnd = top_hwnd

            dwmapi = ctypes.windll.dwmapi
            use_dark = ctypes.c_int(1 if theme_name == "darkly" else 0)
            backdrop = ctypes.c_int(1)  # 1=None，避免 Mica 叠加色偏
            caption_color = ctypes.c_uint32(_hex_to_colorref(style.colors.bg))
            text_color = ctypes.c_uint32(_hex_to_colorref(style.colors.fg))
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_BORDER_COLOR = 34
            DWMWA_CAPTION_COLOR = 35
            DWMWA_TEXT_COLOR = 36
            DWMWA_SYSTEMBACKDROP_TYPE = 38
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(use_dark),
                ctypes.sizeof(use_dark)
            )
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_SYSTEMBACKDROP_TYPE,
                ctypes.byref(backdrop),
                ctypes.sizeof(backdrop)
            )
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_CAPTION_COLOR,
                ctypes.byref(caption_color),
                ctypes.sizeof(caption_color)
            )
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_TEXT_COLOR,
                ctypes.byref(text_color),
                ctypes.sizeof(text_color)
            )
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_BORDER_COLOR,
                ctypes.byref(caption_color),
                ctypes.sizeof(caption_color)
            )
        except Exception:
            # 系统版本不支持时静默回退
            pass

    def update_chrome_style():
        bg_color = style.colors.bg
        fg_color = style.colors.fg
        border_color = style.colors.border
        root.configure(bg=bg_color)
        if 'paned' in locals():
            paned.configure(bg=bg_color, sashrelief='flat', bd=0)
        if 'left_pane' in locals():
            left_pane.configure(bg=bg_color, bd=0, highlightthickness=0)
        if 'log_pane' in locals():
            log_pane.configure(bg=bg_color, fg=fg_color, bd=1, highlightthickness=1,
                               highlightbackground=border_color, highlightcolor=border_color)

    main_container = ttk.Frame(root)
    main_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    paned = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, sashrelief='flat',
                           sashwidth=4, opaqueresize=False, bd=0)
    paned.pack(fill=tk.BOTH, expand=True)

    # 创建左右面板
    left_pane = tk.Frame(paned, bd=0, highlightthickness=0)
    log_pane = tk.LabelFrame(paned, text="运行日志", bd=1, highlightthickness=1)

    left_pane.config(width=1260)
    left_pane.pack_propagate(False)

    log_pane.config(width=340)
    log_pane.pack_propagate(False)
    update_chrome_style()

    # 布局内部内容
    left_frame = ttk.Frame(left_pane)
    left_frame.pack(fill=tk.BOTH, expand=True)

    log_frame_inner = ttk.Frame(log_pane)
    log_frame_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    paned.add(left_pane)
    paned.add(log_pane)

    auto_scroll_var = tk.BooleanVar(value=config.get('log_auto_scroll', True))

    def _save_log_auto_scroll_setting(*_args):
        config.set('log_auto_scroll', auto_scroll_var.get())

    auto_scroll_var.trace_add("write", _save_log_auto_scroll_setting)

    # 日志文本框
    log_text = tk.Text(log_frame_inner, wrap=tk.WORD, height=1)

    def maybe_reenable_auto_scroll():
        """自动滚动关闭时，若用户滚动到底部则自动重新开启。"""
        if not auto_scroll_var.get() and log_text.yview()[1] >= 0.999:
            auto_scroll_var.set(True)

    def on_scrollbar(*args):
        log_text.yview(*args)
        maybe_reenable_auto_scroll()

    scrollbar = ttk.Scrollbar(log_frame_inner, orient=tk.VERTICAL, command=on_scrollbar)
    log_text.configure(yscrollcommand=scrollbar.set)

    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 0), pady=0)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def update_log_text_colors(theme_name):
        default_color = "#ffffff" if theme_name == "darkly" else "#000000"
        log_text.tag_configure("log_error", foreground="#d62728")
        log_text.tag_configure("log_training", foreground="#2ca02c")
        log_text.tag_configure("log_memory_time", foreground="#808080")
        log_text.tag_configure("log_start_stop", foreground="#1f77b4")
        log_text.tag_configure("log_save", foreground="#d4a017")
        log_text.tag_configure("log_default", foreground=default_color)

    update_log_text_colors(current_theme)

    log_text.configure(state='disabled')

    def on_log_mousewheel(event):
        log_text.yview_scroll(int(-event.delta / 120), "units")
        maybe_reenable_auto_scroll()
        return "break"

    def on_log_keyrelease(_event):
        maybe_reenable_auto_scroll()

    log_text.bind("<MouseWheel>", on_log_mousewheel)
    log_text.bind("<KeyRelease>", on_log_keyrelease)

    class TextHandler(logging.Handler):
        MAX_LINES = 500

        def __init__(self, text_widget, auto_scroll_variable):
            super().__init__()
            self.text_widget = text_widget
            self.auto_scroll_variable = auto_scroll_variable

        @staticmethod
        def _resolve_log_tag(record, msg):
            message = record.getMessage()

            if record.levelno >= logging.ERROR:
                return "log_error"

            if any(k in message for k in ("程序启动", "启动主循环", "程序结束", "程序正常退出")):
                return "log_start_stop"

            if "内存使用" in message or "时间" in message:
                return "log_memory_time"

            if any(k in message for k in ("保存", "已保存", "写入", "导出", "另存为")):
                return "log_save"

            if any(k in message for k in ("训练", "Epoch", "mIoU", "Loss", "OA", "验证")):
                return "log_training"

            return "log_default"

        def emit(self, record):
            msg = self.format(record)
            tag = self._resolve_log_tag(record, msg)

            def append():
                text = self.text_widget
                text.configure(state='normal')
                text.insert(tk.END, msg + '\n', tag)
                lines = int(text.index('end-1c').split('.')[0])
                if lines > self.MAX_LINES:
                    text.delete(1.0, f"{lines - self.MAX_LINES}.0")

                # 可选自动滚动到最新日志
                if self.auto_scroll_variable.get():
                    text.see(tk.END)

                text.configure(state='disabled')

            try:
                self.text_widget.after(0, append)
            except RuntimeError:
                # 主线程退出后，后台线程日志可能到达，直接忽略以避免级联异常
                pass

    def setup_logging(text_widget):
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.setLevel(logging.INFO)
        text_handler = TextHandler(text_widget, auto_scroll_var)
        text_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(text_handler)
        return logger

    logger = setup_logging(log_text)

    notebook = ttk.Notebook(left_frame)
    notebook.pack(expand=1, fill="both", padx=10, pady=5)

    model_tools_frame = ttk.Frame(notebook)
    notebook.add(model_tools_frame, text="模型工具")

    model_notebook = ttk.Notebook(model_tools_frame)
    model_notebook.pack(expand=1, fill="both", padx=5, pady=5)

    recent_files = RecentFiles()

    training_tab = TrainingTab(model_notebook)
    recognition_tab = RecognitionTab(model_notebook)
    segmentation_training_tab = SegmentationTrainingTab(model_notebook)
    segmentation_predict_tab = SegmentationPredictTab(model_notebook)

    model_notebook.add(training_tab, text="分类训练")
    model_notebook.add(recognition_tab, text="分类预测")
    model_notebook.add(segmentation_training_tab, text="分割训练")
    model_notebook.add(segmentation_predict_tab, text="分割预测")
    model_notebook.add(CuttingTab(model_notebook), text="图像裁剪")

    settings_tab = SettingsTab(notebook, log_auto_scroll_var=auto_scroll_var)
    notebook.add(settings_tab, text="设置")

    status_frame = ttk.Frame(main_container)
    status_frame.pack(fill=tk.X, side=tk.BOTTOM)

    if torch.cuda.is_available():
        ttk.Label(status_frame, text=f"GPU: {torch.cuda.get_device_name(0)}").pack(side=tk.LEFT, padx=10)

    def update_device_status():
        use_gpu = config.get('use_gpu', 'auto')
        current_device = torch.device("cuda:0" if (use_gpu == 'auto' and torch.cuda.is_available()) or use_gpu is True else "cpu")
        device_var = tk.StringVar(value="GPU" if current_device.type == "cuda" else "CPU")
        root.after(1000, update_device_status)

    def show_error(message):
        messagebox.showerror("错误", message)
        logging.error(message)

    root.report_callback_exception = lambda *args: show_error(str(args[1]))

    def apply_theme(theme_name):
        style.theme_use(theme_name)
        update_chrome_style()
        apply_windows_titlebar_effect(theme_name)
        config.set('theme', theme_name)
        main_container.configure(style='TFrame')
        update_log_text_colors(theme_name)
        root.update_idletasks()

    root.apply_theme = apply_theme
    root.after(0, lambda: apply_windows_titlebar_effect(current_theme))
    root.bind('<Map>', lambda _e: apply_windows_titlebar_effect(config.get('theme', current_theme)))

    return root


# 错误处理器
def enhanced_error_handler(exc_type, exc_value, exc_traceback):
    """错误处理器"""
    error_msg = str(exc_value)
    details = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

    # 记录错误
    logging.error(f"未捕获的异常:\n{details}")

    # 记录内存使用情况
    MemoryMonitor.log_memory_usage()

    try:
        # 错误对话框
        error_dialog = tk.Toplevel()
        error_dialog.title("错误")
        error_dialog.geometry("600x400")

        # 错误消息
        msg_frame = ttk.Frame(error_dialog)
        msg_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(msg_frame, text="发生错误:",
                  font=("微软雅黑", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(msg_frame, text=error_msg,
                  wraplength=550).pack(anchor=tk.W)

        # 详细信息
        detail_frame = tk.LabelFrame(error_dialog, text="详细信息")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 文本框和滚动条
        text = tk.Text(detail_frame, wrap=tk.WORD, height=10)
        scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)

        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text.insert('1.0', details)
        text.configure(state='disabled')

        # 按钮
        btn_frame = ttk.Frame(error_dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(btn_frame, text="复制详细信息",
                   command=lambda: error_dialog.clipboard_append(details)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="确定",
                   command=error_dialog.destroy).pack(side=tk.RIGHT)
    except Exception as e:
        logging.error(f"显示错误对话框失败: {str(e)}")
        messagebox.showerror("错误", error_msg)


def ask_close_behavior(root):
    """弹出关闭行为选择框，返回 (action, remember) 或 (None, False)。"""
    dialog = tk.Toplevel(root)
    dialog.title("关闭程序")
    dialog.geometry("360x210")
    dialog.resizable(False, False)
    dialog.transient(root)
    dialog.grab_set()

    ttk.Label(dialog, text="关闭窗口时执行：", font=("微软雅黑", 11, "bold")).pack(anchor=tk.W, padx=15, pady=(12, 6))
    action_var = tk.StringVar(value="minimize")
    ttk.Radiobutton(dialog, text="最小化到托盘（不关闭程序）", value="minimize", variable=action_var).pack(anchor=tk.W, padx=20, pady=2)
    ttk.Radiobutton(dialog, text="退出程序", value="exit", variable=action_var).pack(anchor=tk.W, padx=20, pady=2)

    remember_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(dialog, text="以后保持该选项", variable=remember_var).pack(anchor=tk.W, padx=20, pady=(10, 6))

    result = {"action": None}

    def on_confirm():
        result["action"] = action_var.get()
        dialog.destroy()

    def on_cancel():
        result["action"] = None
        dialog.destroy()

    btn_frame = ttk.Frame(dialog)
    btn_frame.pack(fill=tk.X, padx=15, pady=(5, 10))
    ttk.Button(btn_frame, text="取消", command=on_cancel).pack(side=tk.RIGHT, padx=5)
    ttk.Button(btn_frame, text="确定", style="success.TButton", command=on_confirm).pack(side=tk.RIGHT)

    dialog.protocol("WM_DELETE_WINDOW", on_cancel)
    root.wait_window(dialog)
    return result["action"], remember_var.get()


def on_closing(root, force_exit=False):
    """窗口关闭处理"""
    try:
        close_action_mode = 'exit' if force_exit else config.get('close_action_mode', 'ask')

        if close_action_mode == 'ask':
            action, remember = ask_close_behavior(root)
            if action is None:
                return
            if remember:
                config.set('close_action_mode', action)
            else:
                config.set('close_action_mode', 'ask')
        else:
            action = close_action_mode

        if action == 'minimize':
            tray_manager = getattr(root, "tray_manager", None)
            if tray_manager is not None:
                tray_manager.minimize_to_tray()
            else:
                root.iconify()
            logging.info("窗口已最小化到系统托盘")
            return

        # action == 'exit'
        tray_manager = getattr(root, "tray_manager", None)
        if tray_manager is not None:
            tray_manager.stop()
        config.save_config()
        logging.info("程序正常退出")
        cleanup_resources()
        root.quit()
    except Exception as e:
        logging.error(f"退出时发生错误: {str(e)}")
        root.quit()


def cleanup_resources():
    try:
        cache_manager = CacheManager()
        cache_manager.clear_cache()

        temp_dir = Path.home() / '.adl_tool' / 'temp'
        if temp_dir.exists():
            for file in temp_dir.glob('*'):
                try:
                    file.unlink()
                except Exception as e:
                    logging.warning(f"无法删除临时文件 {file}: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        MemoryMonitor.log_memory_usage()
        logging.info("资源清理完成")
    except Exception as e:
        logging.error(f"清理资源时发生错误: {str(e)}")


# 主程序入口点
if __name__ == "__main__":
    main()
    for _ in range(5):
        my_function()
    try:
        # 设置增强的错误处理器
        sys.excepthook = enhanced_error_handler

        # 创建缓存管理器
        cache_manager = CacheManager()

        # 应用GPU显存限制
        if torch.cuda.is_available() and config.get('use_gpu', True):
            fraction = config.get('gpu_memory_fraction', 0.8)
            try:
                torch.cuda.set_per_process_memory_fraction(fraction)
                logging.info(f"GPU显存限制: {int(fraction * 100)}%")
            except Exception as e:
                logging.warning(f"设置GPU显存限制失败: {e}")

        # 记录初始内存使用
        MemoryMonitor.log_memory_usage()

        # 创建主窗口
        root = create_gui()

        # 绑定关闭事件
        root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))


        # 定期记录内存使用情况
        def log_memory_periodically():
            MemoryMonitor.log_memory_usage()
            root.after(300000, log_memory_periodically)  # 每5分钟记录一次


        root.after(0, log_memory_periodically)

        # 记录启动日志
        logging.info("程序启动")

        # 启动主循环
        logging.info("启动主循环")
        root.mainloop()

    except Exception as e:
        logging.error(f"程序启动失败: {str(e)}")
        messagebox.showerror("错误", str(e))
        sys.exit(1)
    finally:
        logging.info("程序结束")
        cleanup_resources()