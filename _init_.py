import sys
import logging
from functools import wraps

import torch
from config import config

from tkinter import messagebox


def debounce():
    """防抖动装饰器 - 确保函数只被调用一次"""
    has_been_called = [False]

    def decorator(fn):
        @wraps(fn)
        def debounced(*args, **kwargs):
            if not has_been_called[0]:
                fn(*args, **kwargs)
                has_been_called[0] = True
        return debounced
    return decorator


def get_device():
    """根据配置获取当前计算设备"""
    use_gpu = config.get('use_gpu', 'auto')
    if use_gpu == 'auto':
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")


device = get_device()


def handle_exception(exc_type, exc_value, exc_traceback):
    """全局异常处理器"""
    logging.error("未捕获的异常:", exc_info=(exc_type, exc_value, exc_traceback))
    try:
        messagebox.showerror("错误", str(exc_value))
    except Exception:
        pass


sys.excepthook = handle_exception