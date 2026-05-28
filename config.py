import json
from pathlib import Path
import torch
import logging
import multiprocessing

class Config:
    """配置管理器"""
    def __init__(self):
        self.config_dir = Path.home() / '.adl_tool'
        self.config_file = self.config_dir / 'config.json'
        self.default_config = {
            'theme': 'cosmo',
            'use_gpu': torch.cuda.is_available(),
            'num_workers': max(1, multiprocessing.cpu_count() - 1),
            'cpu_limit': 50,
            'gpu_memory_fraction': 0.8,
            'amp_enabled': True,
            'gradient_accumulation_steps': 1,
            'grad_clip_norm': 1.0,
            'loss_smoothing_alpha': 0.12,
            'auto_save_training_curves': True,
            'recent_files': {'models': [], 'class_files': [], 'folders': []}
        }
        self.load_config()

    def load_config(self):
        """加载配置"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)
        if not self.config_file.exists():
            self.config = self.default_config
            self.save_config()
        else:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception:
                self.config = {}

    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存配置失败: {str(e)}")

    def get(self, key, default=None):
        """获取配置项"""
        return self.config.get(key, default)

    def set(self, key, value):
        """设置配置项"""
        self.config[key] = value
        self.save_config()

# 创建全局配置实例
config = Config()


class RecentFiles:
    """最近使用的文件管理器"""

    def __init__(self, max_items=5):
        self.max_items = max_items
        self.config_dir = Path.home() / '.adl_tool'
        self.config_file = self.config_dir / 'recent.json'
        self.recent_files = self.load_recent_files()

    def load_recent_files(self):
        """加载最近使用的文件记录"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)
        if not self.config_file.exists():
            return {'models': [], 'class_files': [], 'folders': []}
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {'models': [], 'class_files': [], 'folders': []}

    def save_recent_files(self):
        """保存最近使用的文件记录"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.recent_files, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存最近文件记录失败: {str(e)}")

    def add_recent_file(self, file_path, file_type):
        """添加最近使用的文件"""
        if not file_path:
            return

        file_path = str(file_path)

        if file_path in self.recent_files[file_type]:
            self.recent_files[file_type].remove(file_path)

        self.recent_files[file_type].insert(0, file_path)

        self.recent_files[file_type] = self.recent_files[file_type][:self.max_items]

        self.save_recent_files()

    def get_recent_files(self, file_type):
        """获取指定类型的最近文件列表"""
        return self.recent_files.get(file_type, [])

    def clear_recent_files(self, file_type=None):
        """清除最近文件记录"""
        if file_type:
            self.recent_files[file_type] = []
        else:
            self.recent_files = {'models': [], 'class_files': [], 'folders': []}
        self.save_recent_files()