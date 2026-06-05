import tkinter as tk
import ttkbootstrap as ttk
from tkinter import colorchooser, filedialog, messagebox
from ttkbootstrap.constants import *
import os
import csv
import threading
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision import transforms
from config import config, RecentFiles
from SegmentationModelRegistry import (
    MODEL_BUILDERS,
    get_architectures,
    architecture_requires_backbone,
    get_backbones,
    get_default_backbone,
    get_architecture_param_schema,
    normalize_architecture_params,
    compose_model_name,
    get_default_model_name,
    detect_model_type_from_state_dict,
    build_model,
)
from SegmentationAttentionRegistry import AttentionEnhancedModel, CGAMEnhancedModel, resolve_attention_in_channels


class SegmentationPredictor:
    def __init__(self, model_path, num_classes, model_type=None, architecture_params=None):
        self.device = torch.device('cuda' if config.get('use_gpu', True) and torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        model_type = model_type or get_default_model_name()
        self.architecture_params = architecture_params or {}
        self.model = self._load_model(model_path, num_classes, model_type)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def _detect_model_type(state_dict):
        return detect_model_type_from_state_dict(state_dict)

    @staticmethod
    def _detect_num_classes(state_dict):
        """Auto-detect num_classes from the final classifier layer weight shape."""
        for key in (
                'base_model.classifier.classifier.1.weight', 'classifier.classifier.1.weight',
                'base_model.classifier.classifier.weight', 'classifier.classifier.weight',
                'base_model.classifier.3.weight', 'classifier.3.weight',
                'base_model.classifier.2.weight', 'classifier.2.weight',
                'base_model.fuse.2.weight', 'fuse.2.weight',
                'base_model.classifier.4.1.weight', 'classifier.4.1.weight',
                'base_model.classifier.4.weight', 'base_model.classifier.4.bias',
                'classifier.4.weight', 'classifier.4.bias',
                'base_model.classifier.1.weight', 'classifier.1.weight',
                'base_model.classifier.weight', 'classifier.weight',
                'model.decode_head.classifier.weight', 'decode_head.classifier.weight',
                'model.auxiliary_head.classifier.weight', 'auxiliary_head.classifier.weight'
        ):
            if key in state_dict:
                return state_dict[key].shape[0]
        return None

    @staticmethod
    def _is_wrapped_model(state_dict):
        """Check if state dict comes from an AttentionEnhancedModel wrapper."""
        return any(k.startswith('base_model.') for k in state_dict)

    @staticmethod
    def _has_cgam(state_dict):
        """Check if state dict contains CGAM module parameters."""
        return any('cgam.' in k for k in state_dict)

    @staticmethod
    def _has_cross_attention(state_dict):
        """Check if state dict contains cross-attention wrapper parameters."""
        return any(k.startswith('attention.') for k in state_dict)

    @staticmethod
    def _uses_decoder_dropout(state_dict):
        """Check whether classifier head stores weights under dropout-wrapped keys."""
        return any(
            ('classifier.4.1.weight' in k) or ('classifier.1.weight' in k) or ('classifier.classifier.1.weight' in k)
            for k in state_dict.keys()
        )

    @staticmethod
    def _inject_decoder_dropout_for_predict(base_model, p=0.2):
        """Align predictor model structure with training model that wrapped decoder conv by Dropout2d."""
        classifier = getattr(base_model, 'classifier', None)
        if classifier is None:
            return

        if isinstance(classifier, nn.Conv2d):
            base_model.classifier = nn.Sequential(
                nn.Dropout2d(p=p),
                classifier
            )
            return

        if hasattr(classifier, '__getitem__'):
            try:
                target_layer = classifier[4]
                if isinstance(target_layer, nn.Conv2d):
                    classifier[4] = nn.Sequential(
                        nn.Dropout2d(p=p),
                        target_layer
                    )
            except Exception:
                pass
        elif hasattr(classifier, 'classifier') and isinstance(classifier.classifier, nn.Conv2d):
            classifier.classifier = nn.Sequential(
                nn.Dropout2d(p=p),
                classifier.classifier
            )

    def _load_model(self, model_path, num_classes, model_type):
        state_dict = torch.load(model_path, map_location=self.device)

        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}

        detected_type = self._detect_model_type(state_dict)
        if detected_type != model_type:
            logging.warning(f"选择的架构为 {model_type}，但权重文件匹配 {detected_type}，已自动切换")
            model_type = detected_type

        detected_classes = self._detect_num_classes(state_dict)
        if detected_classes is not None and detected_classes != num_classes:
            logging.warning(f"设置的类别数为 {num_classes}，但权重文件包含 {detected_classes} 个类别，已自动修正")
            num_classes = detected_classes
            self.num_classes = num_classes

        if model_type not in MODEL_BUILDERS:
            logging.warning(f"选择的模型组合 {model_type} 不受支持，已回退默认模型")
            model_type = get_default_model_name()

        has_aux = any('aux_classifier' in k for k in state_dict)
        architecture = model_type.split(" (")[0] if " (" in model_type else model_type
        arch_params = normalize_architecture_params(architecture, self.architecture_params)
        base_model, family, feature_dim = build_model(
            model_type, num_classes, use_pretrained=False, aux_loss=has_aux, architecture_params=arch_params
        )
        attention_channels = resolve_attention_in_channels(model_type, base_model, feature_dim)

        if self._uses_decoder_dropout(state_dict):
            self._inject_decoder_dropout_for_predict(base_model)
            logging.info("检测到带解码器Dropout的权重，已自动匹配加载结构")

        has_cgam = self._has_cgam(state_dict)
        has_cross_attention = self._has_cross_attention(state_dict)
        wrapped = self._is_wrapped_model(state_dict)

        if has_cgam and has_cross_attention:
            # 同时包含 CGAM 和交叉注意力：CGAMEnhancedModel + AttentionEnhancedModel
            inner = CGAMEnhancedModel(base_model, in_channels=attention_channels, family=family)
            model = AttentionEnhancedModel(inner, in_channels=attention_channels, family=family)
            model.load_state_dict(state_dict)
            logging.info(f"模型加载成功（含CGAM曲率注意力 + 交叉注意力）: {model_type}, 类别数: {num_classes}")
        elif has_cgam:
            # 仅 CGAM
            model = CGAMEnhancedModel(base_model, in_channels=attention_channels, family=family)
            model.load_state_dict(state_dict)
            logging.info(f"模型加载成功（含CGAM曲率注意力）: {model_type}, 类别数: {num_classes}")
        elif has_cross_attention or wrapped:
            # 仅交叉注意力（原有逻辑）
            model = AttentionEnhancedModel(base_model, in_channels=attention_channels, family=family)
            model.load_state_dict(state_dict)
            logging.info(f"模型加载成功（含交叉注意力）: {model_type}, 类别数: {num_classes}")
        else:
            base_model.load_state_dict(state_dict)
            model = base_model
            logging.info(f"模型加载成功（标准模型）: {model_type}, 类别数: {num_classes}")

        model = model.to(self.device)
        return model

    @staticmethod
    def _default_color_map(num_classes):
        colors = {}
        for class_id in range(num_classes):
            if class_id == 0:
                colors[class_id] = (0, 0, 0)
            else:
                colors[class_id] = (
                    (37 * class_id) % 256,
                    (97 * class_id) % 256,
                    (173 * class_id) % 256,
                )
        return colors

    @staticmethod
    def _colorize_mask(mask_array, color_map):
        rgb = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            rgb[mask_array == class_id] = color
        return Image.fromarray(rgb, mode='RGB')

    @staticmethod
    def _update_confusion_matrix(confusion_matrix, pred, target, num_classes):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        valid = (target_flat >= 0) & (target_flat < num_classes)
        bins = np.bincount(
            num_classes * target_flat[valid].astype(np.int64) + pred_flat[valid].astype(np.int64),
            minlength=num_classes ** 2
        )
        confusion_matrix += bins.reshape(num_classes, num_classes)

    @staticmethod
    def _miou_from_confusion(confusion_matrix):
        cm = confusion_matrix.astype(np.float64)
        tp = np.diag(cm)
        pred_sum = cm.sum(axis=0)
        target_sum = cm.sum(axis=1)
        union = target_sum + pred_sum - tp
        class_iou = np.divide(tp, union, out=np.zeros_like(tp, dtype=np.float64), where=union > 0)
        miou = float(class_iou.mean()) if class_iou.size else 0.0
        return class_iou, miou

    @staticmethod
    def _find_label_path(label_dir, filename):
        name, ext = os.path.splitext(filename)
        candidates = [
            os.path.join(label_dir, f"{name}_lab.png"),
            os.path.join(label_dir, f"{name}.png"),
            os.path.join(label_dir, filename),
            os.path.join(label_dir, f"{name}{ext}"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def _infer_label_dir(input_dir):
        input_dir = os.path.abspath(input_dir)
        parent_dir = os.path.dirname(input_dir)
        candidates = []

        if os.path.basename(input_dir).lower() == "images":
            candidates.append(os.path.join(parent_dir, "masks"))

        candidates.extend([
            os.path.join(input_dir, "masks"),
            os.path.join(parent_dir, "masks"),
            os.path.join(parent_dir, "labels"),
        ])

        for path in candidates:
            if os.path.isdir(path):
                return path
        return None

    @staticmethod
    def _load_label_mask(label_path, target_size, num_classes):
        label = Image.open(label_path)
        if label.size != target_size:
            label = label.resize(target_size, Image.NEAREST)
        label_array = np.array(label)
        if label_array.ndim == 3:
            label_array = label_array[:, :, 0]
        label_array = label_array.astype(np.int64)
        unique_values = sorted(np.unique(label_array).tolist())
        if unique_values and (min(unique_values) < 0 or max(unique_values) >= num_classes):
            value_to_class = {value: idx for idx, value in enumerate(unique_values[:num_classes])}
            remapped = np.zeros_like(label_array, dtype=np.int64)
            for value, class_id in value_to_class.items():
                remapped[label_array == value] = class_id
            label_array = remapped
        return label_array

    @staticmethod
    def _save_prediction_miou(output_dir, class_iou, miou):
        metrics_path = os.path.join(output_dir, "prediction_miou.csv")
        with open(metrics_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "miou"])
            for class_id, iou in enumerate(class_iou):
                writer.writerow([class_id, f"{iou:.6f}"])
            writer.writerow(["mean", f"{miou:.6f}"])
        logging.info(f"预测mIoU指标已保存: {metrics_path}")
        return metrics_path

    def predict(self, input_dir, output_dir, progress_callback=None, stop_event=None):
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = [f for f in os.listdir(input_dir)
                       if f.lower().endswith(extensions)]
        total = len(image_files)
        if total == 0:
            logging.warning("输入文件夹中没有找到图像文件")
            return

        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"开始分割预测，共 {total} 张图像")
        color_map = self._default_color_map(self.num_classes)
        predictions = []
        predicted_classes = set()
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        label_dir = self._infer_label_dir(input_dir)
        has_label = bool(label_dir and os.path.isdir(label_dir))
        if has_label:
            logging.info(f"自动找到标签目录用于mIoU统计: {label_dir}")

        with torch.no_grad():
            for idx, filename in enumerate(image_files):
                if stop_event and stop_event.is_set():
                    logging.info("分割预测已被用户取消")
                    return None

                img_path = os.path.join(input_dir, filename)
                try:
                    original = Image.open(img_path).convert('RGB')
                    orig_w, orig_h = original.size
                    img_tensor = self.transform(original).unsqueeze(0).to(self.device)

                    output = self.model(img_tensor)['out']
                    output = F.interpolate(output, size=(orig_h, orig_w),
                                           mode='bilinear', align_corners=False)
                    pred = output.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
                    image_classes = sorted(np.unique(pred).astype(int).tolist())
                    predicted_classes.update(image_classes)

                    mask_img = Image.fromarray(pred)
                    name, _ = os.path.splitext(filename)
                    mask_path = os.path.join(output_dir, f"{name}_pred.png")
                    color_path = os.path.join(output_dir, f"{name}_color.png")
                    mask_img.save(mask_path)
                    self._colorize_mask(pred, color_map).save(color_path)
                    predictions.append({
                        "filename": filename,
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "color_path": color_path,
                        "classes": image_classes,
                    })

                    if has_label:
                        label_path = self._find_label_path(label_dir, filename)
                        if label_path:
                            label_mask = self._load_label_mask(label_path, (orig_w, orig_h), self.num_classes)
                            self._update_confusion_matrix(confusion_matrix, pred.astype(np.int64), label_mask, self.num_classes)
                        else:
                            logging.warning(f"未找到 {filename} 对应的标签文件，跳过精度统计")
                except Exception as e:
                    logging.error(f"处理 {filename} 失败: {e}")

                if progress_callback:
                    progress_callback((idx + 1) / total * 100)

        metrics_path = None
        class_iou = None
        miou = None
        if has_label and confusion_matrix.sum() > 0:
            class_iou, miou = self._miou_from_confusion(confusion_matrix)
            logging.info(f"预测总体mIoU: {miou:.4f}")
            logging.info("预测逐类mIoU: " + ", ".join(
                [f"类别{idx}={iou:.4f}" for idx, iou in enumerate(class_iou)]
            ))
            metrics_path = self._save_prediction_miou(output_dir, class_iou, miou)
        elif label_dir:
            logging.warning("未统计预测mIoU：标签目录无有效匹配标签")
        else:
            logging.info("未自动找到可匹配的标签目录，跳过预测mIoU统计")

        logging.info("分割预测完成")
        used_color_map = {class_id: color_map[class_id] for class_id in sorted(predicted_classes) if class_id in color_map}
        return {
            "predictions": predictions,
            "color_map": used_color_map or color_map,
            "metrics_path": metrics_path,
            "miou": miou,
            "class_iou": class_iou,
        }


class SegmentationPredictTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.recent_files = RecentFiles()
        self.setup_ui()
        self.bind('<FocusIn>', lambda e: self.focus_set())

    def setup_ui(self):
        frame_left = ttk.Frame(self)
        frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        frame_right = ttk.Frame(self)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 模型设置
        model_frame = tk.LabelFrame(frame_left, text="模型设置")
        model_frame.pack(fill=tk.X, pady=5)

        self.model_path = tk.StringVar()
        ttk.Label(model_frame, text="模型文件：").pack(anchor=tk.W, padx=5)
        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Entry(model_path_frame, textvariable=self.model_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="浏览",
                   command=self.browse_model).pack(side=tk.RIGHT, padx=5)

        # 参数设置
        param_frame = tk.LabelFrame(frame_left, text="参数设置")
        param_frame.pack(fill=tk.X, pady=5)

        ttk.Label(param_frame, text="模型架构:").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        self.model_arch = tk.StringVar(value="DeepLabV3")
        arch_combo = ttk.Combobox(
            param_frame,
            textvariable=self.model_arch,
            values=get_architectures(),
            state="readonly",
            width=20
        )
        arch_combo.grid(row=0, column=1, padx=5, pady=2)

        self.backbone_label = ttk.Label(param_frame, text="主干网络:")
        self.backbone_label.grid(row=1, column=0, padx=5, pady=2, sticky='e')
        self.model_backbone = tk.StringVar(value=get_default_backbone(self.model_arch.get()) or "")
        backbone_combo = ttk.Combobox(
            param_frame,
            textvariable=self.model_backbone,
            values=get_backbones(self.model_arch.get()),
            state="readonly",
            width=20
        )
        backbone_combo.grid(row=1, column=1, padx=5, pady=2)

        def _on_arch_change(_event=None):
            arch = self.model_arch.get()
            need_backbone = architecture_requires_backbone(arch)
            candidates = get_backbones(arch)
            backbone_combo.configure(values=candidates)
            if need_backbone:
                self.backbone_label.grid()
                backbone_combo.grid()
                if self.model_backbone.get() not in candidates:
                    self.model_backbone.set(candidates[0] if candidates else "")
            else:
                self.model_backbone.set("")
                self.backbone_label.grid_remove()
                backbone_combo.grid_remove()
            if hasattr(self, "arch_param_grid"):
                self._refresh_arch_param_controls()

        arch_combo.bind("<<ComboboxSelected>>", _on_arch_change)
        _on_arch_change()

        # 架构参数卡（根据架构动态渲染）
        arch_param_frame = tk.LabelFrame(param_frame, text="架构参数")
        arch_param_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
        self.arch_param_grid = ttk.Frame(arch_param_frame)
        self.arch_param_grid.pack(fill=tk.X, padx=5, pady=5)
        self.arch_param_vars = {}
        self._refresh_arch_param_controls()

        ttk.Label(param_frame, text="类别数量:").grid(row=2, column=0, padx=5, pady=2, sticky='e')
        self.num_classes = tk.StringVar(value="2")
        ttk.Entry(param_frame, textvariable=self.num_classes, width=10).grid(row=2, column=1, padx=5, pady=2)

        # 文件夹选择
        folder_frame = tk.LabelFrame(frame_left, text="处理文件夹")
        folder_frame.pack(fill=tk.X, pady=5)

        input_frame = ttk.Frame(folder_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(input_frame, text="输入文件夹：").pack(side=tk.LEFT)
        self.input_dir = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(input_frame, text="浏览",
                   command=lambda: self.browse_dir(self.input_dir)).pack(side=tk.RIGHT)

        output_frame = ttk.Frame(folder_frame)
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(output_frame, text="输出文件夹：").pack(side=tk.LEFT)
        self.output_dir = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(output_frame, text="浏览",
                   command=lambda: self.browse_dir(self.output_dir)).pack(side=tk.RIGHT)

        # 右侧：输出预览
        preview_frame = tk.LabelFrame(frame_right, text="输出预览")
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_status = tk.StringVar(value="预测完成后将在此显示首张分割结果")
        ttk.Label(preview_frame, textvariable=self.preview_status, font=("微软雅黑", 10)).pack(
            anchor=tk.W, padx=8, pady=(8, 4)
        )

        self.output_preview_label = ttk.Label(preview_frame, anchor=tk.CENTER)
        self.output_preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.preview_metrics = tk.StringVar(value="")
        ttk.Label(preview_frame, textvariable=self.preview_metrics, font=("微软雅黑", 10)).pack(
            anchor=tk.W, padx=8, pady=(0, 8)
        )

        # 开始按钮
        start_frame = ttk.Frame(frame_left)
        start_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 5))

        start_button = ttk.Button(start_frame,
                                  text="开始分割 ※",
                                  style='success.TButton',
                                  command=self.start_segmentation)
        start_button.pack(fill=tk.X, ipady=10)

        ttk.Label(start_frame,
                  text="快捷键：Ctrl + S",
                  font=("微软雅黑", 9)).pack(pady=(5, 0))

        # 局部快捷键绑定
        self.bind('<Control-s>', lambda e: self.start_segmentation())

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)
            self.recent_files.add_recent_file(path, 'models')

    def browse_dir(self, string_var):
        path = filedialog.askdirectory(title="选择文件夹")
        if path:
            string_var.set(path)

    @staticmethod
    def _compose_model_type(architecture, backbone):
        return compose_model_name(architecture, backbone)

    def _refresh_arch_param_controls(self):
        for widget in self.arch_param_grid.winfo_children():
            widget.destroy()
        self.arch_param_vars = {}
        schema = get_architecture_param_schema(self.model_arch.get())
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
                ttk.Checkbutton(self.arch_param_grid, text=label, variable=var).grid(
                    row=row, column=0, columnspan=2, padx=5, pady=2, sticky='w'
                )
            else:
                ttk.Label(self.arch_param_grid, text=f"{label}:").grid(row=row, column=0, padx=5, pady=2, sticky='e')
                var = tk.StringVar(value=str(default))
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
        return normalize_architecture_params(self.model_arch.get(), raw)

    def _update_output_preview(self, prediction_result):
        predictions = prediction_result.get("predictions", [])
        if not predictions:
            self.preview_status.set("没有可预览的预测结果")
            self.preview_metrics.set("")
            self.output_preview_label.configure(image="")
            return

        first = predictions[0]
        preview_path = first.get("color_path") or first.get("mask_path")
        image = Image.open(preview_path).convert("RGB")
        image.thumbnail((620, 620), Image.NEAREST)
        preview_photo = ImageTk.PhotoImage(image)
        self.output_preview_label.configure(image=preview_photo)
        self.output_preview_label.image = preview_photo
        self.preview_status.set(f"预览文件: {first.get('filename', os.path.basename(preview_path))}")

        if prediction_result.get("miou") is not None:
            self.preview_metrics.set(f"预测mIoU: {prediction_result['miou']:.4f}")
        else:
            self.preview_metrics.set("未自动找到匹配标签，未统计mIoU")

    def _recolor_prediction_outputs(self, prediction_result, color_map):
        for item in prediction_result.get("predictions", []):
            mask_array = np.array(Image.open(item["mask_path"]))
            SegmentationPredictor._colorize_mask(mask_array, color_map).save(item["color_path"])
        logging.info("已按用户选择的标签颜色重新生成分割可视化图")

    def _show_color_remap_dialog(self, prediction_result, output_dir):
        predictions = prediction_result.get("predictions", [])
        if not predictions:
            messagebox.showinfo("完成", "分割处理完成，但没有可预览的预测结果。")
            return

        color_map = dict(prediction_result.get("color_map", {}))
        preview_window = tk.Toplevel(self)
        preview_window.title("分割预览与标签颜色设置")
        preview_window.geometry("1080x720")
        preview_window.transient(self.winfo_toplevel())

        main_frame = ttk.Frame(preview_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        preview_frame = tk.LabelFrame(main_frame, text="多图原图 / 分割结果预览")
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        preview_canvas = tk.Canvas(preview_frame, highlightthickness=0)
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=preview_canvas.yview)
        preview_content = ttk.Frame(preview_canvas)
        preview_window_ref = preview_canvas.create_window((0, 0), window=preview_content, anchor="nw")
        preview_canvas.configure(yscrollcommand=preview_scrollbar.set)
        preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        preview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _update_preview_scroll_region(_event=None):
            preview_canvas.configure(scrollregion=preview_canvas.bbox("all"))

        def _fit_preview_content(event):
            preview_canvas.itemconfigure(preview_window_ref, width=event.width)

        preview_content.bind("<Configure>", _update_preview_scroll_region)
        preview_canvas.bind("<Configure>", _fit_preview_content)

        controls_frame = tk.LabelFrame(main_frame, text="标签颜色")
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)

        def get_prediction_classes(item):
            classes = item.get("classes")
            if classes is None:
                mask_array = np.array(Image.open(item["mask_path"]))
                classes = sorted(np.unique(mask_array).astype(int).tolist())
                item["classes"] = classes
            return set(classes)

        target_classes = set(color_map.keys())
        selected_predictions = []
        covered_classes = set()
        remaining_predictions = list(predictions)
        while remaining_predictions and not target_classes.issubset(covered_classes):
            best_item = max(
                remaining_predictions,
                key=lambda item: len(get_prediction_classes(item) - covered_classes)
            )
            selected_predictions.append(best_item)
            covered_classes.update(get_prediction_classes(best_item))
            remaining_predictions.remove(best_item)

        if not selected_predictions:
            selected_predictions = predictions[:1]
        preview_rows = []

        def refresh_preview():
            preview_window.preview_images = []
            for row in preview_rows:
                original = Image.open(row["image_path"]).convert("RGB")
                original.thumbnail((320, 240), Image.LANCZOS)
                original_photo = ImageTk.PhotoImage(original)
                row["original_label"].configure(image=original_photo)
                preview_window.preview_images.append(original_photo)

                mask_array = np.array(Image.open(row["mask_path"]))
                result = SegmentationPredictor._colorize_mask(mask_array, color_map)
                result.thumbnail((320, 240), Image.NEAREST)
                result_photo = ImageTk.PhotoImage(result)
                row["result_label"].configure(image=result_photo)
                preview_window.preview_images.append(result_photo)

        def choose_color(class_id, button):
            initial_color = "#%02x%02x%02x" % color_map[class_id]
            selected = colorchooser.askcolor(color=initial_color, title=f"选择类别{class_id}颜色", parent=preview_window)
            if selected and selected[0]:
                rgb = tuple(int(v) for v in selected[0])
                color_map[class_id] = rgb
                button.configure(text=f"类别{class_id}: #{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")
                refresh_preview()

        coverage_text = f"当前预览 {len(selected_predictions)} 张，覆盖类别: " + ", ".join(
            f"类别{class_id}" for class_id in sorted(covered_classes)
        )
        ttk.Label(preview_content, text=coverage_text, wraplength=680).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, padx=8, pady=(8, 4)
        )
        ttk.Label(preview_content, text="原图", font=("微软雅黑", 10, "bold")).grid(
            row=1, column=0, padx=8, pady=(4, 2)
        )
        ttk.Label(preview_content, text="分割结果", font=("微软雅黑", 10, "bold")).grid(
            row=1, column=1, padx=8, pady=(4, 2)
        )

        for idx, item in enumerate(selected_predictions):
            display_row = idx * 2 + 2
            filename = item.get("filename", os.path.basename(item.get("image_path", "")))
            classes = ", ".join(f"{class_id}" for class_id in sorted(get_prediction_classes(item)))
            ttk.Label(preview_content, text=f"{filename}    类别: {classes}", wraplength=680).grid(
                row=display_row, column=0, columnspan=2, sticky=tk.W, padx=8, pady=(10, 2)
            )
            original_label = ttk.Label(preview_content, anchor=tk.CENTER)
            original_label.grid(row=display_row + 1, column=0, padx=8, pady=(0, 8), sticky="n")
            result_label = ttk.Label(preview_content, anchor=tk.CENTER)
            result_label.grid(row=display_row + 1, column=1, padx=8, pady=(0, 8), sticky="n")
            preview_rows.append({
                "image_path": item.get("image_path"),
                "mask_path": item["mask_path"],
                "original_label": original_label,
                "result_label": result_label,
            })

        ttk.Label(controls_frame, text="默认使用当前标签映射；可为每类选择更醒目的颜色。", wraplength=230).pack(
            fill=tk.X, padx=8, pady=(8, 4)
        )

        for class_id in sorted(color_map.keys()):
            rgb = color_map.get(class_id, (0, 0, 0))
            button = ttk.Button(controls_frame, text=f"类别{class_id}: #{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")
            button.configure(command=lambda cid=class_id, btn=button: choose_color(cid, btn))
            button.pack(fill=tk.X, padx=8, pady=3)

        def apply_colors():
            try:
                self._recolor_prediction_outputs(prediction_result, color_map)
                self._update_output_preview(prediction_result)
                message = f"分割处理完成！\n重着色结果已保存到：{output_dir}"
                if prediction_result.get("miou") is not None:
                    message += f"\n预测mIoU: {prediction_result['miou']:.4f}"
                messagebox.showinfo("完成", message)
                preview_window.destroy()
            except Exception as e:
                messagebox.showerror("重着色失败", str(e))

        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=tk.X, padx=8, pady=12)
        ttk.Button(button_frame, text="应用颜色并保存", style='success.TButton', command=apply_colors).pack(fill=tk.X, pady=3)
        ttk.Button(button_frame, text="保持默认并关闭", command=preview_window.destroy).pack(fill=tk.X, pady=3)

        refresh_preview()

    def start_segmentation(self):
        try:
            model_path = self.model_path.get()
            input_dir = self.input_dir.get()
            output_dir = self.output_dir.get()

            if not model_path or not os.path.isfile(model_path):
                messagebox.showerror("错误", "请选择有效的模型文件！")
                return
            if not input_dir or not os.path.isdir(input_dir):
                messagebox.showerror("错误", "请输入有效的输入文件夹！")
                return
            if not output_dir:
                messagebox.showerror("错误", "请输入输出文件夹！")
                return
            os.makedirs(output_dir, exist_ok=True)

            num_classes = int(self.num_classes.get())
            if num_classes <= 0:
                raise ValueError("类别数量必须大于0")

            progress_window = tk.Toplevel(self)
            progress_window.title("处理进度")
            progress_window.geometry("400x200")
            progress_window.transient(self)
            progress_window.grab_set()

            info_frame = ttk.Frame(progress_window)
            info_frame.pack(fill=tk.X, padx=10, pady=5)

            status_label = ttk.Label(info_frame, text="正在处理...", font=("微软雅黑", 12))
            status_label.pack(pady=10)

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(info_frame, variable=progress_var, maximum=100, length=300)
            progress_bar.pack(pady=10)

            info_label = ttk.Label(info_frame, text="", font=("微软雅黑", 10))
            info_label.pack(pady=5)

            stop_event = threading.Event()

            def update_progress(progress):
                if stop_event.is_set() or not progress_window.winfo_exists():
                    return
                progress_var.set(progress)
                info_label.config(text=f"已处理: {int(progress)}%")
                if progress >= 100:
                    status_label.config(text="处理完成")
                    cancel_button.config(text="关闭")

            cancel_button = ttk.Button(progress_window, text="取消",
                                       style='danger.TButton',
                                       command=lambda: [stop_event.set(), progress_window.destroy()])
            cancel_button.pack(pady=10)

            def process_thread():
                try:
                    arch = self.model_arch.get()
                    backbone = self.model_backbone.get() if architecture_requires_backbone(arch) else None
                    model_type = self._compose_model_type(arch, backbone)
                    predictor = SegmentationPredictor(
                        model_path, num_classes, model_type, architecture_params=self._collect_architecture_params()
                    )
                    result = predictor.predict(
                        input_dir,
                        output_dir,
                        progress_callback=lambda p: self.after(0, lambda: update_progress(p)),
                        stop_event=stop_event
                    )
                    if not stop_event.is_set() and result:
                        self.after(0, lambda r=result: [
                            self._update_output_preview(r),
                            self._show_color_remap_dialog(r, output_dir)
                        ])
                except Exception as e:
                    # 修复闭包变量访问问题：在lambda表达式中正确捕获异常变量
                    error_msg = str(e)
                    if not stop_event.is_set():
                        self.after(0, lambda msg=error_msg: messagebox.showerror("错误", msg))
                finally:
                    if progress_window.winfo_exists():
                        self.after(0, progress_window.destroy)

            threading.Thread(target=process_thread, daemon=True).start()

        except ValueError as e:
            # 修复闭包变量访问问题：在lambda表达式中正确捕获异常变量
            error_msg = f"类别数量格式错误：{str(e)}"
            messagebox.showerror("输入错误", error_msg)
        except Exception as e:
            # 修复闭包变量访问问题：在lambda表达式中正确捕获异常变量
            error_msg = str(e)
            messagebox.showerror("错误", error_msg)
