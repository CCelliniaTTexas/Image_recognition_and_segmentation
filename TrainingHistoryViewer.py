import tkinter as tk
from tkinter import messagebox

import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from TrainingHistory import TrainingHistoryStore


METRIC_SPECS = {
    "loss": ("损失", "loss"),
    "miou": ("mIoU", "miou"),
    "oa": ("OA", "oa"),
    "f1_score": ("F1-score", "f1_score"),
    "recall": ("召回率", "recall"),
}


class TrainingHistoryViewer(tk.Toplevel):
    """Training history overview and detail viewer."""

    _instance = None

    @classmethod
    def show(cls, parent):
        if cls._instance is not None and cls._instance.winfo_exists():
            cls._instance.lift()
            cls._instance.focus_force()
            return cls._instance
        cls._instance = cls(parent)
        return cls._instance

    def __init__(self, parent):
        super().__init__(parent)
        self.store = TrainingHistoryStore()
        self.title("历史训练数据")
        self.geometry("1100x700")
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._setup_ui()
        self.refresh_runs()

    def _setup_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(toolbar, text="刷新", command=self.refresh_runs).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="查看详情", command=self.open_selected_run).pack(side=tk.LEFT, padx=4)

        columns = ("started_at", "model", "attention", "status", "best_miou", "duration", "params")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=18)
        headings = {
            "started_at": "训练时间",
            "model": "架构模型",
            "attention": "注意力",
            "status": "状态",
            "best_miou": "最佳mIoU",
            "duration": "总耗时",
            "params": "参数量",
        }
        widths = {
            "started_at": 150,
            "model": 210,
            "attention": 150,
            "status": 90,
            "best_miou": 90,
            "duration": 100,
            "params": 120,
        }
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor=tk.W)

        y_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=y_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 8))
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=(0, 8))
        self.tree.bind("<Double-1>", lambda _event: self.open_selected_run())

    def refresh_runs(self):
        self.tree.delete(*self.tree.get_children())
        self.runs_by_id = {}
        runs = self.store.list_runs()
        if not runs:
            return

        for run in runs:
            run_id = run.get("run_id")
            params = run.get("parameters", {})
            results = run.get("results", {})
            best = results.get("best_metrics", {})
            best_miou = best.get("miou", results.get("best_miou"))
            param_count = results.get("parameters_count")
            values = (
                run.get("started_at", ""),
                params.get("model_name", ""),
                params.get("attention_label", params.get("attention_type", "")),
                self._status_label(run.get("status")),
                self._format_float(best_miou),
                results.get("training_time", ""),
                f"{param_count:,}" if isinstance(param_count, int) else "",
            )
            self.runs_by_id[run_id] = run
            self.tree.insert("", tk.END, iid=run_id, values=values)

    def open_selected_run(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请选择一条训练记录")
            return
        run_id = selected[0]
        TrainingHistoryDetailWindow(self, self.store, run_id)

    def _on_close(self):
        TrainingHistoryViewer._instance = None
        self.destroy()

    @staticmethod
    def _status_label(status):
        return {
            "running": "训练中",
            "completed": "已完成",
            "stopped": "已停止",
            "failed": "失败",
            "finished_without_summary": "已结束",
        }.get(status, status or "")

    @staticmethod
    def _format_float(value):
        try:
            return f"{float(value):.6f}"
        except Exception:
            return ""


class TrainingHistoryDetailWindow(tk.Toplevel):
    def __init__(self, parent, store, run_id):
        super().__init__(parent)
        self.store = store
        self.run_id = run_id
        self.run = store.get_run(run_id) or {}
        self.metrics = store.load_metrics(run_id)
        self.title("训练详情")
        self.geometry("1150x780")
        self.transient(parent)
        self._setup_ui()

    def _setup_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        params_frame = ttk.Frame(notebook)
        results_frame = ttk.Frame(notebook)
        curves_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="训练参数")
        notebook.add(results_frame, text="训练结果")
        notebook.add(curves_frame, text="训练曲线")

        self._build_key_value_page(params_frame, self._parameter_rows())
        self._build_key_value_page(results_frame, self._result_rows())
        self._build_curves_page(curves_frame)

    def _build_key_value_page(self, parent, rows):
        text = tk.Text(parent, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        for key, value in rows:
            text.insert(tk.END, f"{key}: {value}\n")
        text.configure(state="disabled")

    def _parameter_rows(self):
        params = self.run.get("parameters", {})
        runtime = self.run.get("runtime", {})
        rows = [
            ("训练ID", self.run.get("run_id", "")),
            ("开始时间", self.run.get("started_at", "")),
            ("结束时间", self.run.get("ended_at", "")),
            ("状态", TrainingHistoryViewer._status_label(self.run.get("status"))),
            ("数据集路径", params.get("dataset_path", "")),
            ("权重保存路径", params.get("save_path", "")),
            ("统计输出路径", params.get("metrics_output_path", "")),
            ("架构模型", params.get("model_name", "")),
            ("主干网络", params.get("backbone", "")),
            ("注意力机制", params.get("attention_label", "")),
            ("预训练权重", "是" if params.get("use_pretrained") else "否"),
            ("类别数量", params.get("actual_num_classes", params.get("num_classes", ""))),
            ("批次大小", params.get("batch_size", "")),
            ("学习率", params.get("learning_rate", "")),
            ("训练轮数", params.get("epochs", "")),
            ("图像尺寸", params.get("img_size", "")),
            ("L2权重衰减", params.get("weight_decay", "")),
            ("解码器随机失活", params.get("decoder_dropout", "")),
            ("强数据增强", "是" if params.get("strong_augment") else "否"),
            ("架构参数", params.get("architecture_params", {})),
            ("实际设备", runtime.get("device", "")),
            ("AMP", "开启" if runtime.get("amp_enabled") else "关闭"),
            ("梯度累积步数", runtime.get("gradient_accumulation_steps", "")),
            ("工作线程数", runtime.get("num_workers", "")),
        ]
        return rows

    def _result_rows(self):
        results = self.run.get("results", {})
        artifacts = self.run.get("artifacts", {})
        best = results.get("best_metrics", {})
        final = results.get("final_metrics", {})
        rows = [
            ("训练总时长", results.get("training_time", "")),
            ("训练总秒数", results.get("training_seconds", "")),
            ("最佳mIoU", TrainingHistoryViewer._format_float(results.get("best_miou"))),
            ("最佳Epoch", best.get("epoch", "")),
            ("最佳验证Loss", TrainingHistoryViewer._format_float(best.get("loss"))),
            ("最佳验证OA", TrainingHistoryViewer._format_float(best.get("oa"))),
            ("最佳验证F1-score", TrainingHistoryViewer._format_float(best.get("f1_score"))),
            ("最佳验证Recall", TrainingHistoryViewer._format_float(best.get("recall"))),
            ("最终验证Loss", TrainingHistoryViewer._format_float(final.get("loss"))),
            ("最终验证mIoU", TrainingHistoryViewer._format_float(final.get("miou"))),
            ("最终验证OA", TrainingHistoryViewer._format_float(final.get("oa"))),
            ("最终验证F1-score", TrainingHistoryViewer._format_float(final.get("f1_score"))),
            ("最终验证Recall", TrainingHistoryViewer._format_float(final.get("recall"))),
            ("参数量", f"{results.get('parameters_count', 0):,}" if results.get("parameters_count") else ""),
            ("FLOPs", results.get("flops", "")),
            ("单张推理时间", results.get("inference_time", "")),
            ("峰值显存", f"{results.get('peak_memory_mb', 0):.2f} MB" if "peak_memory_mb" in results else ""),
            ("最佳模型路径", artifacts.get("weight_path", "")),
            ("训练指标Excel", artifacts.get("excel_path", "")),
            ("类别IoU表格", artifacts.get("class_iou_path", "")),
            ("混淆矩阵热力图", artifacts.get("confusion_matrix_path", "")),
            ("训练摘要", artifacts.get("summary_path", "")),
        ]
        return rows

    def _build_curves_page(self, parent):
        if not self.metrics:
            ttk.Label(parent, text="当前训练记录没有可展示的曲线数据").pack(padx=8, pady=8)
            return

        figure = Figure(figsize=(10, 7), dpi=100)
        self.axis_metric_map = {}
        axes = [figure.add_subplot(3, 2, idx + 1) for idx in range(6)]
        axes[-1].axis("off")

        for ax, (metric_key, (title, field)) in zip(axes, METRIC_SPECS.items()):
            series = self._metric_series(field)
            if series["train_x"]:
                ax.plot(series["train_x"], series["train_y"], "r.-", label="训练", markersize=3)
            if series["valid_x"]:
                ax.plot(series["valid_x"], series["valid_y"], "b.-", label="验证", markersize=3)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="best", fontsize=8)
            self.axis_metric_map[ax] = metric_key

        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        canvas.mpl_connect("button_press_event", self._on_curve_click)
        canvas.draw()

    def _on_curve_click(self, event):
        metric_key = self.axis_metric_map.get(event.inaxes)
        if metric_key:
            self._open_metric_window(metric_key)

    def _open_metric_window(self, metric_key):
        title, field = METRIC_SPECS[metric_key]
        series = self._metric_series(field)
        window = tk.Toplevel(self)
        window.title(f"{title} 曲线详情")
        window.geometry("900x600")
        figure = Figure(figsize=(8, 5), dpi=100)
        ax = figure.add_subplot(111)
        if series["train_x"]:
            ax.plot(series["train_x"], series["train_y"], "r.-", label="训练", markersize=4)
        if series["valid_x"]:
            ax.plot(series["valid_x"], series["valid_y"], "b.-", label="验证", markersize=5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def _metric_series(self, field):
        series = {"train_x": [], "train_y": [], "valid_x": [], "valid_y": []}
        for record in self.metrics:
            value = record.get(field)
            if value is None:
                continue
            phase = record.get("phase")
            if phase == "train":
                total = max(1, int(record.get("total_batches") or 1))
                batch = record.get("batch")
                try:
                    x_value = float(record.get("epoch", 0)) - 1 + float(batch) / total
                except Exception:
                    x_value = float(record.get("epoch", 0))
                series["train_x"].append(x_value)
                series["train_y"].append(float(value))
            elif phase == "valid":
                series["valid_x"].append(float(record.get("epoch", 0)))
                series["valid_y"].append(float(value))
        return series
