import ttkbootstrap as ttk
import tkinter as tk

class HelpDialog(tk.Toplevel):
    _instance = None

    @classmethod
    def show(cls, parent):
        """单例显示帮助窗口：已存在则置顶，不重复创建。"""
        if cls._instance is not None and cls._instance.winfo_exists():
            dialog = cls._instance
            if dialog.state() == 'withdrawn':
                dialog.deiconify()
            dialog.lift()
            dialog.focus_force()
            dialog.attributes('-topmost', True)
            dialog.after(120, lambda: dialog.attributes('-topmost', False))
            return dialog

        cls._instance = cls(parent)
        return cls._instance

    def __init__(self, parent):
        super().__init__(parent)
        HelpDialog._instance = self
        self.title("使用说明")
        self.geometry("900x650")
        self.minsize(700, 500)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief='flat', sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tree_frame = ttk.Frame(paned, width=200)
        tree_frame.pack_propagate(False)
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set, show='tree')
        self.tree.pack(fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.tree.yview)

        content_frame = ttk.Frame(paned)
        text_scroll = ttk.Scrollbar(content_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text = tk.Text(content_frame, wrap=tk.WORD, yscrollcommand=text_scroll.set,
                            padx=12, pady=8, spacing1=2, spacing3=2)
        self.text.pack(fill=tk.BOTH, expand=True)
        text_scroll.config(command=self.text.yview)
        paned.add(tree_frame)
        paned.add(content_frame)

        self.setup_tree()
        self.tree.bind('<<TreeviewSelect>>', self.on_select)
        self.add_content()

    def destroy(self):
        if HelpDialog._instance is self:
            HelpDialog._instance = None
        super().destroy()

    def setup_tree(self):
        sections = {
            'about': ('关于软件', None),
            'overview': ('一、功能概述', None),
            'features': ('二、主要功能', {
                'classification': '1. 分类训练',
                'recognition': '2. 分类预测',
                'segmentation': '3. 分割训练',
                'prediction': '4. 分割预测',
                'cutting': '5. 影像裁剪'
            }),
            'settings': ('三、设置与性能', {
                'device': '1. 计算设备',
                'performance': '2. 性能参数',
                'theme': '3. 主题外观',
                'data_mgmt': '4. 数据管理'
            }),
            'shortcuts': ('四、快捷键', None),
            'notes': ('五、注意事项', None),
            'faq': ('六、常见问题', None),
            'contact': ('七、联系方式', None)
        }

        for section_id, (title, subsections) in sections.items():
            main_item = self.tree.insert('', 'end', section_id, text=title)
            if subsections:
                for sub_id, sub_title in subsections.items():
                    self.tree.insert(main_item, 'end', f'{section_id}_{sub_id}', text=sub_title)

        self.tree.item('features', open=True)
        self.tree.item('settings', open=True)

    def on_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return
        selected_id = selection[0]
        self.text.tag_remove('highlight', '1.0', tk.END)

        start_pos = self.section_positions.get(selected_id)
        if start_pos:
            self.text.see(start_pos)
            line_num = int(str(start_pos).split('.')[0])
            line_end = f"{line_num}.end"
            self.text.tag_add('highlight', start_pos, line_end)

    def add_content(self):
        self.text.tag_configure('title', font=('微软雅黑', 14, 'bold'), spacing1=6, spacing3=4)
        self.text.tag_configure('heading', font=('微软雅黑', 12, 'bold'), spacing1=10, spacing3=4)
        self.text.tag_configure('subheading', font=('微软雅黑', 11, 'bold'), spacing1=8, spacing3=2)
        self.text.tag_configure('normal', font=('微软雅黑', 10), spacing1=1, spacing3=1)
        self.text.tag_configure('code', font=('Consolas', 9), background='#f5f5f5',
                                lmargin1=30, lmargin2=30, spacing1=1, spacing3=1)
        self.text.tag_configure('highlight', background='#ddeeff')
        self.text.tag_configure('emphasis', font=('微软雅黑', 10, 'bold'))
        self.text.tag_configure('tip', font=('微软雅黑', 9), foreground='#2980b9')

        self.section_positions = {}

        # ── 关于软件 ──
        self.section_positions['about'] = self.text.index('end')
        self.text.insert('end', "图像识别分割系统  V2.0.1\n", 'title')
        self.text.insert('end', """
作者：AgNO3
版本：2.0.1（2026-5）
平台：Windows / Linux（需 NVIDIA GPU 以获得最佳性能）

本软件是一款集成化深度学习工具，涵盖图像分类、语义分割的模型训练与批量预测，
以及遥感影像的矢量裁剪制作数据集功能。所有操作均通过图形界面完成，无需编写代码。
\n""", 'normal')

        # ── 功能概述 ──
        self.section_positions['overview'] = self.text.index('end')
        self.text.insert('end', "一、功能概述\n", 'heading')
        self.text.insert('end', """
本软件包含五大核心模块：

  模型工具 → 分类训练    基于 ResNet18 训练图像分类模型
  模型工具 → 分类预测    批量识别图像并导出结果
  模型工具 → 分割训练    多架构语义分割训练（含架构参数卡）
  模型工具 → 分割预测    批量生成分割掩码，自动识别架构与类别
  模型工具 → 影像裁剪    根据 TIFF + SHP 文件裁剪生成训练数据集

软件特点：
  • 实时训练监控（损失曲线、准确率/mIoU 曲线）
  • 自动保存最佳模型与训练曲线图
  • 批量处理与进度显示，支持随时取消
  • 深色 / 浅色主题切换
  • CPU 使用率限制、GPU 显存占用限制
  • 运行日志支持自动滚动到最新（可在设置中开关）
\n""", 'normal')

        # ── 二、主要功能 ──
        self.section_positions['features'] = self.text.index('end')
        self.text.insert('end', "二、主要功能\n\n", 'heading')

        # 1. 分类训练
        self.section_positions['features_classification'] = self.text.index('end')
        self.text.insert('end', "1. 分类训练\n", 'subheading')
        self.text.insert('end', """
模型架构：ResNet18（预训练 ImageNet 权重）
  • 最后全连接层替换为：BatchNorm → Dropout(0.5) → Linear
  • 分层学习率：浅层 0.1×lr，深层 1.0×lr
  • 优化器：AdamW（权重衰减 0.01，标签平滑 0.1）
  • 学习率调度：ReduceLROnPlateau（根据验证准确率自动降低）
  • 早停策略：连续 10 轮验证准确率不提升时自动停止

数据集结构要求：
""", 'normal')
        self.text.insert('end', """  dataset/
      train/
          类别A/
              img001.jpg
              img002.jpg
          类别B/
              img003.jpg
      valid/
          类别A/
              img_val01.jpg
          类别B/
              img_val02.jpg
""", 'code')
        self.text.insert('end', """
  • train/ 和 valid/ 下的子文件夹名即为类别名
  • 支持 jpg、png、bmp、gif 格式
  • "类别数量"参数必须与子文件夹数一致

参数说明：
  • 类别数量：数据集中的类别总数
  • 批次大小：建议 16~32（显存不足时减小）
  • 学习率：建议 0.001（默认值）
  • 训练轮数：建议 10~50

操作步骤：
  1. 选择数据集路径（包含 train/ 和 valid/ 的父目录）
  2. 选择模型保存路径（.pth 文件）
  3. 设置训练参数
  4. 点击"开始训练"或按 Ctrl+T
  5. 观察右侧损失曲线和准确率曲线，等待训练完成
\n""", 'normal')

        # 2. 分类预测
        self.section_positions['features_recognition'] = self.text.index('end')
        self.text.insert('end', "2. 分类预测\n", 'subheading')
        self.text.insert('end', """
使用训练好的分类模型对图像进行批量识别。

准备文件：
  • 模型文件（.pth）：分类训练输出的模型
  • 类别文件（.json）：类别编号到名称的映射

""", 'normal')
        self.text.insert('end', """  类别文件示例 (classes.json):
  {
      "0": "猫",
      "1": "狗",
      "2": "鸟"
  }
""", 'code')
        self.text.insert('end', """
操作步骤：
  1. 选择模型文件和类别文件
  2. 添加待识别的文件夹（可添加多个）
  3. 点击"开始识别"或按 Ctrl+R
  4. 识别完成后自动弹出结果查看器

结果说明：
  • 结果查看器显示每张图片的识别类别
  • 可导出为 CSV 文件保存
\n""", 'normal')

        # 3. 分割训练
        self.section_positions['features_segmentation'] = self.text.index('end')
        self.text.insert('end', "3. 分割训练\n", 'subheading')
        self.text.insert('end', """
可选模型架构：
  • DeepLabV3
  • DeepLabV3+（论文版完整解码器）
  • FCN
  • U-Net
  • SFA-Net

模型选择规则：
  • 先选择模型架构
  • 若该架构需要主干网络（如 DeepLabV3 / DeepLabV3+ / FCN），会显示主干网络选项
  • 若该架构不需要主干（如 U-Net / SFA-Net），主干选项自动隐藏

架构参数卡：
  • 根据架构动态显示参数（如输出步长、ASPP 比率、补丁大小、编码层数等）
  • 架构参数会真实参与建模，不是仅用于展示

可选增强：
  • 交叉注意力增强（支持的架构才会生效）
  • 使用预训练权重（按架构和主干自动适配）

数据集结构要求：
""", 'normal')
        self.text.insert('end', """  dataset/
      train/
          images/
              image_0.jpg
              image_1.jpg
          masks/
              image_0_lab.png
              image_1_lab.png
      valid/
          images/
              ...
          masks/
              ...
""", 'code')
        self.text.insert('end', """
""", 'normal')
        self.text.insert('end', "  掩码文件命名规则：原图文件名中 .jpg 替换为 _lab.png\n", 'emphasis')
        self.text.insert('end', """
  • 掩码为灰度图，像素值 0 = 背景，1, 2, 3... = 各类别
  • 图像按"图像尺寸"参数统一缩放后训练
  • 评估指标：mIoU（平均交并比）

训练参数说明（常用）：
  • 类别数量：包含背景在内的总数（如 背景+目标 = 2）
  • 批次大小：建议 4~8（分割任务显存消耗较大）
  • 学习率：建议 0.001
  • 训练轮数：建议 20~100
  • L2权重衰减 / 解码器随机失活 / 强数据增强：用于提升泛化能力

操作步骤：
  1. 选择数据集路径
  2. 选择模型保存路径（.pth 文件）
  3. 选择模型架构（必要时选择主干）
  4. 设置架构参数与训练参数
  5. 点击"开始训练"
  6. 观察损失曲线和 mIoU 曲线
  7. 训练结束后自动保存曲线图到权重同目录
\n""", 'normal')

        # 4. 分割预测
        self.section_positions['features_prediction'] = self.text.index('end')
        self.text.insert('end', "4. 分割预测\n", 'subheading')
        self.text.insert('end', """
使用训练好的分割模型对图像进行批量语义分割。

""", 'normal')
        self.text.insert('end', "  自动识别功能：预测器会从模型权重中自动检测模型架构（DeepLabV3/DeepLabV3+/FCN/U-Net/SFA-Net）\n  和类别数量；若手动选择不匹配会自动修正。\n", 'tip')
        self.text.insert('end', """
操作步骤：
  1. 选择模型文件（.pth）
  2. 选择模型架构（不确定可任选，会自动识别）
  3. 若架构需要主干网络，再选择主干
  4. 按需设置架构参数
  5. 设置类别数量（不确定可临时填写，会自动修正）
  6. 选择输入文件夹（包含待分割图片）和输出文件夹
  7. 点击"开始分割"或按 Ctrl+S

输出说明：
  • 每张图片生成一个对应的掩码文件：{文件名}_pred.png
  • 掩码为灰度图，像素值对应类别编号
  • 输出掩码与原图保持相同分辨率
  • 支持 jpg、png、bmp、tif 等常见格式
\n""", 'normal')

        # 5. 影像裁剪
        self.section_positions['features_cutting'] = self.text.index('end')
        self.text.insert('end', "5. 影像裁剪\n", 'subheading')
        self.text.insert('end', """
将遥感影像（TIFF）按矢量文件（SHP）裁剪为训练用的图像-掩码对。
裁剪结果可直接用于分割训练模块。

输入文件：
  • TIFF 文件：多波段或单波段遥感影像
  • SHP 文件：包含"类别"字段的多边形矢量
  • 坐标系不一致时会自动转换

参数说明：
  • 起始编号：输出文件的序号起点（默认 0），用于多次裁剪时续接编号
  • 颜色数量：标注掩码中分配的灰度值数量（1~255）

处理流程：
  1. 读取 TIFF 影像，对浮点数据执行百分比拉伸归一化
  2. 读取 SHP 矢量，建立类别→灰度值映射
  3. 基础网格采样（256×256 像素，50% 重叠）
  4. 基于 NDVI + 红边斜率的目标区域补充采样
  5. 对每个采样窗口生成图像切片和对应的灰度掩码

输出目录结构：
""", 'normal')
        self.text.insert('end', """  输出目录/
      images/
          image_0.jpg
          image_1.jpg
          ...
      masks/
          image_0_lab.png
          image_1_lab.png
          ...
""", 'code')
        self.text.insert('end', """
  • 掩码为灰度图（mode='L'），像素值为类别 ID
  • 特殊映射："0"类 → 像素值 0（背景），"加黄"类 → 像素值 200
  • 其余类别从 1 开始按间隔 15 递增分配

快捷键：Ctrl+J 开始裁剪
\n""", 'normal')

        # ── 三、设置与性能 ──
        self.section_positions['settings'] = self.text.index('end')
        self.text.insert('end', "三、设置与性能\n\n", 'heading')

        self.section_positions['settings_device'] = self.text.index('end')
        self.text.insert('end', "1. 计算设备\n", 'subheading')
        self.text.insert('end', """
在 设置 → 性能 标签页中配置：

  • 启用 GPU 加速：勾选后所有训练和预测模块使用 GPU
    取消勾选则强制使用 CPU（即使有可用 GPU）
  • GPU 显存占用限制：滑块设置 30%~100%
    限制本程序可使用的 GPU 显存比例，避免与其他程序冲突
    例如设置 80% 表示最多使用总显存的 80%
    修改后立即生效，下次启动程序时也会自动应用

""", 'normal')
        self.text.insert('end', "  提示：如果在训练中遇到 CUDA out of memory 错误，请减小批次大小\n  或降低显存限制后重启程序。\n", 'tip')
        self.text.insert('end', "\n", 'normal')

        self.section_positions['settings_performance'] = self.text.index('end')
        self.text.insert('end', "2. 性能参数\n", 'subheading')
        self.text.insert('end', """
  • CPU 使用率限制（10%~90%）：
    控制批量识别时 CPU 占用率上限。超过限制时自动暂停处理，
    待 CPU 使用率下降后恢复。同时决定工作线程数量。

  • 工作线程数：
    设置训练和预测时数据加载的并行线程数。
    线程数越多数据读取越快，但会占用更多 CPU 和内存。
    建议值：CPU 核心数 - 1（已作为默认值）。

所有设置修改后自动保存，下次启动程序时自动加载。

补充说明：
  • 设置 → 常规 中可控制“日志自动滚动到最新消息”
  • 关闭后可手动查看历史日志，滚动到底部会自动重新开启
\n""", 'normal')

        self.section_positions['settings_theme'] = self.text.index('end')
        self.text.insert('end', "3. 主题外观\n", 'subheading')
        self.text.insert('end', """
在 设置 → 外观 标签页中切换：
  • 白昼模式（cosmo）：浅色背景，适合白天使用
  • 黑夜模式（darkly）：深色背景，减少眼部疲劳
  • 所有界面元素和训练图表自动适配主题色彩
\n""", 'normal')

        self.section_positions['settings_data_mgmt'] = self.text.index('end')
        self.text.insert('end', "4. 数据管理\n", 'subheading')
        self.text.insert('end', """
  • 导出设置：将当前配置导出为 JSON 文件备份
  • 导入设置：从 JSON 文件恢复配置
  • 清理缓存：删除程序缓存文件，释放磁盘空间
  • 清除最近文件：清空模型/数据集/文件夹的使用记录
\n""", 'normal')

        # ── 四、快捷键 ──
        self.section_positions['shortcuts'] = self.text.index('end')
        self.text.insert('end', "四、快捷键\n", 'heading')
        self.text.insert('end', """
  Ctrl + T    开始分类训练 / 分割训练
  Ctrl + R    开始分类预测
  Ctrl + S    开始分割预测
  Ctrl + J    开始影像裁剪
\n""", 'normal')

        # ── 五、注意事项 ──
        self.section_positions['notes'] = self.text.index('end')
        self.text.insert('end', "五、注意事项\n", 'heading')
        self.text.insert('end', """
运行环境：
  • Python 3.8+、PyTorch 1.9+、torchvision
  • 推荐 NVIDIA GPU（CUDA 11.0+）
  • 分割训练建议至少 6GB 显存
  • 分类训练 2GB 显存即可

数据集准备：
  • 分类数据集中每个类别至少 50 张图片
  • 分割掩码必须严格遵守命名规则（xxx_lab.png）
  • 验证集建议占数据集的 10%~20%
  • 图片格式不要混用透明通道（PNG 建议使用 RGB 模式）

训练建议：
  • 首次使用建议先用小数据集（<100 张）快速测试流程
  • 分类训练时，各类别样本数量尽量均衡
  • 分割训练批次大小不宜超过 8（显存消耗大）
  • 训练过程中不要修改数据集内的文件
  • 长时间训练建议关闭不必要的后台程序以释放显存
\n""", 'normal')

        # ── 六、常见问题 ──
        self.section_positions['faq'] = self.text.index('end')
        self.text.insert('end', "六、常见问题\n", 'heading')
        self.text.insert('end', """
Q：训练时显示 CUDA out of memory
""", 'normal')
        self.text.insert('end', "A：减小批次大小（分割建议 2~4，分类建议 8~16）。也可在设置中降低 GPU 显存限制后重启。\n", 'tip')
        self.text.insert('end', """
Q：加载分割模型时报 state_dict 不匹配
""", 'normal')
        self.text.insert('end', "A：请确认训练时与预测时的架构参数一致（例如 DeepLabV3+ 的输出步长、ViT 的补丁大小等）。\n   若不一致，可能导致权重键名不匹配。建议优先使用本软件训练得到的权重并保留默认参数。\n", 'tip')
        self.text.insert('end', """
Q：分类训练准确率始终很低
""", 'normal')
        self.text.insert('end', "A：检查类别数量参数是否与数据集子文件夹数一致。确认各类图片没有放错文件夹。可适当增大训练轮数（30+）。\n", 'tip')
        self.text.insert('end', """
Q：影像裁剪结果为空 / 掩码全黑
""", 'normal')
        self.text.insert('end', "A：通常是坐标系不匹配。在 GIS 软件中确认 TIFF 和 SHP 使用相同的空间参考。程序会自动转换，但无投影信息的文件无法处理。\n", 'tip')
        self.text.insert('end', """
Q：程序界面无响应
""", 'normal')
        self.text.insert('end', "A：训练和预测在后台线程执行，不应导致界面卡死。如果确实卡死，可能是数据加载阶段。请确认数据集路径正确，文件无损坏。\n", 'tip')
        self.text.insert('end', """
Q：点击取消按钮无效
""", 'normal')
        self.text.insert('end', "A：取消操作需要当前批次处理完成后才能响应。如果单张图片很大，请等待几秒钟。\n", 'tip')
        self.text.insert('end', """
Q：CPU 占用过高
""", 'normal')
        self.text.insert('end', "A：在设置 → 性能中降低 CPU 使用率限制，或减少工作线程数。\n", 'tip')
        self.text.insert('end', "\n", 'normal')

        # ── 七、联系方式 ──
        self.section_positions['contact'] = self.text.index('end')
        self.text.insert('end', "七、联系方式\n", 'heading')
        self.text.insert('end', """
如遇到问题或有功能建议，欢迎联系：
  邮箱：1346636108@qq.com
""", 'normal')

        self.text.configure(state='disabled')
