import os
import logging
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance


SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None, img_size=256, enable_augmentation=False,
                 value_to_class=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.img_size = img_size
        self.enable_augmentation = enable_augmentation and phase == 'train'

        self.images_dir = os.path.join(root_dir, phase, 'images')
        self.masks_dir = os.path.join(root_dir, phase, 'masks')

        self.images = [f for f in os.listdir(self.images_dir)
                       if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)]

        self.value_to_class = value_to_class or self.build_class_mapping(root_dir, phases=(phase,))
        self.num_classes = len(self.value_to_class)
        logging.info(f"[{phase}] 掩码像素值→类别映射: {self.value_to_class}  (共 {self.num_classes} 类)")

    @classmethod
    def build_class_mapping(cls, root_dir, phases=('train', 'valid')):
        """Scan all masks in the selected phases and map discovered pixel values to 0..N-1."""
        unique_values = set()
        for phase in phases:
            images_dir = os.path.join(root_dir, phase, 'images')
            masks_dir = os.path.join(root_dir, phase, 'masks')
            if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
                continue

            image_names = [
                f for f in os.listdir(images_dir)
                if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)
            ]
            for img_name in image_names:
                name_without_ext = os.path.splitext(img_name)[0]
                mask_path = os.path.join(masks_dir, f"{name_without_ext}_lab.png")
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path))
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    unique_values.update(int(v) for v in np.unique(mask).tolist())

        sorted_values = sorted(unique_values)
        return {v: i for i, v in enumerate(sorted_values)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        name_without_ext = os.path.splitext(img_name)[0]
        mask_name = f"{name_without_ext}_lab.png"

        image = Image.open(os.path.join(self.images_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.masks_dir, mask_name))

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.enable_augmentation:
            image, mask = self._apply_train_augmentations(image, mask)

        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        remapped = np.zeros_like(mask_np)
        for pixel_val, class_idx in self.value_to_class.items():
            remapped[mask_np == pixel_val] = class_idx

        if self.transform:
            image = self.transform(image)

        mask_tensor = torch.from_numpy(remapped).long()
        return image, mask_tensor

    def _apply_train_augmentations(self, image, mask):
        """对训练集执行同步增强，提升泛化能力。"""
        # 随机裁剪后缩放回原尺寸
        if random.random() < 0.8:
            crop_ratio = random.uniform(0.75, 1.0)
            crop_size = max(64, int(self.img_size * crop_ratio))
            max_offset = self.img_size - crop_size
            top = random.randint(0, max_offset) if max_offset > 0 else 0
            left = random.randint(0, max_offset) if max_offset > 0 else 0
            image = image.crop((left, top, left + crop_size, top + crop_size))
            mask = mask.crop((left, top, left + crop_size, top + crop_size))
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # 几何增强（图像与掩码保持一致）
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.2:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # 颜色扰动（仅图像）
        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))

        # 高斯噪声（仅图像）
        if random.random() < 0.3:
            image_np = np.array(image).astype(np.float32)
            std = random.uniform(5.0, 15.0)
            noise = np.random.normal(0, std, image_np.shape).astype(np.float32)
            image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_np)

        return image, mask
