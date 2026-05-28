import torch
import threading
import logging
import multiprocessing
import psutil
import queue
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
from PIL import Image
from threading import Event
from config import config



class ParallelProcessor:
    def __init__(self, num_workers=None):
        self.cpu_limit = config.get('cpu_limit', 50)
        if num_workers is None:
            # 根据CPU限制自动设置工作线程数
            cpu_count = multiprocessing.cpu_count()
            self.num_workers = max(1, int(cpu_count * (self.cpu_limit / 100)))
        else:
            self.num_workers = num_workers

        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.tasks = queue.Queue()
        self.results = queue.Queue()
        self.stop_event = Event()
        self.pause_event = Event()

    def monitor_cpu_usage(self):
        """监控CPU使用率"""
        while not self.stop_event.is_set():
            current_cpu = int(psutil.cpu_percent(interval=0.5))
            if current_cpu > self.cpu_limit:
                # CPU使用率过高时设置暂停标志
                self.pause_event.set()
                logging.info(f"CPU使用率 {current_cpu}% 超过限制 {self.cpu_limit}%，暂停处理")
                time.sleep(2)
            else:
                self.pause_event.clear()
            time.sleep(0.1)

    def process_batch(self, progress_callback=None):
        """并行处理任务批次"""
        monitor_thread = threading.Thread(target=self.monitor_cpu_usage)
        monitor_thread.daemon = True
        monitor_thread.start()

        futures = []
        total_tasks = self.tasks.qsize()
        completed_tasks = 0

        while not self.tasks.empty() and not self.stop_event.is_set():
            # 检查是否需要暂停
            if self.pause_event.is_set():
                time.sleep(1)
                continue

            # 限制同时执行的任务数量
            if len(futures) >= self.num_workers:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                futures = [f for f in futures if f not in done]

            task_func, args, kwargs = self.tasks.get()
            future = self.executor.submit(task_func, *args, **kwargs)
            futures.append(future)
            time.sleep(0.1)

    def stop(self):
        """停止所有任务"""
        self.stop_event.set()
        self.executor.shutdown(wait=False)

    def get_results(self):
        """获取所有结果"""
        results = []
        while not self.results.empty():
            results.append(self.results.get())
        return results


class BatchImageLoader:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_batch(self, image_paths):
        images = []
        valid_paths = []

        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img.verify()

                # 重新打开图片进行处理
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    img_tensor = self.transform(img)
                    images.append(img_tensor)
                    valid_paths.append(path)
            except Exception as e:
                logging.warning(f"加载图片 {path} 失败: {str(e)}")
                continue

        if not images:
            return None, []

        return torch.stack(images), valid_paths


class BatchRecognizer:
    def __init__(self, model, class_names, device, batch_size=32):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.batch_size = batch_size
        self.processor = ParallelProcessor()
        self.image_loader = BatchImageLoader(batch_size)
        self._stop_event = threading.Event()

    def stop(self):
        """触发中断标志"""
        self._stop_event.set()

    def process_image_batch(self, image_paths):
        """处理一批图像"""
        try:
            # 使用优化后的批处理加载器
            batch_tensor, valid_paths = self.image_loader.load_batch(image_paths)
            if batch_tensor is None:
                return []

            # 模型推理
            with torch.no_grad():
                batch_tensor = batch_tensor.to(self.device)
                outputs = self.model(batch_tensor)
                _, predicted = torch.max(outputs.data, 1)

            # 整理结果
            results = []
            for path, pred in zip(valid_paths, predicted.cpu().numpy()):
                class_name = self.class_names.get(str(pred), f"Unknown Class {pred}")
                results.append((path, class_name))

            return results

        except Exception as e:
            logging.error(f"批处理推理失败: {str(e)}")
            return []
        finally:
            # 清理 GPU 内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def recognize_all(self, image_paths, progress_callback=None):
        """识别所有图像"""
        try:
            # 将图像分成批次
            all_results = []  # 改名以避免混淆
            total_images = len(image_paths)
            processed_images = 0

            # 按批次处理图像
            for i in range(0, total_images, self.batch_size):
                if self._stop_event.is_set():
                    logging.warning("正在停止……")
                    break
                batch_paths = image_paths[i:i + self.batch_size]
                batch_results = self.process_image_batch(batch_paths)

                if batch_results:
                    all_results.extend(batch_results)

                # 更新进度
                processed_images += len(batch_paths)
                if progress_callback:
                    progress = min((processed_images / total_images) * 100, 100)
                    progress_callback(progress)

            # 确保进度达到100%
            if progress_callback:
                progress_callback(100)

            return all_results

        except Exception as e:
            logging.error(f"批量识别失败: {str(e)}")
            return []
