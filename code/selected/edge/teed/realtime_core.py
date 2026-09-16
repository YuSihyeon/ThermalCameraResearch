"""Fixed-size TEED inference with reused input buffers and optional CUDA Graph.

This module keeps Python 3.6 syntax. CUDA Graph requires a supported PyTorch
build; use eager CUDA on JetPack builds that do not expose that API.
"""
import time
import cv2
import numpy as np
import torch

try:
    from . import run_teed
except (ImportError, ValueError):
    import run_teed


class RealtimeCore(object):
    def __init__(self, width=320, height=240, device='cuda', half=False,
                 cuda_graph=False, checkpoint=None, threshold=.75, contrast_clip_limit=2.):
        if width < 64 or height < 64 or width % 8 or height % 8:
            raise ValueError('width/height must be multiples of 8 and at least 64')
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold must be finite in [0,1]')
        if not np.isfinite(contrast_clip_limit) or not 0 <= contrast_clip_limit <= 10:
            raise ValueError('contrast strength must be finite in [0,10]')
        self.device = run_teed.resolve_device(device)
        if self.device.type != 'cuda' and (half or cuda_graph):
            raise ValueError('FP16 and CUDA Graph require CUDA')
        self.width, self.height = int(width), int(height)
        self.half = bool(half)
        self.threshold = float(threshold)
        self.checkpoint = checkpoint or run_teed.default_checkpoint_path()
        self.model = run_teed.load_model(self.checkpoint, self.device)
        if self.half:
            self.model.half()
        self.clahe = (cv2.createCLAHE(clipLimit=contrast_clip_limit, tileGridSize=(8, 8))
                      if contrast_clip_limit > 0 else None)
        dtype = torch.float16 if half else torch.float32
        self.host_input = torch.empty((1, 3, height, width), dtype=dtype,
                                      pin_memory=self.device.type == 'cuda')
        self.host_array = self.host_input.numpy()[0]
        self.gpu_input = torch.zeros_like(self.host_input, device=self.device)
        self.graph = None
        self.graph_output = None
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = False
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = False
        if cuda_graph:
            self._capture()

    def _forward(self):
        logits = self.model(self.gpu_input, single_test=True)[-1]
        return torch.sigmoid(logits)[0, 0, :self.height, :self.width]

    def _capture(self):
        if not hasattr(torch.cuda, 'CUDAGraph'):
            raise RuntimeError('this PyTorch has no CUDA Graph; omit --cuda-graph')
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.no_grad(), torch.cuda.stream(side):
            for _ in range(8):
                self._forward()
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.graph_output = self._forward()

    def infer(self, frame):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('input must be HxWx3 uint8 BGR')
        started = time.perf_counter()
        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        model_input = resized
        if self.clahe is not None:
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            channels = cv2.split(lab)
            model_input = cv2.cvtColor(cv2.merge((self.clahe.apply(channels[0]), channels[1], channels[2])), cv2.COLOR_LAB2BGR)
        array = model_input.astype(np.float32) - run_teed.MEAN_BGR
        np.copyto(self.host_array, array.transpose(2, 0, 1), casting='unsafe')
        self.gpu_input.copy_(self.host_input, non_blocking=self.device.type == 'cuda')
        model_started = time.perf_counter()
        with torch.no_grad():
            if self.graph is not None:
                self.graph.replay()
                probability = self.graph_output
            else:
                probability = self._forward()
            # Synchronous D2H waits for all CUDA work. Returned storage is owned
            # by this call, not the graph's static output buffer.
            probability = probability.float().cpu().numpy().copy()
        model_ms = (time.perf_counter() - model_started) * 1000.
        mask = (probability >= self.threshold).astype(np.uint8) * 255
        return {'image': resized, 'probability': probability, 'mask': mask,
                'model_transfer_ms': model_ms, 'core_ms': (time.perf_counter() - started) * 1000.}

    def describe(self):
        return {'device': str(self.device), 'gpu_name': torch.cuda.get_device_name(self.device)
                if self.device.type == 'cuda' else None, 'torch': torch.__version__,
                'opencv': cv2.__version__, 'resolution': [self.width, self.height],
                'precision': 'fp16' if self.half else 'fp32', 'cuda_graph': self.graph is not None,
                'checkpoint': self.checkpoint, 'threshold': self.threshold,
                'timing_scope': 'host preprocess, H2D, completed GPU inference, D2H and threshold'}
