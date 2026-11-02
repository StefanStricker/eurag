"""Cross-encoder reranking on the GPU (fastembed-gpu, onnxruntime CUDA).

On the CPU (Ryzen 5800X) bge-reranker-v2-m3 needs ~11-14 s for 30 chunks, on the GPU 0.4 s. It shares the
8 GB with the LLM, so the fp16 build is used and onnxruntime's memory pool is capped (`rerank_gpu_mem_gb`).
"""

from __future__ import annotations

import math

import onnxruntime as ort
from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder

# Models fastembed doesn't list: Hugging Face repo -> ONNX file, size in GB
CUSTOM_MODELS = {"onnx-community/bge-reranker-v2-m3-ONNX": ("onnx/model_fp16.onnx", 1.14)}
# One pair at a time: as fast as 2-4 on the GPU, and the attention buffers of the longest chunks (~2900 chars)
# stay within the capped pool (batch 4 runs out at 1.5 GB)
BATCH_SIZE = 1


def register_custom_models() -> None:
    for repo, (model_file, size) in CUSTOM_MODELS.items():
        TextCrossEncoder.add_custom_model(repo, sources=ModelSource(hf=repo), model_file=model_file, size_in_gb=size)


class Reranker:
    def __init__(self, model: str, gpu_mem_gb: float):
        ort.preload_dlls()  # CUDA/cuDNN come from the nvidia-* wheels, not the system
        register_custom_models()
        cuda = {
            "gpu_mem_limit": int(gpu_mem_gb * 2**30),
            "arena_extend_strategy": "kSameAsRequested",  # grow by what is needed, not in powers of two
        }
        self._model = TextCrossEncoder(model, providers=[("CUDAExecutionProvider", cuda), "CPUExecutionProvider"])

    def scores(self, question: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the question in (0, 1): the cross-encoder logit through a sigmoid,
        so downstream products (recital expansion) and the trace stay on a positive scale."""
        return [1 / (1 + math.exp(-s)) for s in self._model.rerank(question, texts, batch_size=BATCH_SIZE)]
