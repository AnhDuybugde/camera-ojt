"""Verify that PyTorch and ONNX Runtime can execute on the NVIDIA GPU."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    try:
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA")

        # ONNX Runtime's Windows wheel loads CUDA/cuDNN DLLs shipped with the
        # matching PyTorch wheel. Register that directory before importing ORT.
        if os.name == "nt":
            torch_lib = Path(torch.__file__).resolve().parent / "lib"
            if torch_lib.is_dir():
                os.add_dll_directory(str(torch_lib))

        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper

        node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
        graph = helper.make_graph(
            [node],
            "cuda_smoke_test",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10

        session = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        active_providers = session.get_providers()
        if not active_providers or active_providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"ONNX Runtime fell back to {active_providers}")

        expected = np.array([[1.0, 2.0]], dtype=np.float32)
        actual = session.run(None, {"input": expected})[0]
        if not np.array_equal(actual, expected):
            raise RuntimeError("ONNX Runtime CUDA inference returned an invalid result")

        print(f"PyTorch {torch.__version__}: {torch.cuda.get_device_name(0)}")
        print(f"ONNX Runtime {ort.__version__}: {active_providers[0]}")
        return 0
    except Exception as exc:
        print(f"GPU runtime check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
