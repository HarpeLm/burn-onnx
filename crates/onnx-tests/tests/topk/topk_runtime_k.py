#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: topk_runtime_k.onnx
#
# TopK whose `k` arrives as a 1-element int64 graph input (runtime k).

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [5])
    k = helper.make_tensor_value_info("k", TensorProto.INT64, [1])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [None])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [None])

    node = helper.make_node("TopK", ["x", "k"], ["values", "indices"], axis=0)
    graph = helper.make_graph([node], "topk_runtime_k_graph", [x, k], [values, indices])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "topk_runtime_k.onnx")

    ref = ReferenceEvaluator(model)
    data = np.array([1.0, 5.0, 3.0, 4.0, 2.0], dtype=np.float32)
    out_values, out_indices = ref.run(None, {"x": data, "k": np.array([2], dtype=np.int64)})
    print(out_values.tolist(), out_indices.tolist())


if __name__ == "__main__":
    main()
