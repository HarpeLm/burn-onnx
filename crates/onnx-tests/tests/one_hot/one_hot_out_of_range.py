#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: one_hot_out_of_range.onnx
#
# OneHot with a static depth of 3 and indices outside [-depth, depth - 1].
# The spec says those give a row filled with off_value. Expected values come from
# ONNX Runtime: onnx's ReferenceEvaluator wraps out-of-range indices with np.mod
# (7 becomes class 1), which doesn't follow the spec, so it isn't used here.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    depth = numpy_helper.from_array(np.array(3, dtype=np.int64), name="depth")
    values = numpy_helper.from_array(np.array([0.0, 5.0], dtype=np.float32), name="values")
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [6, 3])

    node = helper.make_node("OneHot", ["indices", "depth", "values"], ["y"], axis=-1)
    graph = helper.make_graph(
        [node], "one_hot_out_of_range_graph", [indices], [y], [depth, values]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "one_hot_out_of_range.onnx")

    feeds = {"indices": np.array([0, 7, -4, 2, -3, -1], dtype=np.int64)}
    (ort_out,) = ort.InferenceSession(model.SerializeToString()).run(None, feeds)
    (ref_out,) = ReferenceEvaluator(model).run(None, feeds)
    print("ONNX Runtime (expected):", ort_out.tolist())
    print("ReferenceEvaluator (wraps, not used):", ref_out.tolist())


if __name__ == "__main__":
    main()
