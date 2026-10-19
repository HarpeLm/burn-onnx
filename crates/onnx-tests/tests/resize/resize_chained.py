#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: resize_chained.onnx
#
# A stretch Resize from 2x4 to 4x2, then two Resizes that read the static shape of its
# output: keep_aspect_ratio_policy not_larger with sizes [6, 6] on axes [2, 3]
# (scale min(6/4, 6/2) = 1.5 -> 6x3) and sizes [5] on axes [3] (the unlisted height keeps
# its size, 4 -> 4x5). Using the input shape 2x4 instead would give 3x6 and 2x5.
# asymmetric/floor matches burn's nearest interpolation. Expected values from ONNX Runtime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 4])
    y_not_larger = helper.make_tensor_value_info("y_not_larger", TensorProto.FLOAT, [1, 1, 6, 3])
    y_axes = helper.make_tensor_value_info("y_axes", TensorProto.FLOAT, [1, 1, 4, 5])
    initializers = [
        numpy_helper.from_array(np.array([1, 1, 4, 2], dtype=np.int64), name="stretch_sizes"),
        numpy_helper.from_array(np.array([6, 6], dtype=np.int64), name="aspect_sizes"),
        numpy_helper.from_array(np.array([5], dtype=np.int64), name="width_sizes"),
    ]
    nearest = dict(mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    nodes = [
        helper.make_node("Resize", ["x", "", "", "stretch_sizes"], ["stretched"], **nearest),
        helper.make_node(
            "Resize", ["stretched", "", "", "aspect_sizes"], ["y_not_larger"],
            axes=[2, 3], keep_aspect_ratio_policy="not_larger", **nearest,
        ),
        helper.make_node(
            "Resize", ["stretched", "", "", "width_sizes"], ["y_axes"], axes=[3], **nearest
        ),
    ]
    graph = helper.make_graph(nodes, "resize_chained_graph", [x], [y_not_larger, y_axes], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "resize_chained.onnx")

    data = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)
    not_larger, axes = ort.InferenceSession(model.SerializeToString()).run(None, {"x": data})
    print("y_not_larger", not_larger.shape, not_larger.tolist())
    print("y_axes", axes.shape, axes.tolist())


if __name__ == "__main__":
    main()
