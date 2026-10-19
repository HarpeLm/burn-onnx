#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: resize_chained_unknown_shape.onnx
#
# GridSample with an identity grid (its output carries no static shape in onnx-ir), then a
# stretch Resize to [1, 1, 4, 2] with constant sizes, then keep_aspect_ratio_policy
# not_larger with sizes [6, 6] on axes [2, 3]. The second Resize needs the spatial sizes
# the first one sets (scale min(6/4, 6/2) = 1.5 -> 6x3), although the shape before it is
# unknown. asymmetric/floor matches burn's nearest interpolation. Expected values from
# ONNX Runtime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def identity_grid(height, width):
    # Pixel centers in normalized coordinates (align_corners=0)
    ys = (2 * np.arange(height) + 1) / height - 1
    xs = (2 * np.arange(width) + 1) / width - 1
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)
    return grid[np.newaxis].astype(np.float32)


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 6, 3])
    initializers = [
        numpy_helper.from_array(identity_grid(2, 4), name="grid"),
        numpy_helper.from_array(np.array([1, 1, 4, 2], dtype=np.int64), name="stretch_sizes"),
        numpy_helper.from_array(np.array([6, 6], dtype=np.int64), name="aspect_sizes"),
    ]
    nearest = dict(mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    nodes = [
        helper.make_node("GridSample", ["x", "grid"], ["sampled"], align_corners=0),
        helper.make_node("Resize", ["sampled", "", "", "stretch_sizes"], ["stretched"], **nearest),
        helper.make_node(
            "Resize", ["stretched", "", "", "aspect_sizes"], ["y"],
            axes=[2, 3], keep_aspect_ratio_policy="not_larger", **nearest,
        ),
    ]
    graph = helper.make_graph(nodes, "resize_chained_unknown_shape_graph", [x], [y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "resize_chained_unknown_shape.onnx")

    data = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)
    (output,) = ort.InferenceSession(model.SerializeToString()).run(None, {"x": data})
    print("y", output.shape, output.tolist())


if __name__ == "__main__":
    main()
