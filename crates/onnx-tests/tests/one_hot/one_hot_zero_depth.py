#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: one_hot_zero_depth.onnx
#
# OneHot with a static depth of 0: every index is outside [-depth, depth - 1], so
# the one-hot axis is empty and the output has shape [3, 0]. ONNX Runtime rejects
# this model ("Depth is negative"); the expected shape follows the spec, like
# onnx's ReferenceEvaluator.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    depth = numpy_helper.from_array(np.array(0, dtype=np.int64), name="depth")
    values = numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name="values")
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 0])

    node = helper.make_node("OneHot", ["indices", "depth", "values"], ["y"], axis=-1)
    graph = helper.make_graph([node], "one_hot_zero_depth_graph", [indices], [y], [depth, values])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "one_hot_zero_depth.onnx")

    feeds = {"indices": np.array([0, 1, -1], dtype=np.int64)}
    (ref_out,) = ReferenceEvaluator(model).run(None, feeds)
    print("ReferenceEvaluator shape:", ref_out.shape)


if __name__ == "__main__":
    main()
