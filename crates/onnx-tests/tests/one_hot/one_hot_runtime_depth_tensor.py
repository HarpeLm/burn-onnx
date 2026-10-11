#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: one_hot_runtime_depth_tensor.onnx
#
# OneHot whose `depth` arrives as a 1-element int64 tensor graph input (runtime depth).

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [3])
    depth = helper.make_tensor_value_info("depth", TensorProto.INT64, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, None])
    values = numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name="values")

    node = helper.make_node("OneHot", ["indices", "depth", "values"], ["y"], axis=-1)
    graph = helper.make_graph(
        [node], "one_hot_runtime_depth_tensor_graph", [indices, depth], [y], [values]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "one_hot_runtime_depth_tensor.onnx")

    # The ONNX spec allows a scalar or a 1-element depth, but onnx's
    # ReferenceEvaluator only accepts a scalar, so feed it the same value as one.
    ref = ReferenceEvaluator(model)
    (out,) = ref.run(
        None,
        {"indices": np.array([0, 2, 1], dtype=np.int64), "depth": np.array(3, dtype=np.int64)},
    )
    print(out.tolist())


if __name__ == "__main__":
    main()
