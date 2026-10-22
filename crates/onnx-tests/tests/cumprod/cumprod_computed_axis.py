#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "numpy",
# ]
# ///

# used to generate model: cumprod_computed_axis.onnx
#
# CumProd (opset 26, reverse) whose axis is computed in the graph: ArgMax over a rank-1
# input with keepdims=0 gives a scalar that lives on the device. Expected values from
# onnx.reference.ReferenceEvaluator, for an ArgMax of 1 and of 0.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    selector = helper.make_tensor_value_info("selector", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    nodes = [
        helper.make_node("ArgMax", ["selector"], ["axis"], keepdims=0),
        helper.make_node("CumProd", ["x", "axis"], ["y"], reverse=1),
    ]
    graph = helper.make_graph(nodes, "cumprod_computed_axis_graph", [x, selector], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 26)])
    model.ir_version = 11
    onnx.checker.check_model(model)
    onnx.save(model, "cumprod_computed_axis.onnx")

    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    for selector_data in ([0.0, 5.0], [5.0, 0.0]):
        feed = {"x": data, "selector": np.array(selector_data, dtype=np.float32)}
        (result,) = ReferenceEvaluator(model).run(None, feed)
        print("selector", selector_data, "->", result.tolist())


if __name__ == "__main__":
    main()
