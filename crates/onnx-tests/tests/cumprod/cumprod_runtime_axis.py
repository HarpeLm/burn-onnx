#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "numpy",
# ]
# ///

# used to generate model: cumprod_runtime_axis.onnx
#
# CumProd (opset 26) on a float [2, 3] input with `axis` as a scalar graph input, as in the
# official tests. Four outputs cover every attribute combination: plain, reverse, exclusive,
# and exclusive + reverse. Expected values from onnx.reference.ReferenceEvaluator for a
# negative axis (-1) and axis 0.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

VARIANTS = [
    ("y", {}),
    ("y_reverse", {"reverse": 1}),
    ("y_exclusive", {"exclusive": 1}),
    ("y_exclusive_reverse", {"exclusive": 1, "reverse": 1}),
]


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    axis = helper.make_tensor_value_info("axis", TensorProto.INT64, [])
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 3]) for name, _ in VARIANTS]
    nodes = [helper.make_node("CumProd", ["x", "axis"], [name], **attrs) for name, attrs in VARIANTS]
    graph = helper.make_graph(nodes, "cumprod_runtime_axis_graph", [x, axis], outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 26)])
    model.ir_version = 11
    onnx.checker.check_model(model)
    onnx.save(model, "cumprod_runtime_axis.onnx")

    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    for axis_value in (-1, 0):
        feed = {"x": data, "axis": np.array(axis_value, dtype=np.int64)}
        results = ReferenceEvaluator(model).run(None, feed)
        print("axis", axis_value)
        for (name, _), result in zip(VARIANTS, results):
            print(" ", name, result.tolist())


if __name__ == "__main__":
    main()
