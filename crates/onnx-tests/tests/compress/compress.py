#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Used to generate multiple Compress ONNX test models

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET = 18


def _model(name, input_shape, axis, output_shape):
    """Build a Compress model with the given axis attribute (None = no attribute)."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    # Rank 1 with an unknown length: the spec allows it to be shorter than the
    # axis it selects along.
    condition_tensor = helper.make_tensor_value_info("condition", TensorProto.BOOL, [None])
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )

    kwargs = {"axis": axis} if axis is not None else {}
    node = helper.make_node(
        "Compress",
        inputs=["input", "condition"],
        outputs=["output"],
        name=name,
        **kwargs,
    )

    graph = helper.make_graph(
        [node], f"{name}_graph", [input_tensor, condition_tensor], [output_tensor]
    )

    return helper.make_model(
        graph, producer_name=f"{name}_test", opset_imports=[helper.make_opsetid("", OPSET)]
    )


def build_compress_axis0_model():
    """Compress along axis 0 of a 3x2 tensor"""
    return _model("compress_axis0", [3, 2], 0, [None, 2])


def build_compress_axis1_model():
    """Compress along axis 1 of a 3x2 tensor"""
    return _model("compress_axis1", [3, 2], 1, [3, None])


def build_compress_negative_axis_model():
    """Compress along a negative axis, which must count from the back"""
    return _model("compress_negative_axis", [3, 2], -1, [3, None])


def build_compress_default_axis_model():
    """Compress without an axis flattens the input first"""
    return _model("compress_default_axis", [3, 2], None, [None])


def build_compress_3d_model():
    """Compress a 3D tensor along its middle axis"""
    return _model("compress_3d", [2, 3, 4], 1, [2, None, 4])


def build_compress_int64_model():
    """Compress an Int64 tensor, checking the dtype is preserved"""
    model = _model("compress_int64", [3, 2], 0, [None, 2])
    model.graph.input[0].type.tensor_type.elem_type = TensorProto.INT64
    model.graph.output[0].type.tensor_type.elem_type = TensorProto.INT64
    return model


def build_compress_all_false_model():
    """An all-false condition selects nothing, producing an empty output"""
    return _model("compress_all_false", [3, 2], 0, [None, 2])


def validate_with_reference_evaluator():
    """Validate Compress using ONNX ReferenceEvaluator as ground truth"""
    print("\n=== ONNX ReferenceEvaluator Validation ===")

    input_3x2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    test_cases = [
        {
            "name": "axis 0",
            "model_func": build_compress_axis0_model,
            "input": input_3x2,
            "condition": np.array([False, True, True]),
        },
        {
            "name": "axis 1",
            "model_func": build_compress_axis1_model,
            "input": input_3x2,
            "condition": np.array([False, True]),
        },
        {
            "name": "negative axis",
            "model_func": build_compress_negative_axis_model,
            "input": input_3x2,
            "condition": np.array([False, True]),
        },
        {
            "name": "default axis (flatten)",
            "model_func": build_compress_default_axis_model,
            "input": input_3x2,
            "condition": np.array([False, True, False, False, True]),
        },
        {
            "name": "3D middle axis",
            "model_func": build_compress_3d_model,
            "input": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            "condition": np.array([True, False, True]),
        },
        {
            "name": "int64",
            "model_func": build_compress_int64_model,
            "input": input_3x2.astype(np.int64),
            "condition": np.array([False, True, True]),
        },
        {
            "name": "all false (empty output)",
            "model_func": build_compress_all_false_model,
            "input": input_3x2,
            "condition": np.array([False, False, False]),
        },
    ]

    for test_case in test_cases:
        model = test_case["model_func"]()
        evaluator = ReferenceEvaluator(model)
        result = evaluator.run(
            None,
            {
                "input": test_case["input"],
                "condition": test_case["condition"],
            },
        )

        output = result[0]
        print(f"\n{test_case['name']}:")
        print(f"  input shape: {test_case['input'].shape}")
        print(f"  condition:   {test_case['condition']}")
        print(f"  output shape: {output.shape} dtype={output.dtype}")
        print(f"  output:\n{output}")


def main():
    models = [
        ("compress_axis0.onnx", build_compress_axis0_model()),
        ("compress_axis1.onnx", build_compress_axis1_model()),
        ("compress_negative_axis.onnx", build_compress_negative_axis_model()),
        ("compress_default_axis.onnx", build_compress_default_axis_model()),
        ("compress_3d.onnx", build_compress_3d_model()),
        ("compress_int64.onnx", build_compress_int64_model()),
        ("compress_all_false.onnx", build_compress_all_false_model()),
    ]

    for filename, model in models:
        onnx.save(model, filename)
        print(f"Generated {filename}")

    validate_with_reference_evaluator()


if __name__ == "__main__":
    main()