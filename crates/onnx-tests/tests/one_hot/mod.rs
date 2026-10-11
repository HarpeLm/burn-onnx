use crate::include_models;
include_models!(
    one_hot,
    one_hot_axis0,
    one_hot_float_values,
    one_hot_2d,
    one_hot_runtime_depth,
    one_hot_runtime_depth_tensor
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, Int, Tensor, TensorData};

    #[test]
    fn one_hot() {
        let device = Default::default();
        let model = one_hot::Model::new(&device);
        // ONNX OneHot's `indices` input is int64 per spec; pin the dtype so the
        // generated model receives an I64 tensor and the output carries the
        // ONNX-specified dtype. The assertion uses strict `assert_eq` against
        // an I64 TensorData literal (not `assert_approx_eq`) so it enforces the
        // output dtype, not just the element values. This is the only test that
        // covers the Int->Int one_hot_fill branch where burn-onnx must emit a
        // `.cast(DType::I64)` on top of `one_hot_fill` (which otherwise returns
        // the backend default int).
        let input: Tensor<1, Int> =
            Tensor::from_data(TensorData::from([1i64, 0, 2]), (&device, DType::I64));
        let output: Tensor<2, Int> = model.forward(input);
        let expected = TensorData::from([[0i64, 1, 0], [1, 0, 0], [0, 0, 1]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn one_hot_axis0() {
        // axis=0: one-hot dim inserted at front, depth=4, indices [0, 2, 3]
        // output shape [4, 3]
        let device = Default::default();
        let model = one_hot_axis0::Model::new(&device);

        // ONNX OneHot's `indices` input is int64 per spec; construct the test
        // tensor with explicit I64 dtype rather than relying on the backend's
        // default Int element.
        let input: Tensor<1, Int> =
            Tensor::from_data(TensorData::from([0i64, 2, 3]), (&device, DType::I64));
        let output: Tensor<2, Int> = model.forward(input);

        let expected = TensorData::from([[1i64, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn one_hot_float_values() {
        // Int indices with float values [0.0, 5.0] -> Float output
        let device = Default::default();
        let model = one_hot_float_values::Model::new(&device);

        let input: Tensor<1, Int> = Tensor::from_ints([0, 1, 2, 1], &device);
        let output: Tensor<2> = model.forward(input);

        let expected = TensorData::from([
            [5.0f32, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 5.0, 0.0],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn one_hot_2d() {
        // 2D input indices [2, 3] -> output [2, 3, 4]
        let device = Default::default();
        let model = one_hot_2d::Model::new(&device);

        // ONNX OneHot's `indices` input is int64 per spec; pass the dtype
        // explicitly so the model sees I64 rather than the backend-default
        // int that `from_data(data, &device)` would produce.
        let input: Tensor<2, Int> = Tensor::from_data(
            TensorData::from([[0i64, 1, 2], [3, 0, 1]]),
            (&device, DType::I64),
        );
        let output: Tensor<3, Int> = model.forward(input);

        let expected = TensorData::from([
            [[1i64, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn one_hot_runtime_depth() {
        // Expected values from one_hot_runtime_depth.py (onnx ReferenceEvaluator).
        let device = Default::default();
        let model = one_hot_runtime_depth::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/one_hot_runtime_depth.bpk"),
            &device,
        );
        let indices = Tensor::<1, Int>::from_ints([0, 2, 1], &device);

        let output = model.forward(indices, 3);

        output.to_data().assert_eq(
            &TensorData::from([[1.0f32, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            true,
        );
    }

    // A negative runtime depth is out-of-spec; debug builds stop with a clear message.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "OneHot: depth must be non-negative, got -1")]
    fn one_hot_runtime_depth_negative_panics_in_debug() {
        let device = Default::default();
        let model = one_hot_runtime_depth::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/one_hot_runtime_depth.bpk"),
            &device,
        );
        let indices = Tensor::<1, Int>::from_ints([0, 2, 1], &device);

        let _ = model.forward(indices, -1);
    }

    #[test]
    fn one_hot_runtime_depth_tensor() {
        // Expected values from one_hot_runtime_depth_tensor.py (onnx ReferenceEvaluator).
        let device = Default::default();
        let model = one_hot_runtime_depth_tensor::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/one_hot_runtime_depth_tensor.bpk"),
            &device,
        );
        let indices = Tensor::<1, Int>::from_ints([0, 2, 1], &device);
        let depth = Tensor::<1, Int>::from_ints([3], &device);

        let output = model.forward(indices, depth);

        output.to_data().assert_eq(
            &TensorData::from([[1.0f32, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            true,
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "OneHot: depth must be non-negative, got -1")]
    fn one_hot_runtime_depth_tensor_negative_panics_in_debug() {
        let device = Default::default();
        let model = one_hot_runtime_depth_tensor::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/one_hot_runtime_depth_tensor.bpk"),
            &device,
        );
        let indices = Tensor::<1, Int>::from_ints([0, 2, 1], &device);
        let depth = Tensor::<1, Int>::from_ints([-1], &device);

        let _ = model.forward(indices, depth);
    }
}
