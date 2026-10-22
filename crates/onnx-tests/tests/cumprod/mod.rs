use crate::include_models;
include_models!(
    cumprod_runtime_axis,
    cumprod_int32,
    cumprod_computed_axis,
    cumprod_shape
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Int, Tensor, TensorData};

    fn runtime_axis_model(device: &burn::tensor::Device) -> cumprod_runtime_axis::Model {
        cumprod_runtime_axis::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/cumprod_runtime_axis.bpk"),
            device,
        )
    }

    fn input(device: &burn::tensor::Device) -> Tensor<2> {
        Tensor::<2>::from_floats([[1., 2., 3.], [4., 5., 6.]], device)
    }

    #[test]
    fn cumprod_negative_runtime_axis() {
        // Expected values from cumprod_runtime_axis.py (ONNX reference), axis = -1
        let device = Default::default();
        let (y, y_reverse, y_exclusive, y_exclusive_reverse) =
            runtime_axis_model(&device).forward(input(&device), -1);

        y.to_data()
            .assert_eq(&TensorData::from([[1f32, 2., 6.], [4., 20., 120.]]), true);
        y_reverse
            .to_data()
            .assert_eq(&TensorData::from([[6f32, 6., 3.], [120., 30., 6.]]), true);
        y_exclusive
            .to_data()
            .assert_eq(&TensorData::from([[1f32, 1., 2.], [1., 4., 20.]]), true);
        y_exclusive_reverse
            .to_data()
            .assert_eq(&TensorData::from([[6f32, 3., 1.], [30., 6., 1.]]), true);
    }

    #[test]
    fn cumprod_runtime_axis_0() {
        // Expected values from cumprod_runtime_axis.py (ONNX reference), axis = 0
        let device = Default::default();
        let (y, y_reverse, y_exclusive, y_exclusive_reverse) =
            runtime_axis_model(&device).forward(input(&device), 0);

        y.to_data()
            .assert_eq(&TensorData::from([[1f32, 2., 3.], [4., 10., 18.]]), true);
        y_reverse
            .to_data()
            .assert_eq(&TensorData::from([[4f32, 10., 18.], [4., 5., 6.]]), true);
        y_exclusive
            .to_data()
            .assert_eq(&TensorData::from([[1f32, 1., 1.], [1., 2., 3.]]), true);
        y_exclusive_reverse
            .to_data()
            .assert_eq(&TensorData::from([[4f32, 5., 6.], [1., 1., 1.]]), true);
    }

    #[test]
    #[should_panic(expected = "CumProd: axis 2 is out of range for a rank-2 input")]
    fn cumprod_runtime_axis_out_of_range() {
        let device = Default::default();
        let _ = runtime_axis_model(&device).forward(input(&device), 2);
    }

    #[test]
    fn cumprod_int32_keeps_dtype() {
        // Expected values from cumprod_int32.py (ONNX reference), constant axis 1
        let device = Default::default();
        let model = cumprod_int32::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/cumprod_int32.bpk"),
            &device,
        );
        let x = Tensor::<2, Int>::from_data(
            TensorData::from([[2i32, 3, 4], [-1, 5, 2]]),
            (&device, DType::I32),
        );

        let (y_exclusive, y_exclusive_reverse) = model.forward(x);

        assert_eq!(y_exclusive.dtype(), DType::I32);
        assert_eq!(y_exclusive_reverse.dtype(), DType::I32);
        y_exclusive
            .to_data()
            .assert_eq(&TensorData::from([[1i32, 2, 6], [1, -1, -5]]), true);
        y_exclusive_reverse
            .to_data()
            .assert_eq(&TensorData::from([[12i32, 4, 1], [10, 2, 1]]), true);
    }

    #[test]
    fn cumprod_computed_axis() {
        // Expected values from cumprod_computed_axis.py (ONNX reference). The axis is an
        // ArgMax with keepdims=0, a scalar on the device.
        let device = Default::default();
        let model = cumprod_computed_axis::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/cumprod_computed_axis.bpk"),
            &device,
        );

        // ArgMax = 1: reverse cumprod along the rows
        let selector = Tensor::<1>::from_floats([0., 5.], &device);
        let output = model.forward(input(&device), selector);
        output
            .to_data()
            .assert_eq(&TensorData::from([[6f32, 6., 3.], [120., 30., 6.]]), true);

        // ArgMax = 0: reverse cumprod along the columns
        let selector = Tensor::<1>::from_floats([5., 0.], &device);
        let output = model.forward(input(&device), selector);
        output
            .to_data()
            .assert_eq(&TensorData::from([[4f32, 10., 18.], [4., 5., 6.]]), true);
    }

    #[test]
    fn cumprod_shape_input() {
        // Expected values from cumprod_shape.py (ONNX reference). The data is the Shape of an
        // input with dynamic dimensions: constant axis 0, then runtime axis -1 with exclusive
        // and reverse.
        let device = Default::default();
        let model = cumprod_shape::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/cumprod_shape.bpk"),
            &device,
        );
        let x = Tensor::<3>::zeros([2, 3, 4], &device);

        let (y, y_exclusive_reverse) = model.forward(x, -1);

        assert_eq!(y, [2, 6, 24]);
        assert_eq!(y_exclusive_reverse, [12, 4, 1]);
    }

    #[test]
    #[should_panic(expected = "CumProd: axis 1 is out of range for a rank-1 input")]
    fn cumprod_shape_input_axis_out_of_range() {
        let device = Default::default();
        let model = cumprod_shape::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/cumprod_shape.bpk"),
            &device,
        );
        let _ = model.forward(Tensor::<3>::zeros([2, 3, 4], &device), 1);
    }
}
