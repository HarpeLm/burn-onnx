use crate::include_models;
include_models!(
    compress_axis0,
    compress_axis1,
    compress_negative_axis,
    compress_default_axis,
    compress_3d,
    compress_int64,
    compress_all_false
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, Int, Tensor, TensorData};

    fn input_3x2(device: &Device) -> Tensor<2> {
        Tensor::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device)
    }

    #[test]
    fn compress_axis0_test() {
        let device = Default::default();
        let model = compress_axis0::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/compress_axis0.bpk"),
            &device,
        );

        let condition = Tensor::from_bool([false, true, true], &device);

        let output = model.forward(input_3x2(&device), condition);

        let expected = TensorData::from([[3.0f32, 4.0], [5.0, 6.0]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn compress_axis1_test() {
        let device = Default::default();
        let model = compress_axis1::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/compress_axis1.bpk"),
            &device,
        );

        let condition = Tensor::from_bool([false, true], &device);

        let output = model.forward(input_3x2(&device), condition);

        let expected = TensorData::from([[2.0f32], [4.0], [6.0]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn compress_negative_axis_test() {
        let device = Default::default();
        let model = compress_negative_axis::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/compress_negative_axis.bpk"),
            &device,
        );

        let condition = Tensor::from_bool([false, true], &device);

        let output = model.forward(input_3x2(&device), condition);

        // axis = -1 counts from the back, so this matches axis = 1.
        let expected = TensorData::from([[2.0f32], [4.0], [6.0]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn compress_default_axis_test() {
        let device = Default::default();
        let model = compress_default_axis::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/compress_default_axis.bpk"),
            &device,
        );

        // The condition is shorter than the 6 flattened elements, which the spec
        // allows: the trailing element is simply never selected.
        let condition = Tensor::from_bool([false, true, false, false, true], &device);

        let output = model.forward(input_3x2(&device), condition);

        let expected = TensorData::from([2.0f32, 5.0]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn compress_3d_test() {
        let device = Default::default();
        let model = compress_3d::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/compress_3d.bpk"),
            &device,
        );

        let input = Tensor::<3>::from_floats(
            [
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                ],
                [
                    [12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0],
                ],
            ],
            &device,
        );
        let condition = Tensor::from_bool([true, false, true], &device);

        let output = model.forward(input, condition);

        let expected = TensorData::from([
            [[0.0f32, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]],
            [[12.0, 13.0, 14.0, 15.0], [20.0, 21.0, 22.0, 23.0]],
        ]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn compress_int64_test() {
        let device = Default::default();
        let model = compress_int64::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/compress_int64.bpk"),
            &device,
        );

        let input = Tensor::<2, Int>::from_data(
            TensorData::from([[1i64, 2], [3, 4], [5, 6]]),
            (&device, DType::I64),
        );
        let condition = Tensor::from_bool([false, true, true], &device);

        let output = model.forward(input, condition);

        // The ONNX-specified dtype must survive codegen, not fall back to the
        // device's default int dtype.
        assert_eq!(output.dtype(), DType::I64);
        let expected = TensorData::from([[3i64, 4], [5, 6]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn compress_all_false_test() {
        let device = Default::default();
        let model = compress_all_false::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/compress_all_false.bpk"),
            &device,
        );

        let condition = Tensor::from_bool([false, false, false], &device);

        let output = model.forward(input_3x2(&device), condition);

        assert_eq!(output.dims(), [0, 2]);
    }
}
