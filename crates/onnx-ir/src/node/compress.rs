//! # Compress
//!
//! Selects slices from an input tensor along a given axis where the condition
//! evaluates to true for each axis index. When `axis` is absent the input is
//! flattened before elements are selected.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Compress.html>
//!
//! ## Opset Versions
//! - **Opset 9**: Initial version
//! - **Opset 11**: Widened type constraints (no signature change)
//!
//! ## Type Constraints (from ONNX spec)
//! - T: all tensor types except string/complex, which burn cannot represent.
//!   `element_type_from_proto` already rejects those at parse time.
//! - T1: tensor(bool), rank 1

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for Compress operation
#[derive(Debug, Clone, Default, new)]
pub struct CompressConfig {
    /// Axis along which slices are selected. `None` flattens the input first.
    /// Stored verbatim from the ONNX attribute; normalization against the input
    /// rank happens where a concrete axis is needed.
    pub axis: Option<i64>,
}

/// Node representation for Compress operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct CompressNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: CompressConfig,
}

pub(crate) struct CompressProcessor;

impl NodeProcessor for CompressProcessor {
    type Config = CompressConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 9,
            max_opset: None,
            inputs: InputSpec::Exact(2), // data and condition
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let config = self.extract_config(node, opset)?;

        let (rank, dtype, static_shape) = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => (tensor.rank, tensor.dtype, tensor.static_shape.clone()),
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        };

        // Spec requires rank r >= 1. A rank-0 input has no axis to compress
        // along, and flattening it would silently promote it to rank 1.
        if rank == 0 {
            return Err(ProcessError::Custom(
                "Compress requires an input of rank >= 1, got a scalar".to_string(),
            ));
        }

        match &node.inputs[1].ty {
            ArgType::Tensor(condition) => {
                if condition.rank != 1 {
                    return Err(ProcessError::Custom(format!(
                        "Compress requires a rank 1 condition, got rank {}",
                        condition.rank
                    )));
                }
                if !condition.dtype.is_bool() {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Bool condition".to_string(),
                        actual: format!("{:?}", condition.dtype),
                    });
                }
            }
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        }

        // The number of retained slices depends on the data, so the selected
        // dimension is always dynamic. Without an axis the input is flattened
        // first, which also drops every dimension.
        let static_shape = match config.axis {
            Some(axis) => {
                let axis = normalize_axis(axis, rank)?;
                let mut shape = static_shape.unwrap_or_else(|| vec![None; rank]);
                shape.truncate(rank);
                shape.resize(rank, None);
                shape[axis] = None;
                Some(shape)
            }
            None => Some(vec![None]),
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype,
            rank: if config.axis.is_some() { rank } else { 1 },
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let axis = node
            .attrs
            .get("axis")
            .map(|v| v.clone().into_i64())
            .transpose()?;

        Ok(CompressConfig { axis })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        Ok(Node::Compress(CompressNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }))
    }
}

/// Resolve a possibly negative ONNX axis against a tensor rank.
fn normalize_axis(axis: i64, rank: usize) -> Result<usize, ProcessError> {
    let rank = rank as i64;
    let normalized = if axis < 0 { axis + rank } else { axis };

    if normalized < 0 || normalized >= rank {
        return Err(ProcessError::Custom(format!(
            "Compress: axis {axis} is out of bounds for tensor of rank {rank}"
        )));
    }

    Ok(normalized as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    fn build(axis: Option<i64>, rank: usize) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::Compress, "test_compress")
            .input_tensor_f32("input", rank, Some(vec![3; rank]))
            .input_tensor_bool("condition", 1, Some(vec![2]))
            .output_tensor_f32("output", rank, None);
        if let Some(axis) = axis {
            builder = builder.attr_int("axis", axis);
        }
        builder.build()
    }

    #[test]
    fn test_compress_axis_marks_dim_dynamic() {
        let mut node = build(Some(1), 2);
        let processor = CompressProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 2);
                // Only the compressed axis is dynamic; the other dim is inherited.
                assert_eq!(tensor.static_shape, Some(vec![Some(3), None]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_compress_negative_axis_marks_dim_dynamic() {
        let mut node = build(Some(-1), 3);
        let processor = CompressProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, Some(vec![Some(3), Some(3), None]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_compress_without_axis_flattens() {
        let mut node = build(None, 2);
        let processor = CompressProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 1);
                assert_eq!(tensor.static_shape, Some(vec![None]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_compress_without_static_shape_input() {
        let mut node = TestNodeBuilder::new(NodeType::Compress, "test_compress")
            .input_tensor_f32("input", 2, None)
            .input_tensor_bool("condition", 1, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 0)
            .build();

        let processor = CompressProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, Some(vec![None, None]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_compress_axis_out_of_bounds() {
        let mut node = build(Some(3), 2);
        let processor = CompressProcessor;
        let prefs = OutputPreferences::new();
        let err = processor.infer_types(&mut node, 16, &prefs).unwrap_err();
        assert!(matches!(err, ProcessError::Custom(_)));
    }

    #[test]
    fn test_compress_input_must_have_rank_at_least_1() {
        // Spec requires r >= 1; a scalar has no axis to compress along.
        for axis in [None, Some(0)] {
            let mut builder = TestNodeBuilder::new(NodeType::Compress, "test_compress")
                .input_tensor_f32("input", 0, None)
                .input_tensor_bool("condition", 1, Some(vec![1]))
                .output_tensor_f32("output", 1, None);
            if let Some(axis) = axis {
                builder = builder.attr_int("axis", axis);
            }
            let mut node = builder.build();

            let processor = CompressProcessor;
            let prefs = OutputPreferences::new();
            let err = processor.infer_types(&mut node, 16, &prefs).unwrap_err();
            assert!(matches!(err, ProcessError::Custom(_)), "axis={axis:?}");
        }
    }

    #[test]
    fn test_compress_condition_must_be_rank_1() {
        let mut node = TestNodeBuilder::new(NodeType::Compress, "test_compress")
            .input_tensor_f32("input", 2, Some(vec![3, 3]))
            .input_tensor_bool("condition", 2, Some(vec![3, 3]))
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 0)
            .build();

        let processor = CompressProcessor;
        let prefs = OutputPreferences::new();
        let err = processor.infer_types(&mut node, 16, &prefs).unwrap_err();
        assert!(matches!(err, ProcessError::Custom(_)));
    }

    #[test]
    fn test_compress_condition_must_be_bool() {
        let mut node = TestNodeBuilder::new(NodeType::Compress, "test_compress")
            .input_tensor_f32("input", 2, Some(vec![3, 3]))
            .input_tensor_f32("condition", 1, Some(vec![3]))
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 0)
            .build();

        let processor = CompressProcessor;
        let prefs = OutputPreferences::new();
        let err = processor.infer_types(&mut node, 16, &prefs).unwrap_err();
        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn test_compress_config_default_has_no_axis() {
        let node = build(None, 2);
        let processor = CompressProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.axis, None);
    }

    #[test]
    fn test_compress_config_keeps_axis_verbatim() {
        let node = build(Some(-1), 2);
        let processor = CompressProcessor;
        // Config mirrors ONNX, so the raw negative axis is preserved.
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.axis, Some(-1));
    }
}
