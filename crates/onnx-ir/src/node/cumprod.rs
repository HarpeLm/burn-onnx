//! # CumProd
//!
//! Performs cumulative product of the input tensor along the given axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__CumProd.html>
//!
//! ## Opset Versions
//! - **Opset 26**: Initial version

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::TensorDataExt;
use crate::ir::{ArgType, Argument, Node, RawNode, RuntimeInputRef};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Represents either a static axis value or a runtime input reference for CumProd.
#[derive(Debug, Clone)]
pub enum CumProdAxis {
    /// Static axis known at compile time (normalized to positive value).
    Static(usize),
    /// Runtime axis determined during execution (Shape type - CPU array of i64).
    Runtime(RuntimeInputRef),
}

/// Configuration for CumProd operation
#[derive(Debug, Clone, new)]
pub struct CumProdConfig {
    /// The axis along which to compute cumulative product
    pub axis: CumProdAxis,
    /// If true, the j-th output is the product of the first (j-1) elements (excludes current)
    pub exclusive: bool,
    /// If true, perform cumulative product in reverse direction
    pub reverse: bool,
}

/// Node representation for CumProd operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct CumProdNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: CumProdConfig,
}

pub(crate) struct CumProdProcessor;

impl NodeProcessor for CumProdProcessor {
    type Config = CumProdConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 26,
            max_opset: None,
            inputs: InputSpec::Exact(2), // x and axis
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Try to lift axis input (input[1]) to static value if it's a constant
        // If it's not a constant, it will remain as a runtime input
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Output type is the same as input type
        crate::processor::same_as_input(node);
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Extract axis from input[1] (scalar tensor) - can be static or runtime
        let axis_input = &node.inputs[1];
        let axis = match axis_input.value() {
            Some(value) => {
                // Static axis - extract from constant value
                let axis_vec = value.to_i64_vec().map_err(|e| {
                    ProcessError::Custom(format!("CumProd: failed to extract axis: {}", e))
                })?;

                // Axis is a scalar (0-D tensor), so it should have exactly one element
                let axis_value = if axis_vec.len() != 1 {
                    return Err(ProcessError::Custom(
                        "CumProd: axis must be a scalar (0-D tensor)".to_string(),
                    ));
                } else {
                    axis_vec[0]
                };

                // Rank for negative axis handling. A Shape input (e.g. from a Shape node) is
                // a rank-1 vector of int64.
                let rank = match &node.inputs[0].ty {
                    ArgType::Tensor(t) => t.rank,
                    ArgType::Shape(_) => 1,
                    _ => {
                        return Err(ProcessError::TypeMismatch {
                            expected: "Tensor or Shape".to_string(),
                            actual: format!("{:?}", node.inputs[0].ty),
                        });
                    }
                };

                // Handle negative axis
                let axis_normalized = if axis_value < 0 {
                    (rank as i64 + axis_value) as usize
                } else {
                    axis_value as usize
                };

                // Validate axis
                if axis_normalized >= rank {
                    return Err(ProcessError::Custom(format!(
                        "CumProd: axis {} is out of bounds for tensor of rank {}",
                        axis_value, rank
                    )));
                }

                CumProdAxis::Static(axis_normalized)
            }
            None => {
                // Runtime axis - create reference to input
                CumProdAxis::Runtime(RuntimeInputRef::new(axis_input.name.clone(), 1))
            }
        };

        // Extract exclusive attribute (default: 0)
        let exclusive = node
            .attrs
            .get("exclusive")
            .map(|v| v.clone().into_i64().map(|x| x != 0))
            .transpose()?
            .unwrap_or(false);

        // Extract reverse attribute (default: 0)
        let reverse = node
            .attrs
            .get("reverse")
            .map(|v| v.clone().into_i64().map(|x| x != 0))
            .transpose()?
            .unwrap_or(false);

        Ok(CumProdConfig {
            axis,
            exclusive,
            reverse,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        Ok(Node::CumProd(CumProdNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(axis: i64, exclusive: i64, reverse: i64, rank: usize) -> RawNode {
        TestNodeBuilder::new(NodeType::CumProd, "test_cumprod")
            .input_tensor_f32("x", rank, None)
            .input_tensor_i64_data("axis", vec![axis], vec![])
            .output_tensor_f32("y", rank, None)
            .attr_int("exclusive", exclusive)
            .attr_int("reverse", reverse)
            .build_with_graph_data(26)
    }

    #[test]
    fn test_cumprod_config_default() {
        let node = create_test_node(0, 0, 0, 3);
        let processor = CumProdProcessor;
        let config = processor.extract_config(&node, 26).unwrap();
        assert!(matches!(config.axis, CumProdAxis::Static(0)));
        assert!(!config.exclusive);
        assert!(!config.reverse);
    }

    #[test]
    fn test_cumprod_config_exclusive() {
        let node = create_test_node(1, 1, 0, 3);
        let processor = CumProdProcessor;
        let config = processor.extract_config(&node, 26).unwrap();
        assert!(matches!(config.axis, CumProdAxis::Static(1)));
        assert!(config.exclusive);
        assert!(!config.reverse);
    }

    #[test]
    fn test_cumprod_config_reverse() {
        let node = create_test_node(0, 0, 1, 3);
        let processor = CumProdProcessor;
        let config = processor.extract_config(&node, 26).unwrap();
        assert!(matches!(config.axis, CumProdAxis::Static(0)));
        assert!(!config.exclusive);
        assert!(config.reverse);
    }

    #[test]
    fn test_cumprod_config_exclusive_reverse() {
        let node = create_test_node(2, 1, 1, 3);
        let processor = CumProdProcessor;
        let config = processor.extract_config(&node, 26).unwrap();
        assert!(matches!(config.axis, CumProdAxis::Static(2)));
        assert!(config.exclusive);
        assert!(config.reverse);
    }

    #[test]
    fn test_cumprod_config_negative_axis() {
        let node = create_test_node(-1, 0, 0, 3);
        let processor = CumProdProcessor;
        let config = processor.extract_config(&node, 26).unwrap();
        assert!(matches!(config.axis, CumProdAxis::Static(2))); // -1 + 3 = 2
    }

    #[test]
    fn test_cumprod_shape_input() {
        // A Shape input (e.g. from a Shape node) is a rank-1 int64 vector: -1 is axis 0, and
        // the output keeps the Shape type
        let mut node = TestNodeBuilder::new(NodeType::CumProd, "test_cumprod_shape")
            .input_shape("x", 3)
            .input_tensor_i64_data("axis", vec![-1], vec![])
            .output_tensor_i64("y", 1, None)
            .build_with_graph_data(26);
        let processor = CumProdProcessor;
        let config = processor.extract_config(&node, 26).unwrap();
        assert!(matches!(config.axis, CumProdAxis::Static(0)));
        processor
            .infer_types(&mut node, 26, &OutputPreferences::new())
            .unwrap();
        assert!(matches!(node.outputs[0].ty, ArgType::Shape(3)));

        // Axis 1 is out of range for a rank-1 Shape
        let node = TestNodeBuilder::new(NodeType::CumProd, "test_cumprod_shape")
            .input_shape("x", 3)
            .input_tensor_i64_data("axis", vec![1], vec![])
            .output_tensor_i64("y", 1, None)
            .build_with_graph_data(26);
        assert!(processor.extract_config(&node, 26).is_err());
    }

    fn create_runtime_cumprod_node() -> RawNode {
        TestNodeBuilder::new(NodeType::CumProd, "test_cumprod_runtime")
            .input_tensor_f32("x", 3, Some(vec![2, 3, 4]))
            .input_tensor_i64("axis", 0, None) // Runtime input - no static value
            .output_tensor_f32("y", 3, None)
            .attr_int("exclusive", 0)
            .attr_int("reverse", 0)
            .build()
    }

    #[test]
    fn test_cumprod_config_runtime_axis() {
        let node = create_runtime_cumprod_node();
        let processor = CumProdProcessor;
        let config = processor.extract_config(&node, 26).unwrap();
        assert!(matches!(config.axis, CumProdAxis::Runtime(ref r) if r.name == "axis"));
        assert!(!config.exclusive);
        assert!(!config.reverse);
    }

    #[test]
    fn test_cumprod_type_inference() {
        let mut node = create_test_node(0, 0, 0, 3);
        let processor = CumProdProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 26, &prefs).unwrap();

        // Output should have same type as input
        match (&node.inputs[0].ty, &node.outputs[0].ty) {
            (ArgType::Tensor(input_t), ArgType::Tensor(output_t)) => {
                assert_eq!(input_t.dtype, output_t.dtype);
                assert_eq!(input_t.rank, output_t.rank);
            }
            _ => panic!("Expected tensor types"),
        }
    }
}
