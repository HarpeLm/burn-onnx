//! # Resize
//!
//! Resizes input tensor using various interpolation methods.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Resize.html>
//!
//! ## Opset Versions
//! - **Opset 10**: Initial version with scales and sizes inputs.
//! - **Opset 11**: Added coordinate_transformation_mode attribute for more control over interpolation. Added support for linear mode (previously only nearest).
//! - **Opset 13**: Added cubic mode support and cubic_coeff_a attribute. Added antialias attribute for downsampling.
//! - **Opset 18**: Added keep_aspect_ratio_policy and axes attributes for selective resizing.
//! - **Opset 19**: Added antialiasing improvements and clarified coordinate transformation modes.
//!
//! **Implementation Note**: This implementation requires opset 11+ for coordinate transformation mode support. Many attributes are ignored or have restricted values (see validation in infer_types).
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::TensorDataExt;
use crate::ir::{ArgType, Node, RawNode, RuntimeInputRef, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use std::str::FromStr;

/// Interpolation mode for resize operation
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ResizeMode {
    /// Nearest neighbor interpolation
    #[default]
    Nearest,
    /// Linear interpolation (bilinear for 2D, trilinear for 3D)
    Linear,
    /// Cubic interpolation
    Cubic,
}

impl FromStr for ResizeMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nearest" => Ok(ResizeMode::Nearest),
            "linear" => Ok(ResizeMode::Linear),
            "cubic" => Ok(ResizeMode::Cubic),
            _ => Err(format!("Unsupported resize mode: {}", s)),
        }
    }
}

/// Coordinate transformation mode for resize operation.
///
/// Determines how coordinates in the resized tensor map to coordinates in the original tensor.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CoordinateTransformMode {
    /// Half-pixel coordinate transformation (default for opset 11+)
    #[default]
    HalfPixel,
    /// Align corners coordinate transformation
    AlignCorners,
    /// Asymmetric coordinate transformation (default for opset <11)
    Asymmetric,
    /// PyTorch-style half-pixel transformation
    PytorchHalfPixel,
    /// TensorFlow crop-and-resize transformation
    TfCropAndResize,
    /// TensorFlow half-pixel for nearest-neighbor
    TfHalfPixelForNn,
}

impl FromStr for CoordinateTransformMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "half_pixel" => Ok(Self::HalfPixel),
            "align_corners" => Ok(Self::AlignCorners),
            "asymmetric" => Ok(Self::Asymmetric),
            "pytorch_half_pixel" => Ok(Self::PytorchHalfPixel),
            "tf_crop_and_resize" => Ok(Self::TfCropAndResize),
            "tf_half_pixel_for_nn" => Ok(Self::TfHalfPixelForNn),
            _ => Err(format!("Unsupported coordinate transformation mode: {}", s)),
        }
    }
}

/// Nearest-neighbor rounding mode for resize operation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum NearestMode {
    /// Round half down (default)
    #[default]
    RoundPreferFloor,
    /// Round half up
    RoundPreferCeil,
    /// Floor rounding
    Floor,
    /// Ceil rounding
    Ceil,
}

impl FromStr for NearestMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "round_prefer_floor" => Ok(Self::RoundPreferFloor),
            "round_prefer_ceil" => Ok(Self::RoundPreferCeil),
            "floor" => Ok(Self::Floor),
            "ceil" => Ok(Self::Ceil),
            _ => Err(format!("Unsupported nearest mode: {}", s)),
        }
    }
}

/// Configuration for the Resize operation.
#[derive(Debug, Clone, new)]
#[allow(clippy::too_many_arguments)]
pub struct ResizeConfig {
    pub mode: ResizeMode,
    pub scales: Option<ResizeScales>,
    pub sizes: Option<ResizeSizes>,
    /// Coordinate transformation mode
    pub coordinate_transformation_mode: CoordinateTransformMode,
    /// Cubic coefficient for cubic interpolation (default: -0.75)
    pub cubic_coeff_a: f32,
    /// Nearest mode rounding strategy
    pub nearest_mode: NearestMode,
    /// Exclude outside weights (default: 0)
    pub exclude_outside: i32,
    /// Extrapolation value for tf_crop_and_resize mode (default: 0.0)
    pub extrapolation_value: f32,
    /// Antialias flag (default: 0) - opset 13+
    pub antialias: i32,
    /// Axes that the scales/sizes values apply to (`axes` attribute, opset 18+),
    /// normalized to non-negative indices. Static scales/sizes are already expanded to
    /// every axis, so only runtime values are laid out in this order.
    pub axes: Option<Vec<usize>>,
}

impl Default for ResizeConfig {
    fn default() -> Self {
        Self {
            mode: ResizeMode::Nearest,
            scales: None,
            sizes: None,
            coordinate_transformation_mode: CoordinateTransformMode::HalfPixel,
            cubic_coeff_a: -0.75,
            nearest_mode: NearestMode::RoundPreferFloor,
            exclude_outside: 0,
            extrapolation_value: 0.0,
            antialias: 0,
            axes: None,
        }
    }
}

/// Represents either a static value or a runtime argument for resize scales.
#[derive(Debug, Clone)]
pub enum ResizeScales {
    /// Static scales known at compile time.
    Static(Vec<f32>),
    /// Runtime scales determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

impl Default for ResizeScales {
    fn default() -> Self {
        Self::Static(Vec::new())
    }
}

/// Represents either a static value or a runtime argument for resize sizes.
#[derive(Debug, Clone)]
pub enum ResizeSizes {
    /// Static sizes known at compile time.
    Static(Vec<usize>),
    /// Runtime sizes determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

impl Default for ResizeSizes {
    fn default() -> Self {
        Self::Static(Vec::new())
    }
}

/// Node representation for Resize operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ResizeNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ResizeConfig,
}

/// Extract scales input as either static or runtime
fn extract_scales_input(
    node: &RawNode,
    input_rank: usize,
    axes: Option<&[usize]>,
    idx: usize,
) -> Result<Option<ResizeScales>, ProcessError> {
    let Some(input) = node.inputs.get(idx) else {
        return Ok(None);
    };
    // Skip optional inputs (those that were never provided)
    if input.is_optional() {
        return Ok(None);
    }

    match &input.ty {
        // A Shape input has a value too once simplification folds it to a constant,
        // and constant lifting then clears its name, so it must not become Runtime.
        ArgType::Tensor(_) | ArgType::Shape(_) => {
            // Check if it's a static value (lifted constant) or constant
            match input.value() {
                Some(tensor_data) => {
                    // `to_f32_vec` also accepts the i64 data of a Shape input
                    let scales: Vec<f32> = tensor_data.to_f32_vec().map_err(|e| {
                        ProcessError::Custom(format!("Resize: cannot read scales: {e:?}"))
                    })?;
                    if scales.is_empty() {
                        return Ok(None);
                    }
                    let expected = axes.map_or(input_rank, <[usize]>::len);
                    if scales.len() != expected {
                        return Err(ProcessError::Custom(format!(
                            "Resize: scales has {} values, expected {expected}",
                            scales.len()
                        )));
                    }
                    // Unlisted axes keep their size (scale 1.0)
                    let scales = match axes {
                        Some(axes) => expand_axes_to_full(&scales, axes, &vec![1.0; input_rank]),
                        None => scales,
                    };
                    // ignore the first two items from scales
                    // because they are the batch and channel dimensions
                    Ok(Some(ResizeScales::Static(scales[2..].to_vec())))
                }
                // Runtime input - store reference instead of cloning the argument
                None => Ok(Some(ResizeScales::Runtime(RuntimeInputRef::new(
                    input.name.clone(),
                    idx,
                )))),
            }
        }
        _ => Ok(None),
    }
}

/// Extract sizes input as either static or runtime
fn extract_sizes_input(
    node: &RawNode,
    input: &TensorType,
    axes: Option<&[usize]>,
    idx: usize,
) -> Result<Option<ResizeSizes>, ProcessError> {
    let input_rank = input.rank;
    let Some(sizes_input) = node.inputs.get(idx) else {
        return Ok(None);
    };
    // Skip optional inputs (those that were never provided)
    if sizes_input.is_optional() {
        return Ok(None);
    }

    match &sizes_input.ty {
        // A Shape input has a value too once simplification folds it to a constant,
        // and constant lifting then clears its name, so it must not become Runtime.
        ArgType::Tensor(_) | ArgType::Shape(_) => {
            // Check if it's a static value (lifted constant) or constant
            match sizes_input.value() {
                Some(tensor_data) => {
                    let i64_sizes: Vec<i64> = tensor_data.try_into_vec().map_err(|e| {
                        ProcessError::Custom(format!("Resize: cannot read sizes: {e:?}"))
                    })?;
                    if i64_sizes.is_empty() {
                        return Ok(None);
                    }
                    let expected = axes.map_or(input_rank, <[usize]>::len);
                    if i64_sizes.len() != expected {
                        return Err(ProcessError::Custom(format!(
                            "Resize: sizes has {} values, expected {expected}",
                            i64_sizes.len()
                        )));
                    }
                    // Unlisted axes keep their input size, which must be known statically.
                    // Batch and channel (axes 0 and 1) are dropped below, so any value works.
                    let i64_sizes = match axes {
                        Some(axes) => {
                            let mut unchanged = vec![0i64; input_rank];
                            for (dim, size) in unchanged.iter_mut().enumerate().skip(2) {
                                if axes.contains(&dim) {
                                    continue;
                                }
                                let known = input
                                    .static_shape
                                    .as_ref()
                                    .and_then(|shape| shape.get(dim).copied().flatten());
                                *size = known.ok_or_else(|| {
                                    ProcessError::Custom(format!(
                                        "Resize: sizes with axes {axes:?} needs the static size of axis {dim}"
                                    ))
                                })? as i64;
                            }
                            expand_axes_to_full(&i64_sizes, axes, &unchanged)
                        }
                        None => i64_sizes,
                    };
                    // ignore the first two items from sizes
                    // because they are the batch and channel dimensions
                    let sizes = i64_sizes[2..]
                        .iter()
                        .map(|&x| usize::try_from(x))
                        .collect::<Result<Vec<usize>, _>>()
                        .map_err(|_| {
                            ProcessError::Custom(format!(
                                "Resize: sizes must be non-negative, got {i64_sizes:?}"
                            ))
                        })?;
                    Ok(Some(ResizeSizes::Static(sizes)))
                }
                // Runtime input - store reference instead of cloning the argument
                None => Ok(Some(ResizeSizes::Runtime(RuntimeInputRef::new(
                    sizes_input.name.clone(),
                    idx,
                )))),
            }
        }
        _ => Ok(None),
    }
}

/// Place per-axis `values` at their `axes`, starting from `full` (one value per input axis).
fn expand_axes_to_full<T: Copy>(values: &[T], axes: &[usize], full: &[T]) -> Vec<T> {
    let mut full = full.to_vec();
    for (&axis, &value) in axes.iter().zip(values) {
        full[axis] = value;
    }
    full
}

/// Normalize the `axes` attribute: negative axes count from the end, and every axis
/// must be in range and appear only once.
fn normalize_axes(axes: &[i64], rank: usize) -> Result<Vec<usize>, ProcessError> {
    let mut normalized = Vec::with_capacity(axes.len());
    for &axis in axes {
        let resolved = if axis < 0 { axis + rank as i64 } else { axis };
        if resolved < 0 || resolved >= rank as i64 {
            return Err(ProcessError::InvalidAttribute {
                name: "axes".to_string(),
                reason: format!("axis {axis} is out of range for rank {rank}"),
            });
        }
        let resolved = resolved as usize;
        if normalized.contains(&resolved) {
            return Err(ProcessError::InvalidAttribute {
                name: "axes".to_string(),
                reason: format!("axis {axis} appears more than once"),
            });
        }
        normalized.push(resolved);
    }
    Ok(normalized)
}

pub(crate) struct ResizeProcessor;

impl NodeProcessor for ResizeProcessor {
    type Config = ResizeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::Range(1, 4),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift roi input (input[1]) if present and constant
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        // Lift scales input (input[2]) if present and constant
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        // Lift sizes input (input[3]) if present and constant
        if node.inputs.len() > 3 && node.inputs[3].is_constant() {
            node.inputs[3].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "antialias" if value.clone().into_i32()? != 0 => {
                    return Err(ProcessError::InvalidAttribute {
                        name: "antialias".to_string(),
                        reason: "antialias other than 0 is not supported".to_string(),
                    });
                }
                "axes" | "coordinate_transformation_mode" | "cubic_coeff_a" => {
                    // Parsed in extract_config
                }
                "exclude_outside" if value.clone().into_i32()? != 0 => {
                    return Err(ProcessError::InvalidAttribute {
                        name: "exclude_outside".to_string(),
                        reason: "exclude_outside other than 0 is not supported".to_string(),
                    });
                }
                "extrapolation_value" if value.clone().into_f32()? != 0.0 => {
                    return Err(ProcessError::InvalidAttribute {
                        name: "extrapolation_value".to_string(),
                        reason: "extrapolation_value other than 0.0 is not supported".to_string(),
                    });
                }
                "keep_aspect_ratio_policy"
                    if value.clone().into_string()?.to_lowercase() != "stretch" =>
                {
                    return Err(ProcessError::InvalidAttribute {
                        name: "keep_aspect_ratio_policy".to_string(),
                        reason: "keep_aspect_ratio_policy other than 'stretch' is not supported"
                            .to_string(),
                    });
                }
                "mode" | "nearest_mode" => {
                    // Parsed in extract_config
                }
                _ => {}
            }
        }

        // Opset 10: inputs are [X, scales] (no roi input)
        // Opset 11+: inputs are [X, roi, scales, sizes]
        if opset >= 11 {
            let has_roi = node
                .inputs
                .get(1)
                .and_then(|input| input.value())
                .is_some_and(|roi| roi.num_elements() > 0);

            if has_roi {
                return Err(ProcessError::Custom(
                    "Resize: roi input is not supported".to_string(),
                ));
            }
        }

        let config = self.extract_config(node, opset)?;

        // Exactly one of scales or sizes must be provided
        match (&config.scales, &config.sizes) {
            (None, None) => {
                return Err(ProcessError::Custom(
                    "Resize: either scales or sizes input is required".to_string(),
                ));
            }
            (Some(_), Some(_)) => {
                return Err(ProcessError::Custom(
                    "Resize: scales and sizes are mutually exclusive".to_string(),
                ));
            }
            _ => {}
        }

        // Infer output type
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, opset: usize) -> Result<Self::Config, ProcessError> {
        let mut mode: Option<ResizeMode> = None;
        // Opset 10 had no coordinate_transformation_mode (implicit asymmetric)
        let mut coordinate_transformation_mode = if opset < 11 {
            CoordinateTransformMode::Asymmetric
        } else {
            CoordinateTransformMode::HalfPixel
        };
        let mut cubic_coeff_a = -0.75f32;
        let mut nearest_mode = NearestMode::RoundPreferFloor;
        let mut exclude_outside = 0i32;
        let mut extrapolation_value = 0.0f32;
        let mut antialias = 0i32;
        let mut axes_attr: Option<Vec<i64>> = None;

        let input = if let ArgType::Tensor(tensor) = &node
            .inputs
            .first()
            .ok_or_else(|| ProcessError::MissingInput("input".to_string()))?
            .ty
        {
            tensor
        } else {
            return Err(ProcessError::TypeMismatch {
                expected: "Tensor".to_string(),
                actual: format!("{:?}", node.inputs.first().unwrap().ty),
            });
        };

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "mode" => {
                    mode = Some(value.clone().into_string()?.parse::<ResizeMode>().map_err(
                        |e| ProcessError::InvalidAttribute {
                            name: "mode".to_string(),
                            reason: format!("Failed to parse resize mode: {}", e),
                        },
                    )?)
                }
                "coordinate_transformation_mode" => {
                    coordinate_transformation_mode = value
                        .clone()
                        .into_string()?
                        .parse::<CoordinateTransformMode>()
                        .map_err(|e| ProcessError::InvalidAttribute {
                            name: "coordinate_transformation_mode".to_string(),
                            reason: e,
                        })?;
                }
                "cubic_coeff_a" => {
                    cubic_coeff_a = value.clone().into_f32()?;
                }
                "nearest_mode" => {
                    nearest_mode = value
                        .clone()
                        .into_string()?
                        .parse::<NearestMode>()
                        .map_err(|e| ProcessError::InvalidAttribute {
                            name: "nearest_mode".to_string(),
                            reason: e,
                        })?;
                }
                "exclude_outside" => {
                    exclude_outside = value.clone().into_i32()?;
                }
                "extrapolation_value" => {
                    extrapolation_value = value.clone().into_f32()?;
                }
                "antialias" => {
                    antialias = value.clone().into_i32()?;
                }
                "axes" => {
                    axes_attr = Some(value.clone().into_i64s()?);
                }
                _ => {}
            }
        }

        // Opset 10: inputs are [X, scales] (no roi, no sizes)
        // Opset 11+: inputs are [X, roi, scales, sizes]
        let (scales_idx, sizes_idx) = if opset < 11 { (1, usize::MAX) } else { (2, 3) };

        let axes = axes_attr
            .map(|axes| normalize_axes(&axes, input.rank))
            .transpose()?;

        let scales = extract_scales_input(node, input.rank, axes.as_deref(), scales_idx)?;
        let sizes = extract_sizes_input(node, input, axes.as_deref(), sizes_idx)?;

        let mode = mode.ok_or_else(|| ProcessError::MissingAttribute("mode".to_string()))?;

        let config = ResizeConfig {
            mode,
            scales,
            sizes,
            coordinate_transformation_mode,
            cubic_coeff_a,
            nearest_mode,
            exclude_outside,
            extrapolation_value,
            antialias,
            axes,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        Ok(Node::Resize(ResizeNode {
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

    fn create_test_node(
        mode: &str,
        scales: Option<Vec<f32>>,
        sizes: Option<Vec<i64>>,
        roi: Option<Vec<f32>>,
    ) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Resize, "test_resize")
            .input_tensor_f32("X", 4, None) // N,C,H,W format
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", mode);

        // Add ROI input if provided
        if let Some(roi_data) = roi {
            builder = builder.input_tensor_f32_data("roi", roi_data.clone(), vec![8]);
            // For 4D input (start x, start y, end x, end y)
        } else {
            // Empty ROI still needs to be added as a placeholder with empty name
            builder = builder.input_tensor_f32("", 1, None);
        }

        // Add scales input if provided
        if let Some(scales_data) = scales {
            builder = builder.input_tensor_f32_data("scales", scales_data.clone(), vec![4]);
            // N,C,H,W scales
        } else {
            // Empty scales still needs to be added as a placeholder with empty name
            builder = builder.input_tensor_f32("", 1, None);
        }

        // Add sizes input if provided
        if let Some(sizes_data) = sizes {
            builder = builder.input_tensor_i64_data("sizes", sizes_data.clone(), vec![4]);
            // N,C,H,W sizes
        } else {
            // Empty sizes still needs to be added as a placeholder with empty name
            builder = builder.input_tensor_i64("", 1, None);
        }

        builder
    }

    #[test]
    fn test_resize_config_with_scales() {
        let node = create_test_node(
            "nearest",
            Some(vec![1.0, 1.0, 2.0, 2.0]), // Keep N,C same, double H,W
            None,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.mode, ResizeMode::Nearest);
        match &config.scales {
            Some(ResizeScales::Static(scales)) => {
                assert_eq!(*scales, vec![2.0, 2.0]); // Only the spatial scales (H,W)
            }
            _ => panic!("Expected static scales"),
        }
        assert!(config.sizes.is_none(), "Expected no sizes");
        // Verify default attribute values
        assert_eq!(
            config.coordinate_transformation_mode,
            CoordinateTransformMode::HalfPixel
        );
        assert_eq!(config.cubic_coeff_a, -0.75);
        assert_eq!(config.nearest_mode, NearestMode::RoundPreferFloor);
        assert_eq!(config.exclude_outside, 0);
        assert_eq!(config.extrapolation_value, 0.0);
        assert_eq!(config.antialias, 0);
    }

    #[test]
    fn test_resize_config_with_sizes() {
        let node = create_test_node(
            "linear",
            None,
            Some(vec![1, 3, 224, 224]), // Fixed output size
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.mode, ResizeMode::Linear);
        assert!(config.scales.is_none(), "Expected no scales");
        match &config.sizes {
            Some(ResizeSizes::Static(sizes)) => {
                assert_eq!(*sizes, vec![224, 224]); // Only the spatial sizes (H,W)
            }
            _ => panic!("Expected static sizes"),
        }
    }

    #[test]
    fn test_resize_config_with_runtime_shape_sizes() {
        let node = TestNodeBuilder::new(NodeType::Resize, "test_resize")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .input_tensor_f32("", 1, None)
            .input_tensor_f32("", 1, None)
            .input_shape("sizes", 4)
            .build_with_graph_data(16);
        let config = ResizeProcessor.extract_config(&node, 16).unwrap();
        match &config.sizes {
            Some(ResizeSizes::Runtime(r)) => assert_eq!(r.name, "sizes"),
            other => panic!("Expected runtime sizes, got {other:?}"),
        }
    }

    #[test]
    fn test_resize_config_with_lifted_shape_sizes() {
        // Simplification can fold a Shape computation feeding `sizes` into a constant;
        // once lifted its name is cleared, so the config must carry the value.
        let mut node = TestNodeBuilder::new(NodeType::Resize, "test_resize")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .input_tensor_f32("", 1, None)
            .input_tensor_f32("", 1, None)
            .input_shape_with_data("sizes", vec![1, 3, 8, 8])
            .build_with_graph_data(16);
        let processor = ResizeProcessor;
        processor.lift_constants(&mut node, 16).unwrap();
        assert!(node.inputs[3].is_static());
        let config = processor.extract_config(&node, 16).unwrap();
        match &config.sizes {
            Some(ResizeSizes::Static(sizes)) => assert_eq!(*sizes, vec![8, 8]),
            other => panic!("Expected static sizes, got {other:?}"),
        }
    }

    #[test]
    fn test_resize_config_with_lifted_shape_scales() {
        let mut node = TestNodeBuilder::new(NodeType::Resize, "test_resize")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .input_tensor_f32("", 1, None)
            .input_shape_with_data("scales", vec![1, 1, 2, 2])
            .input_tensor_i64("", 1, None)
            .build_with_graph_data(16);
        let processor = ResizeProcessor;
        processor.lift_constants(&mut node, 16).unwrap();
        assert!(node.inputs[2].is_static());
        let config = processor.extract_config(&node, 16).unwrap();
        match &config.scales {
            Some(ResizeScales::Static(scales)) => assert_eq!(*scales, vec![2.0, 2.0]),
            other => panic!("Expected static scales, got {other:?}"),
        }
    }

    #[test]
    fn test_resize_config_with_roi() {
        let node = create_test_node(
            "nearest",
            Some(vec![1.0, 1.0, 2.0, 2.0]),
            None,
            Some(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), // ROI values
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_resize_config_no_scales_or_sizes() {
        let node = create_test_node("nearest", None, None, None).build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_resize_config_no_mode() {
        let mut node = create_test_node("nearest", Some(vec![1.0, 1.0, 2.0, 2.0]), None, None)
            .build_with_graph_data(16);
        node.attrs.clear(); // Remove all attributes including mode
        let node = node;
        let processor = ResizeProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::MissingAttribute(_))));
    }

    #[test]
    fn test_resize_invalid_coordinate_transformation_mode() {
        let mut node = create_test_node("nearest", Some(vec![1.0, 1.0, 2.0, 2.0]), None, None)
            .build_with_graph_data(16);
        node.attrs.insert(
            "coordinate_transformation_mode".to_string(),
            crate::ir::AttributeValue::String("invalid_mode".to_string()),
        );
        let processor = ResizeProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_resize_invalid_nearest_mode() {
        let mut node = create_test_node("nearest", Some(vec![1.0, 1.0, 2.0, 2.0]), None, None)
            .build_with_graph_data(16);
        node.attrs.insert(
            "nearest_mode".to_string(),
            crate::ir::AttributeValue::String("bad_mode".to_string()),
        );
        let processor = ResizeProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_resize_scales_and_sizes_both_provided() {
        let node = create_test_node(
            "nearest",
            Some(vec![1.0, 1.0, 2.0, 2.0]),
            Some(vec![1, 3, 224, 224]),
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_resize_coordinate_transformation_mode_opset10_default() {
        // For opset 10, only inputs are [X, scales] (no roi, no sizes)
        let node = TestNodeBuilder::new(NodeType::Resize, "test_resize")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .input_tensor_f32_data("scales", vec![1.0, 1.0, 2.0, 2.0], vec![4])
            .build_with_graph_data(10);
        let processor = ResizeProcessor;
        let config = processor.extract_config(&node, 10).unwrap();
        assert_eq!(
            config.coordinate_transformation_mode,
            CoordinateTransformMode::Asymmetric
        );
    }

    /// Resize node with the `axes` attribute, a static input shape and either
    /// constant scales or constant sizes for the listed axes.
    fn create_axes_node(
        axes: Vec<i64>,
        scales: Option<Vec<f32>>,
        sizes: Option<Vec<i64>>,
    ) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::Resize, "test_resize")
            .input_tensor_f32("X", 4, Some(vec![1, 1, 2, 4]))
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .attr_ints("axes", axes)
            .input_tensor_f32("", 1, None);
        builder = match scales {
            Some(data) => {
                let len = data.len();
                builder.input_tensor_f32_data("scales", data, vec![len])
            }
            None => builder.input_tensor_f32("", 1, None),
        };
        if let Some(data) = sizes {
            let len = data.len();
            builder = builder.input_tensor_i64_data("sizes", data, vec![len]);
        }
        builder.build_with_graph_data(18)
    }

    #[test]
    fn test_resize_axes_static_scales_reordered() {
        // axes=[3, 2]: the first scale is for the width, the second for the height
        let node = create_axes_node(vec![3, 2], Some(vec![3.0, 2.0]), None);
        let config = ResizeProcessor.extract_config(&node, 18).unwrap();
        match &config.scales {
            Some(ResizeScales::Static(scales)) => assert_eq!(*scales, vec![2.0, 3.0]),
            other => panic!("Expected static scales, got {other:?}"),
        }
        assert_eq!(config.axes, Some(vec![3, 2]));
    }

    #[test]
    fn test_resize_axes_static_scales_unlisted_axis_unchanged() {
        // Only the width is resized; the height keeps scale 1.0
        let node = create_axes_node(vec![-1], Some(vec![2.0]), None);
        let config = ResizeProcessor.extract_config(&node, 18).unwrap();
        match &config.scales {
            Some(ResizeScales::Static(scales)) => assert_eq!(*scales, vec![1.0, 2.0]),
            other => panic!("Expected static scales, got {other:?}"),
        }
        assert_eq!(config.axes, Some(vec![3]));
    }

    #[test]
    fn test_resize_axes_static_sizes_unlisted_axis_unchanged() {
        // Only the height is resized; the width keeps its static input size (4)
        let node = create_axes_node(vec![2], None, Some(vec![6]));
        let config = ResizeProcessor.extract_config(&node, 18).unwrap();
        match &config.sizes {
            Some(ResizeSizes::Static(sizes)) => assert_eq!(*sizes, vec![6, 4]),
            other => panic!("Expected static sizes, got {other:?}"),
        }
    }

    #[test]
    fn test_resize_axes_static_sizes_need_static_shape() {
        // Without a static input shape, the unlisted width has no known size
        let node = TestNodeBuilder::new(NodeType::Resize, "test_resize")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .attr_ints("axes", vec![2])
            .input_tensor_f32("", 1, None)
            .input_tensor_f32("", 1, None)
            .input_tensor_i64_data("sizes", vec![6], vec![1])
            .build_with_graph_data(18);
        let result = ResizeProcessor.extract_config(&node, 18);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_resize_axes_runtime_scales_keep_axes() {
        let node = TestNodeBuilder::new(NodeType::Resize, "test_resize")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .attr_ints("axes", vec![3, 2])
            .input_tensor_f32("", 1, None)
            .input_tensor_f32("scales", 1, None)
            .build();
        let config = ResizeProcessor.extract_config(&node, 18).unwrap();
        assert!(matches!(&config.scales, Some(ResizeScales::Runtime(r)) if r.name == "scales"));
        assert_eq!(config.axes, Some(vec![3, 2]));
    }

    #[test]
    fn test_resize_axes_wrong_value_count() {
        let node = create_axes_node(vec![2, 3], Some(vec![2.0]), None);
        let result = ResizeProcessor.extract_config(&node, 18);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_resize_axes_invalid() {
        for axes in [vec![4], vec![2, -2]] {
            let node = create_axes_node(axes, Some(vec![2.0, 2.0]), None);
            let result = ResizeProcessor.extract_config(&node, 18);
            assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
        }
    }
}
