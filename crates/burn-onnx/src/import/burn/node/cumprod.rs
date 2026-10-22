use super::prelude::*;
use onnx_ir::cumprod::CumProdAxis;
use onnx_ir::ir::ArgType;

impl NodeCodegen for onnx_ir::cumprod::CumProdNode {
    fn inputs(&self) -> &[Argument] {
        match &self.config.axis {
            // Static axis: only data input needed
            CumProdAxis::Static(_) => &self.inputs[..1],
            // Runtime axis: include both data and axis inputs
            CumProdAxis::Runtime(_) => &self.inputs,
        }
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let exclusive = self.config.exclusive;
        let reverse = self.config.reverse;

        let shape_len = match &self.inputs[0].ty {
            ArgType::Shape(len) => Some(*len),
            _ => None,
        };
        match &self.config.axis {
            CumProdAxis::Static(axis) => match shape_len {
                // A Shape input is a host [i64; N] array: compute on the host
                Some(len) => generate_shape_cumprod(&input, &output, len, None, exclusive, reverse),
                None => generate_static_cumprod(&input, &output, *axis, exclusive, reverse),
            },
            CumProdAxis::Runtime(_) => {
                // The axis as an i64 expression: a native scalar, a device scalar (e.g. a
                // computed ArgMax), or a one-element Shape
                let axis_arg = &self.inputs[1];
                let axis_value = match &axis_arg.ty {
                    ArgType::Shape(_) => {
                        let axis = arg_to_ident(axis_arg);
                        quote! { #axis[0] as i64 }
                    }
                    _ => scalar_as_i64(axis_arg, scope.arg(axis_arg)),
                };
                match shape_len {
                    Some(len) => generate_shape_cumprod(
                        &input,
                        &output,
                        len,
                        Some(axis_value),
                        exclusive,
                        reverse,
                    ),
                    None => generate_runtime_cumprod(
                        &input,
                        &output,
                        axis_value,
                        self.inputs[0].ty.rank(),
                        exclusive,
                        reverse,
                    ),
                }
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Tensor is needed for Tensor::cat in exclusive mode
        if self.config.exclusive {
            imports.register("burn::tensor::Tensor");
        }
    }
}

/// Generate code for static (compile-time known) axis
fn generate_static_cumprod(
    input: &TokenStream,
    output: &syn::Ident,
    axis: usize,
    exclusive: bool,
    reverse: bool,
) -> TokenStream {
    let axis = axis.to_tokens();

    match (exclusive, reverse) {
        (false, false) => {
            quote! {
                let #output = #input.cumprod(#axis);
            }
        }
        (false, true) => {
            quote! {
                let #output = #input.flip([#axis]).cumprod(#axis).flip([#axis]);
            }
        }
        (true, false) => {
            // Exclusive: output[i] = product(input[0..i]), excludes current element
            // Shift cumprod right by prepending ones and dropping last element
            quote! {
                let #output = {
                    let cumprod_result = #input.cumprod(#axis);
                    let shape = cumprod_result.shape();
                    let dim_size = shape[#axis];
                    if dim_size <= 1 {
                        // For empty or single element, exclusive cumprod is all ones
                        cumprod_result.ones_like()
                    } else {
                        let sliced = cumprod_result.narrow(#axis, 0, dim_size - 1);
                        let ones = sliced.ones_like().narrow(#axis, 0, 1);
                        Tensor::cat([ones, sliced].into(), #axis)
                    }
                };
            }
        }
        (true, true) => {
            // Exclusive + Reverse: output[i] = product(input[i+1..n])
            // Reverse cumprod, then shift left by dropping first and appending ones
            quote! {
                let #output = {
                    let flipped = #input.flip([#axis]);
                    let cumprod_result = flipped.cumprod(#axis);
                    let cumprod_back = cumprod_result.flip([#axis]);
                    let shape = cumprod_back.shape();
                    let dim_size = shape[#axis];
                    if dim_size <= 1 {
                        // For empty or single element, exclusive reverse cumprod is all ones
                        cumprod_back.ones_like()
                    } else {
                        let sliced = cumprod_back.narrow(#axis, 1, dim_size - 1);
                        let ones = sliced.ones_like().narrow(#axis, 0, 1);
                        Tensor::cat([sliced, ones].into(), #axis)
                    }
                };
            }
        }
    }
}

/// Generate code for a Shape input (a host `[i64; len]` array, logically rank 1): the
/// product runs on the host, with wrapping arithmetic like integer tensors. The only valid
/// axis is 0 (or -1).
fn generate_shape_cumprod(
    input: &TokenStream,
    output: &syn::Ident,
    len: usize,
    runtime_axis: Option<TokenStream>,
    exclusive: bool,
    reverse: bool,
) -> TokenStream {
    let len = len.to_tokens();
    let indices = if reverse {
        quote! { (0..#len).rev() }
    } else {
        quote! { 0..#len }
    };
    let step = if exclusive {
        quote! {
            output[i] = product;
            product = product.wrapping_mul(input[i]);
        }
    } else {
        quote! {
            product = product.wrapping_mul(input[i]);
            output[i] = product;
        }
    };
    // Both graph values are read before any temporary is declared
    let read_inputs = match runtime_axis {
        Some(axis_value) => quote! {
            let (input, axis): ([i64; #len], i64) = (#input, #axis_value);
            assert!(
                (-1..1).contains(&axis),
                "CumProd: axis {} is out of range for a rank-1 input", axis
            );
        },
        None => quote! { let input: [i64; #len] = #input; },
    };
    quote! {
        let #output: [i64; #len] = {
            #read_inputs
            let mut output = [0i64; #len];
            let mut product = 1i64;
            for i in #indices {
                #step
            }
            output
        };
    }
}

/// Generate code for runtime axis. A negative axis counts from the end, so it is normalized
/// against the input rank (known at codegen time) before it reaches shape, flip and narrow.
fn generate_runtime_cumprod(
    input: &TokenStream,
    output: &syn::Ident,
    axis_value: TokenStream,
    rank: usize,
    exclusive: bool,
    reverse: bool,
) -> TokenStream {
    let rank = proc_macro2::Literal::i64_unsuffixed(rank as i64);
    // Both graph values are read before any temporary is declared, so none can shadow them
    let read_inputs = quote! {
        let (input, axis) = (#input, #axis_value);
        assert!(
            (-#rank..#rank).contains(&axis),
            "CumProd: axis {} is out of range for a rank-{} input", axis, #rank
        );
        let axis = axis.rem_euclid(#rank) as usize;
    };

    let body = match (exclusive, reverse) {
        (false, false) => quote! { input.cumprod(axis) },
        (false, true) => quote! { input.flip([axis]).cumprod(axis).flip([axis]) },
        // Exclusive: output[i] = product(input[0..i]), shifted right with a leading one
        (true, false) => quote! {
            let cumprod_result = input.cumprod(axis);
            let dim_size = cumprod_result.shape()[axis];
            if dim_size <= 1 {
                // For empty or single element, exclusive cumprod is all ones
                cumprod_result.ones_like()
            } else {
                let sliced = cumprod_result.narrow(axis, 0, dim_size - 1);
                let ones = sliced.ones_like().narrow(axis, 0, 1);
                Tensor::cat([ones, sliced].into(), axis)
            }
        },
        // Exclusive + Reverse: output[i] = product(input[i+1..n]), with a trailing one
        (true, true) => quote! {
            let cumprod_back = input.flip([axis]).cumprod(axis).flip([axis]);
            let dim_size = cumprod_back.shape()[axis];
            if dim_size <= 1 {
                // For empty or single element, exclusive reverse cumprod is all ones
                cumprod_back.ones_like()
            } else {
                let sliced = cumprod_back.narrow(axis, 1, dim_size - 1);
                let ones = sliced.ones_like().narrow(axis, 0, 1);
                Tensor::cat([sliced, ones].into(), axis)
            }
        },
    };

    quote! {
        let #output = {
            #read_inputs
            #body
        };
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::CumProdAxis;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::cumprod::{CumProdConfig, CumProdNode, CumProdNodeBuilder};
    use onnx_ir::ir::RuntimeInputRef;

    fn create_cumprod_node(
        name: &str,
        axis: usize,
        exclusive: bool,
        reverse: bool,
        rank: usize,
    ) -> CumProdNode {
        let config = CumProdConfig::new(CumProdAxis::Static(axis), exclusive, reverse);

        CumProdNodeBuilder::new(name)
            .input_tensor("input", rank, DType::F32)
            .input_tensor("axis", 0, DType::I64)
            .output_tensor("output", rank, DType::F32)
            .config(config)
            .build()
    }

    fn create_runtime_cumprod_node(
        name: &str,
        exclusive: bool,
        reverse: bool,
        rank: usize,
    ) -> CumProdNode {
        let config = CumProdConfig::new(
            CumProdAxis::Runtime(RuntimeInputRef::new("axis".to_string(), 1)),
            exclusive,
            reverse,
        );

        CumProdNodeBuilder::new(name)
            .input_tensor("input", rank, DType::F32)
            .input_shape("axis", 1) // Shape type for runtime axis
            .output_tensor("output", rank, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_cumprod_default() {
        let node = create_cumprod_node("cumprod1", 0, false, false, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<1>) -> Tensor<1> {
            let output = input.cumprod(0);
            output
        }
        ");
    }

    #[test]
    fn test_cumprod_axis_1() {
        let node = create_cumprod_node("cumprod1", 1, false, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input.cumprod(1);
            output
        }
        ");
    }

    #[test]
    fn test_cumprod_reverse() {
        let node = create_cumprod_node("cumprod1", 0, false, true, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<1>) -> Tensor<1> {
            let output = input.flip([0]).cumprod(0).flip([0]);
            output
        }
        ");
    }

    #[test]
    fn test_cumprod_exclusive() {
        let node = create_cumprod_node("cumprod1", 0, true, false, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<1>) -> Tensor<1> {
            let output = {
                let cumprod_result = input.cumprod(0);
                let shape = cumprod_result.shape();
                let dim_size = shape[0];
                if dim_size <= 1 {
                    cumprod_result.ones_like()
                } else {
                    let sliced = cumprod_result.narrow(0, 0, dim_size - 1);
                    let ones = sliced.ones_like().narrow(0, 0, 1);
                    Tensor::cat([ones, sliced].into(), 0)
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_cumprod_exclusive_reverse() {
        let node = create_cumprod_node("cumprod1", 0, true, true, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<1>) -> Tensor<1> {
            let output = {
                let flipped = input.flip([0]);
                let cumprod_result = flipped.cumprod(0);
                let cumprod_back = cumprod_result.flip([0]);
                let shape = cumprod_back.shape();
                let dim_size = shape[0];
                if dim_size <= 1 {
                    cumprod_back.ones_like()
                } else {
                    let sliced = cumprod_back.narrow(0, 1, dim_size - 1);
                    let ones = sliced.ones_like().narrow(0, 0, 1);
                    Tensor::cat([sliced, ones].into(), 0)
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_cumprod_reverse_axis_1() {
        let node = create_cumprod_node("cumprod1", 1, false, true, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input.flip([1]).cumprod(1).flip([1]);
            output
        }
        ");
    }

    #[test]
    fn test_cumprod_exclusive_axis_1() {
        let node = create_cumprod_node("cumprod1", 1, true, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = {
                let cumprod_result = input.cumprod(1);
                let shape = cumprod_result.shape();
                let dim_size = shape[1];
                if dim_size <= 1 {
                    cumprod_result.ones_like()
                } else {
                    let sliced = cumprod_result.narrow(1, 0, dim_size - 1);
                    let ones = sliced.ones_like().narrow(1, 0, 1);
                    Tensor::cat([ones, sliced].into(), 1)
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_cumprod_runtime_axis_rank2() {
        let node = create_runtime_cumprod_node("cumprod1", false, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<2>, axis: [i64; 1]) -> Tensor<2> {
            let output = {
                let (input, axis) = (input, axis[0] as i64);
                assert!(
                    (- 2..2).contains(& axis),
                    "CumProd: axis {} is out of range for a rank-{} input", axis, 2
                );
                let axis = axis.rem_euclid(2) as usize;
                input.cumprod(axis)
            };
            output
        }
        "#);
    }

    #[test]
    fn test_cumprod_runtime_axis_rank3() {
        let node = create_runtime_cumprod_node("cumprod1", false, false, 3);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<3>, axis: [i64; 1]) -> Tensor<3> {
            let output = {
                let (input, axis) = (input, axis[0] as i64);
                assert!(
                    (- 3..3).contains(& axis),
                    "CumProd: axis {} is out of range for a rank-{} input", axis, 3
                );
                let axis = axis.rem_euclid(3) as usize;
                input.cumprod(axis)
            };
            output
        }
        "#);
    }

    #[test]
    fn test_cumprod_runtime_axis_exclusive() {
        let node = create_runtime_cumprod_node("cumprod1", true, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<2>, axis: [i64; 1]) -> Tensor<2> {
            let output = {
                let (input, axis) = (input, axis[0] as i64);
                assert!(
                    (- 2..2).contains(& axis),
                    "CumProd: axis {} is out of range for a rank-{} input", axis, 2
                );
                let axis = axis.rem_euclid(2) as usize;
                let cumprod_result = input.cumprod(axis);
                let dim_size = cumprod_result.shape()[axis];
                if dim_size <= 1 {
                    cumprod_result.ones_like()
                } else {
                    let sliced = cumprod_result.narrow(axis, 0, dim_size - 1);
                    let ones = sliced.ones_like().narrow(axis, 0, 1);
                    Tensor::cat([ones, sliced].into(), axis)
                }
            };
            output
        }
        "#);
    }

    #[test]
    fn test_cumprod_runtime_axis_reverse() {
        let node = create_runtime_cumprod_node("cumprod1", false, true, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<2>, axis: [i64; 1]) -> Tensor<2> {
            let output = {
                let (input, axis) = (input, axis[0] as i64);
                assert!(
                    (- 2..2).contains(& axis),
                    "CumProd: axis {} is out of range for a rank-{} input", axis, 2
                );
                let axis = axis.rem_euclid(2) as usize;
                input.flip([axis]).cumprod(axis).flip([axis])
            };
            output
        }
        "#);
    }

    fn create_scalar_axis_node(input: &str, axis: &str) -> CumProdNode {
        let config = CumProdConfig::new(
            CumProdAxis::Runtime(RuntimeInputRef::new(axis.to_string(), 1)),
            true,
            true,
        );
        CumProdNodeBuilder::new("cumprod1")
            .input_tensor(input, 2, DType::F32)
            .input_scalar(axis, DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_cumprod_runtime_scalar_axis_exclusive_reverse() {
        // The official tests pass the axis as a scalar graph input
        let node = create_scalar_axis_node("input", "axis");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<2>, axis: i64) -> Tensor<2> {
            let output = {
                let (input, axis) = (input, axis);
                assert!(
                    (- 2..2).contains(& axis),
                    "CumProd: axis {} is out of range for a rank-{} input", axis, 2
                );
                let axis = axis.rem_euclid(2) as usize;
                let cumprod_back = input.flip([axis]).cumprod(axis).flip([axis]);
                let dim_size = cumprod_back.shape()[axis];
                if dim_size <= 1 {
                    cumprod_back.ones_like()
                } else {
                    let sliced = cumprod_back.narrow(axis, 1, dim_size - 1);
                    let ones = sliced.ones_like().narrow(axis, 0, 1);
                    Tensor::cat([sliced, ones].into(), axis)
                }
            };
            output
        }
        "#);
    }

    #[test]
    fn inputs_named_like_its_temporaries_are_accepted() {
        // Both inputs are read before any temporary is declared
        let temporaries = [
            "input",
            "axis",
            "cumprod_back",
            "cumprod_result",
            "dim_size",
            "sliced",
            "ones",
        ];
        for temporary in temporaries {
            let as_input = create_scalar_axis_node(temporary, "axis_value");
            assert_eq!(
                shadow_check_result(&as_input),
                Ok(()),
                "input named {temporary}"
            );
            let as_axis = create_scalar_axis_node("data", temporary);
            assert_eq!(
                shadow_check_result(&as_axis),
                Ok(()),
                "axis named {temporary}"
            );
        }
    }

    #[test]
    fn test_cumprod_runtime_scalar_tensor_axis() {
        // A computed axis (e.g. ArgMax with keepdims=0) is a scalar on the device
        let config = CumProdConfig::new(
            CumProdAxis::Runtime(RuntimeInputRef::new("axis".to_string(), 1)),
            false,
            false,
        );
        let node = CumProdNodeBuilder::new("cumprod1")
            .input_tensor("input", 2, DType::F32)
            .input_scalar_tensor("axis", DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<2>, axis: Tensor<1, Int>) -> Tensor<2> {
            let output = {
                let (input, axis) = (input, (axis).into_scalar::<i64>());
                assert!(
                    (- 2..2).contains(& axis),
                    "CumProd: axis {} is out of range for a rank-{} input", axis, 2
                );
                let axis = axis.rem_euclid(2) as usize;
                input.cumprod(axis)
            };
            output
        }
        "#);
    }

    fn create_shape_node(input: &str, axis: Option<&str>) -> CumProdNode {
        let axis_config = match axis {
            Some(axis) => CumProdAxis::Runtime(RuntimeInputRef::new(axis.to_string(), 1)),
            None => CumProdAxis::Static(0),
        };
        let builder = CumProdNodeBuilder::new("cumprod1").input_shape(input, 3);
        let builder = match axis {
            Some(axis) => builder.input_scalar(axis, DType::I64),
            None => builder,
        };
        builder
            .output_shape("output", 3)
            .config(CumProdConfig::new(axis_config, true, true))
            .build()
    }

    #[test]
    fn test_cumprod_shape_input_static_axis() {
        // A Shape input is a host [i64; N] array, so the product runs on the host
        let code = codegen_forward_default(&create_shape_node("shape", None));
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape: [i64; 3]) -> [i64; 3] {
            let output: [i64; 3] = {
                let input: [i64; 3] = shape;
                let mut output = [0i64; 3];
                let mut product = 1i64;
                for i in (0..3).rev() {
                    output[i] = product;
                    product = product.wrapping_mul(input[i]);
                }
                output
            };
            output
        }
        ");
    }

    #[test]
    fn test_cumprod_shape_input_runtime_axis() {
        let code = codegen_forward_default(&create_shape_node("shape", Some("axis")));
        assert_snapshot!(code, @r#"
        pub fn forward(&self, shape: [i64; 3], axis: i64) -> [i64; 3] {
            let output: [i64; 3] = {
                let (input, axis): ([i64; 3], i64) = (shape, axis);
                assert!(
                    (- 1..1).contains(& axis),
                    "CumProd: axis {} is out of range for a rank-1 input", axis
                );
                let mut output = [0i64; 3];
                let mut product = 1i64;
                for i in (0..3).rev() {
                    output[i] = product;
                    product = product.wrapping_mul(input[i]);
                }
                output
            };
            output
        }
        "#);
    }

    #[test]
    fn shape_inputs_named_like_its_temporaries_are_accepted() {
        for temporary in ["input", "axis", "output", "product", "i"] {
            let as_input = create_shape_node(temporary, Some("axis_value"));
            assert_eq!(
                shadow_check_result(&as_input),
                Ok(()),
                "input named {temporary}"
            );
            let as_axis = create_shape_node("data", Some(temporary));
            assert_eq!(
                shadow_check_result(&as_axis),
                Ok(()),
                "axis named {temporary}"
            );
        }
    }
}
