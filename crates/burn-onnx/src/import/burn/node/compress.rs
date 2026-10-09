use super::prelude::*;

impl NodeCodegen for onnx_ir::compress::CompressNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = &self.inputs[0];
        let condition_arg = &self.inputs[1];
        let output = arg_to_ident(&self.outputs[0]);

        let input = scope.arg(input_arg);
        let condition = scope.arg(condition_arg);

        // `argwhere` returns `[num_selected, rank]`; for a rank 1 condition that
        // is `[num_selected, 1]`, so squeeze it down to the flat index vector
        // that `select` expects. `reshape([-1])` rather than `squeeze` because
        // the length is a runtime value.
        let indices = quote! { #condition.argwhere().reshape([-1]) };

        match self.config.axis {
            Some(axis) => {
                let axis = axis.to_tokens();
                quote! {
                    let #output = #input.select(#axis, #indices);
                }
            }
            None => quote! {
                let #output = #input.reshape([-1]).select(0, #indices);
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::compress::{CompressConfig, CompressNodeBuilder};

    fn builder(axis: Option<i64>) -> CompressNodeBuilder {
        CompressNodeBuilder::new("compress1")
            .input_tensor("input", 2, DType::F32)
            .input_tensor("condition", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 2, DType::F32)
            .config(CompressConfig { axis })
    }

    #[test]
    fn test_compress_axis_0() {
        let code = codegen_forward_default(&builder(Some(0)).build());
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>, condition: Tensor<1, Bool>) -> Tensor<2> {
            let output = input.select(0, condition.argwhere().reshape([-1]));
            output
        }
        ");
    }

    #[test]
    fn test_compress_axis_1() {
        let code = codegen_forward_default(&builder(Some(1)).build());
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>, condition: Tensor<1, Bool>) -> Tensor<2> {
            let output = input.select(1, condition.argwhere().reshape([-1]));
            output
        }
        ");
    }

    #[test]
    fn test_compress_negative_axis() {
        let code = codegen_forward_default(&builder(Some(-1)).build());
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>, condition: Tensor<1, Bool>) -> Tensor<2> {
            let output = input.select(-1, condition.argwhere().reshape([-1]));
            output
        }
        ");
    }

    #[test]
    fn test_compress_default_axis_flattens() {
        let code = codegen_forward_default(&builder(None).build());
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>, condition: Tensor<1, Bool>) -> Tensor<2> {
            let output = input.reshape([-1]).select(0, condition.argwhere().reshape([-1]));
            output
        }
        ");
    }

    #[test]
    fn test_compress_int_input() {
        let node = CompressNodeBuilder::new("compress2")
            .input_tensor("input", 2, DType::I64)
            .input_tensor("condition", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 2, DType::I64)
            .config(CompressConfig { axis: Some(1) })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<2, Int>,
            condition: Tensor<1, Bool>,
        ) -> Tensor<2, Int> {
            let output = input.select(1, condition.argwhere().reshape([-1]));
            output
        }
        ");
    }

    #[test]
    fn test_compress_bool_input() {
        let node = CompressNodeBuilder::new("compress3")
            .input_tensor("input", 2, DType::Bool(BoolStore::Native))
            .input_tensor("condition", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 2, DType::Bool(BoolStore::Native))
            .config(CompressConfig { axis: Some(0) })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<2, Bool>,
            condition: Tensor<1, Bool>,
        ) -> Tensor<2, Bool> {
            let output = input.select(0, condition.argwhere().reshape([-1]));
            output
        }
        ");
    }
}
