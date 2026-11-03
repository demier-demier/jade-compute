//! Checked, operation-level IR lowered to native MSL or CoreX C++ kernels.
//! This does not parse CUDA C++, PTX or SASS. Every supported semantic is explicit.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum KernelSpec {
    Elementwise {
        len: u32,
        inputs: u32,
        expression: Expr,
    },
    Axpy {
        len: u32,
        alpha: f32,
    },
    Add {
        len: u32,
    },
    Gemm {
        m: u32,
        n: u32,
        k: u32,
        tiled: bool,
    },
    /// Native Metal SIMD-group matrix multiply, FP32 throughout.
    GemmSimd {
        m: u32,
        n: u32,
        k: u32,
    },
    /// Native Apple MPS library operation; emits no standalone device source.
    GemmMps {
        m: u32,
        n: u32,
        k: u32,
    },
    /// Native Metal self-attention stages. Tensor layout is [heads, seq, dim].
    AttentionScores {
        heads: u32,
        seq: u32,
        dim: u32,
        causal: bool,
    },
    AttentionApply {
        heads: u32,
        seq: u32,
        dim: u32,
    },
    AttentionOnline {
        heads: u32,
        seq: u32,
        dim: u32,
        causal: bool,
    },
    RmsNorm {
        rows: u32,
        cols: u32,
        epsilon: f32,
    },
    SwiGlu {
        len: u32,
    },
    Softmax {
        rows: u32,
        cols: u32,
    },
}

/// A small extensible kernel-language frontend, independent of vendor APIs.
/// Loads are element-aligned. Arbitrary addressing, pointers and control flow
/// are deliberately outside this initial language.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Expr {
    Input { binding: u32 },
    Constant { value: f32 },
    Add { left: Box<Expr>, right: Box<Expr> },
    Multiply { left: Box<Expr>, right: Box<Expr> },
    Maximum { left: Box<Expr>, right: Box<Expr> },
    Negate { value: Box<Expr> },
    Exp { value: Box<Expr> },
}
impl Expr {
    fn validate(&self, inputs: u32, depth: usize, nodes: &mut usize) -> Result<()> {
        *nodes += 1;
        ensure!(
            depth <= 32 && *nodes <= 512,
            "expression exceeds compiler complexity limit"
        );
        match self {
            Self::Input { binding } => {
                ensure!(*binding < inputs, "expression input binding out of range")
            }
            Self::Constant { value } => {
                ensure!(value.is_finite(), "expression constant must be finite")
            }
            Self::Add { left, right }
            | Self::Multiply { left, right }
            | Self::Maximum { left, right } => {
                left.validate(inputs, depth + 1, nodes)?;
                right.validate(inputs, depth + 1, nodes)?;
            }
            Self::Negate { value } | Self::Exp { value } => {
                value.validate(inputs, depth + 1, nodes)?
            }
        }
        Ok(())
    }
    fn code(&self, dialect: Dialect) -> String {
        let exp = if matches!(dialect, Dialect::Metal) {
            "exp"
        } else {
            "expf"
        };
        let max = if matches!(dialect, Dialect::Metal) {
            "fmax"
        } else {
            "fmaxf"
        };
        match self {
            Self::Input { binding } => format!("input{binding}[i]"),
            Self::Constant { value } => format!("{value:e}f"),
            Self::Add { left, right } => {
                format!("({}+{})", left.code(dialect), right.code(dialect))
            }
            Self::Multiply { left, right } => {
                format!("({}*{})", left.code(dialect), right.code(dialect))
            }
            Self::Maximum { left, right } => {
                format!("{max}({},{})", left.code(dialect), right.code(dialect))
            }
            Self::Negate { value } => format!("(-{})", value.code(dialect)),
            Self::Exp { value } => format!("{exp}({})", value.code(dialect)),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Launch {
    /// Threadgroup / block counts, not total thread counts.
    pub grid: [u32; 3],
    pub block: [u32; 3],
}
#[derive(Clone, Debug, Serialize)]
pub struct Binding {
    pub name: &'static str,
    pub elements: usize,
    pub write: bool,
}

#[derive(Clone, Copy)]
enum Dialect {
    Metal,
    Corex,
}

impl KernelSpec {
    pub fn requires_simdgroup_matrix(&self) -> bool {
        matches!(self, Self::GemmSimd { .. })
    }
    pub fn requires_simdgroup_ops(&self) -> bool {
        matches!(self, Self::AttentionOnline { .. })
    }
    pub fn is_metal_specific(&self) -> bool {
        matches!(
            self,
            Self::GemmSimd { .. }
                | Self::GemmMps { .. }
                | Self::AttentionScores { .. }
                | Self::AttentionApply { .. }
                | Self::AttentionOnline { .. }
        )
    }
    pub fn name(&self) -> &'static str {
        match self {
            Self::Elementwise { .. } => "elementwise",
            Self::Axpy { .. } => "axpy",
            Self::Add { .. } => "add",
            Self::Gemm { tiled: true, .. } => "gemm_register32",
            Self::Gemm { tiled: false, .. } => "gemm_naive",
            Self::GemmSimd { .. } => "gemm_simdgroup_fp32",
            Self::GemmMps { .. } => "gemm_mps_fp32",
            Self::AttentionScores { .. } => "attention_scores",
            Self::AttentionApply { .. } => "attention_apply",
            Self::AttentionOnline { .. } => "attention_online",
            Self::RmsNorm { .. } => "rms_norm",
            Self::SwiGlu { .. } => "swi_glu",
            Self::Softmax { .. } => "softmax",
        }
    }
    pub fn validate(&self) -> Result<()> {
        fn dim(v: u32) -> Result<()> {
            ensure!(
                v > 0 && v <= i32::MAX as u32,
                "dimension must be in 1..=INT_MAX"
            );
            Ok(())
        }
        fn matrix(x: u32, y: u32) -> Result<()> {
            dim(x)?;
            dim(y)?;
            ensure!(
                u64::from(x) * u64::from(y) <= i32::MAX as u64,
                "tensor exceeds checked 31-bit indexing"
            );
            Ok(())
        }
        if let Self::Elementwise {
            len,
            inputs,
            expression,
        } = self
        {
            dim(*len)?;
            ensure!(
                *inputs > 0 && *inputs <= 8,
                "elementwise input count must be 1..8"
            );
            return expression.validate(*inputs, 0, &mut 0);
        }
        match *self {
            Self::Elementwise { .. } => unreachable!(),
            Self::Axpy { len, alpha } => {
                dim(len)?;
                ensure!(alpha.is_finite(), "alpha must be finite");
            }
            Self::Add { len } | Self::SwiGlu { len } => dim(len)?,
            Self::Gemm { m, n, k, .. } | Self::GemmSimd { m, n, k } | Self::GemmMps { m, n, k } => {
                matrix(m, k)?;
                matrix(k, n)?;
                matrix(m, n)?;
            }
            Self::AttentionScores {
                heads, seq, dim, ..
            }
            | Self::AttentionApply { heads, seq, dim }
            | Self::AttentionOnline {
                heads, seq, dim, ..
            } => {
                ensure!((1..=32).contains(&heads), "attention heads must be 1..=32");
                ensure!(
                    (1..=2048).contains(&seq),
                    "attention sequence must be 1..=2048"
                );
                ensure!(
                    (1..=128).contains(&dim),
                    "attention head dimension must be 1..=128"
                );
                matrix(heads * seq, dim)?;
                matrix(heads * seq, seq)?;
            }
            Self::RmsNorm {
                rows,
                cols,
                epsilon,
            } => {
                matrix(rows, cols)?;
                ensure!(
                    epsilon.is_finite() && epsilon > 0.0,
                    "epsilon must be finite and positive"
                );
            }
            Self::Softmax { rows, cols } => matrix(rows, cols)?,
        }
        Ok(())
    }
    pub fn bindings(&self) -> Result<Vec<Binding>> {
        self.validate()?;
        let b = |name, elements, write| Binding {
            name,
            elements,
            write,
        };
        Ok(match *self {
            Self::Elementwise { len, inputs, .. } => {
                let names = [
                    "input0", "input1", "input2", "input3", "input4", "input5", "input6", "input7",
                ];
                let mut result: Vec<_> = names[..inputs as usize]
                    .iter()
                    .map(|&name| b(name, len as usize, false))
                    .collect();
                result.push(b("out", len as usize, true));
                result
            }
            Self::Axpy { len, .. } | Self::Add { len } => vec![
                b("a", len as usize, false),
                b("b", len as usize, false),
                b("out", len as usize, true),
            ],
            Self::SwiGlu { len } => vec![
                b("gate", len as usize, false),
                b("up", len as usize, false),
                b("out", len as usize, true),
            ],
            Self::Gemm { m, n, k, .. } | Self::GemmSimd { m, n, k } | Self::GemmMps { m, n, k } => {
                vec![
                    b("a", m as usize * k as usize, false),
                    b("b", k as usize * n as usize, false),
                    b("out", m as usize * n as usize, true),
                ]
            }
            Self::AttentionScores {
                heads, seq, dim, ..
            } => vec![
                b("q", heads as usize * seq as usize * dim as usize, false),
                b("k", heads as usize * seq as usize * dim as usize, false),
                b("scores", heads as usize * seq as usize * seq as usize, true),
            ],
            Self::AttentionApply { heads, seq, dim } => vec![
                b(
                    "probabilities",
                    heads as usize * seq as usize * seq as usize,
                    false,
                ),
                b("v", heads as usize * seq as usize * dim as usize, false),
                b("out", heads as usize * seq as usize * dim as usize, true),
            ],
            Self::AttentionOnline {
                heads, seq, dim, ..
            } => vec![
                b("q", heads as usize * seq as usize * dim as usize, false),
                b("k", heads as usize * seq as usize * dim as usize, false),
                b("v", heads as usize * seq as usize * dim as usize, false),
                b("out", heads as usize * seq as usize * dim as usize, true),
            ],
            Self::RmsNorm { rows, cols, .. } => vec![
                b("x", rows as usize * cols as usize, false),
                b("weight", cols as usize, false),
                b("out", rows as usize * cols as usize, true),
            ],
            Self::Softmax { rows, cols } => vec![
                b("x", rows as usize * cols as usize, false),
                b("out", rows as usize * cols as usize, true),
            ],
        })
    }
    pub fn validate_bindings(&self, lengths: &[usize], ids: &[u64]) -> Result<()> {
        let expected = self.bindings()?;
        ensure!(
            lengths.len() == expected.len() && ids.len() == expected.len(),
            "{} expects {} bindings",
            self.name(),
            expected.len()
        );
        for (i, binding) in expected.iter().enumerate() {
            ensure!(
                lengths[i] == binding.elements,
                "{} binding {} expected {} floats, got {}",
                self.name(),
                binding.name,
                binding.elements,
                lengths[i]
            );
            if binding.write {
                for j in 0..ids.len() {
                    ensure!(
                        i == j || ids[i] != ids[j],
                        "output binding must not alias another binding"
                    );
                }
            }
        }
        Ok(())
    }
    pub fn launch(&self) -> Result<Launch> {
        self.validate()?;
        Ok(match *self {
            Self::GemmMps { .. } => {
                anyhow::bail!("MPS library operations own their launch geometry")
            }
            Self::GemmSimd { m, n, k } => {
                let (grid, block) = super::metal_simd::launch(m, n, k);
                Launch { grid, block }
            }
            Self::AttentionOnline {
                heads,
                seq,
                dim,
                causal,
            } => {
                let (grid, block) = super::metal_attention::online_launch(heads, seq, dim, causal);
                Launch { grid, block }
            }
            Self::AttentionScores {
                heads,
                seq,
                dim,
                causal,
            } => {
                let (grid, block) = super::metal_attention::scores_launch(heads, seq, dim, causal);
                Launch { grid, block }
            }
            Self::AttentionApply { heads, seq, dim } => {
                let (grid, block) = super::metal_attention::apply_launch(heads, seq, dim);
                Launch { grid, block }
            }
            Self::Elementwise { len, .. } => Launch {
                grid: [len.div_ceil(256), 1, 1],
                block: [256, 1, 1],
            },
            Self::Axpy { len, .. } | Self::Add { len } | Self::SwiGlu { len } => Launch {
                grid: [len.div_ceil(256), 1, 1],
                block: [256, 1, 1],
            },
            Self::RmsNorm { rows, .. } | Self::Softmax { rows, .. } => Launch {
                grid: [rows, 1, 1],
                block: [256, 1, 1],
            },
            Self::Gemm { m, n, tiled, .. } => {
                let tile = if tiled { 32 } else { 8 };
                Launch {
                    grid: [n.div_ceil(tile), m.div_ceil(tile), 1],
                    block: [8, 8, 1],
                }
            }
        })
    }
    pub fn metal_source(&self) -> Result<String> {
        self.validate()?;
        match *self {
            Self::GemmMps { .. } => {
                anyhow::bail!("MPS GEMM is a library operation, not generated MSL source")
            }
            Self::GemmSimd { m, n, k } => Ok(super::metal_simd::source(m, n, k)),
            Self::AttentionOnline {
                heads,
                seq,
                dim,
                causal,
            } => Ok(super::metal_attention::online_source(
                heads, seq, dim, causal,
            )),
            Self::AttentionScores {
                heads,
                seq,
                dim,
                causal,
            } => Ok(super::metal_attention::scores_source(
                heads, seq, dim, causal,
            )),
            Self::AttentionApply { heads, seq, dim } => {
                Ok(super::metal_attention::apply_source(heads, seq, dim))
            }
            _ => self.source(Dialect::Metal),
        }
    }
    pub fn corex_source(&self) -> Result<String> {
        ensure!(
            !self.is_metal_specific(),
            "{} is a Metal-only operation; CoreX lowering is not implemented",
            self.name()
        );
        self.source(Dialect::Corex)
    }
    fn source(&self, dialect: Dialect) -> Result<String> {
        self.validate()?;
        let bindings = self.bindings()?;
        let (header,signature,prefix,shared,barrier,exponential,rsqrt,maximum,fused) = match dialect {
            Dialect::Metal => ("#include <metal_stdlib>\nusing namespace metal;\n", "kernel void jade_main", "", "threadgroup", "threadgroup_barrier(mem_flags::mem_threadgroup);", "exp", "rsqrt", "fmax", "fma"),
            Dialect::Corex => ("#include <cuda_runtime.h>\n#include <math.h>\n", "__global__ void jade_main", "    const uint3 gid = make_uint3(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y, 0);\n    const uint3 tid = threadIdx;\n    const uint3 group = blockIdx;\n", "__shared__", "__syncthreads();", "expf", "rsqrtf", "fmaxf", "fmaf"),
        };
        let args: Vec<String> = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let qualifier = if b.write { "" } else { "const " };
                match dialect {
                    Dialect::Metal => {
                        format!("device {qualifier}float* {} [[buffer({i})]]", b.name)
                    }
                    Dialect::Corex => format!("{qualifier}float* {}", b.name),
                }
            })
            .collect();
        let mut arguments = args.join(", ");
        if matches!(dialect, Dialect::Metal) {
            arguments.push_str(", uint3 gid [[thread_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]");
        }
        let body = match self {
            Self::GemmMps { .. } | Self::GemmSimd { .. } | Self::AttentionScores { .. } | Self::AttentionApply { .. } | Self::AttentionOnline { .. } => unreachable!("Metal-specific operations use their own checked source emitters"),
            Self::Elementwise {len,expression,..} => format!("    const unsigned i=gid.x;\n    if(i<{len}u) out[i]={};\n",expression.code(dialect)),
            Self::Axpy {len,alpha} => format!("    const unsigned i=gid.x;\n    if(i<{len}u) out[i]={fused}({alpha:e}f,a[i],b[i]);\n"),
            Self::Add {len} => format!("    const unsigned i=gid.x;\n    if(i<{len}u) out[i]=a[i]+b[i];\n"),
            Self::SwiGlu {len} => format!("    const unsigned i=gid.x;\n    if(i<{len}u) {{\n        const float v=gate[i];\n        const float e={exponential}(-fabs(v));\n        const float sigmoid=v>=0.0f?1.0f/(1.0f+e):e/(1.0f+e);\n        out[i]=(v*sigmoid)*up[i];\n    }}\n"),
            Self::RmsNorm {rows:_,cols,epsilon} => format!(r#"    {shared} float partial[256];
    const unsigned row=group.x, lane=tid.x;
    float sum=0.0f;
    for(unsigned c=lane;c<{cols}u;c+=256u) {{ float v=x[row*{cols}u+c]; sum={fused}(v,v,sum); }}
    partial[lane]=sum;
    {barrier}
    for(unsigned step=128u;step>0u;step/=2u) {{
        if(lane<step) partial[lane]+=partial[lane+step];
        {barrier}
    }}
    const float inv={rsqrt}(partial[0]/{cols}.0f+{epsilon:e}f);
    for(unsigned c=lane;c<{cols}u;c+=256u) out[row*{cols}u+c]=x[row*{cols}u+c]*inv*weight[c];
"#),
            Self::Softmax {rows:_,cols} => format!(r#"    {shared} float partial[256];
    const unsigned row=group.x, lane=tid.x;
    float maxv=-INFINITY;
    for(unsigned c=lane;c<{cols}u;c+=256u) maxv={maximum}(maxv,x[row*{cols}u+c]);
    partial[lane]=maxv;
    {barrier}
    for(unsigned step=128u;step>0u;step/=2u) {{
        if(lane<step) partial[lane]={maximum}(partial[lane],partial[lane+step]);
        {barrier}
    }}
    maxv=partial[0];
    {barrier}
    float sum=0.0f;
    for(unsigned c=lane;c<{cols}u;c+=256u) sum+={exponential}(x[row*{cols}u+c]-maxv);
    partial[lane]=sum;
    {barrier}
    for(unsigned step=128u;step>0u;step/=2u) {{
        if(lane<step) partial[lane]+=partial[lane+step];
        {barrier}
    }}
    const float denom=partial[0];
    for(unsigned c=lane;c<{cols}u;c+=256u) out[row*{cols}u+c]={exponential}(x[row*{cols}u+c]-maxv)/denom;
"#),
            Self::Gemm {m,n,k,tiled:false} => format!(r#"    const unsigned row=gid.y,col=gid.x;
    if(row<{m}u && col<{n}u) {{
        float sum=0.0f;
        for(unsigned p=0u;p<{k}u;++p) sum={fused}(a[row*{k}u+p],b[p*{n}u+col],sum);
        out[row*{n}u+col]=sum;
    }}
"#),
            Self::Gemm {m,n,k,tiled:true} => format!(r#"    {shared} float tile_a[512], tile_b[512];
    const unsigned linear_thread=tid.y*8u+tid.x;
    const unsigned tile_row=group.y*32u, tile_col=group.x*32u;
    const unsigned local_row=tid.y*4u, local_col=tid.x*4u;
    float acc[4][4]={{}};
    for(unsigned base=0u;base<{k}u;base+=16u) {{
        for(unsigned load=0u;load<8u;++load) {{
            const unsigned slot=linear_thread+load*64u;
            const unsigned ar=tile_row+slot/16u, ac=base+slot%16u;
            const unsigned br=base+slot/32u, bc=tile_col+slot%32u;
            tile_a[slot]=(ar<{m}u && ac<{k}u)?a[ar*{k}u+ac]:0.0f;
            tile_b[slot]=(br<{k}u && bc<{n}u)?b[br*{n}u+bc]:0.0f;
        }}
        {barrier}
        for(unsigned p=0u;p<16u;++p) {{
            for(unsigned r=0u;r<4u;++r) {{
                const float av=tile_a[(local_row+r)*16u+p];
                for(unsigned c=0u;c<4u;++c) acc[r][c]={fused}(av,tile_b[p*32u+local_col+c],acc[r][c]);
            }}
        }}
        {barrier}
    }}
    for(unsigned r=0u;r<4u;++r) for(unsigned c=0u;c<4u;++c) {{
        const unsigned row=tile_row+local_row+r, col=tile_col+local_col+c;
        if(row<{m}u && col<{n}u) out[row*{n}u+col]=acc[r][c];
    }}
"#),
        };
        let mut source=format!("// Generated by Jade checked operation IR: {}\n{header}{signature}({arguments}) {{\n{prefix}{body}}}\n",self.name());
        if matches!(dialect, Dialect::Corex) {
            let launch = self.launch()?;
            let casts: Vec<_> = bindings
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    format!(
                        "static_cast<{}float*>(buffers[{i}])",
                        if b.write { "" } else { "const " }
                    )
                })
                .collect();
            source.push_str(&format!("extern \"C\" int jade_generated_launch(void* stream, void** buffers) {{\n    jade_main<<<dim3({},{},{}),dim3({},{},{}),0,static_cast<cudaStream_t>(stream)>>>({});\n    return static_cast<int>(cudaGetLastError());\n}}\n",launch.grid[0],launch.grid[1],launch.grid[2],launch.block[0],launch.block[1],launch.block[2],casts.join(",")));
        }
        Ok(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metal_specific_operations_fail_closed_on_other_targets_and_invalid_shapes() {
        let specs = [
            KernelSpec::GemmSimd {
                m: 33,
                n: 17,
                k: 65,
            },
            KernelSpec::AttentionOnline {
                heads: 2,
                seq: 33,
                dim: 17,
                causal: true,
            },
            KernelSpec::AttentionScores {
                heads: 2,
                seq: 33,
                dim: 17,
                causal: true,
            },
            KernelSpec::AttentionApply {
                heads: 2,
                seq: 33,
                dim: 17,
            },
        ];
        for spec in specs {
            assert!(spec.corex_source().is_err());
            assert!(spec
                .metal_source()
                .unwrap()
                .contains("kernel void jade_main"));
            let restored: KernelSpec =
                serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
            assert_eq!(
                restored.metal_source().unwrap(),
                spec.metal_source().unwrap()
            );
            let bindings = spec.bindings().unwrap();
            let lengths: Vec<_> = bindings.iter().map(|b| b.elements).collect();
            let mut ids: Vec<_> = (0..bindings.len() as u64).collect();
            assert!(spec.validate_bindings(&lengths, &ids).is_ok());
            *ids.last_mut().unwrap() = 0;
            assert!(spec.validate_bindings(&lengths, &ids).is_err());
        }
        for (heads, seq, dim) in [
            (0, 1, 1),
            (33, 1, 1),
            (1, 0, 1),
            (1, 2049, 1),
            (1, 1, 0),
            (1, 1, 129),
            (u32::MAX, u32::MAX, u32::MAX),
        ] {
            assert!(KernelSpec::AttentionOnline {
                heads,
                seq,
                dim,
                causal: false
            }
            .metal_source()
            .is_err());
        }
        assert!(KernelSpec::GemmSimd {
            m: u32::MAX,
            n: u32::MAX,
            k: u32::MAX
        }
        .metal_source()
        .is_err());
    }
    #[test]
    fn expression_frontend_rejects_unbound_loads_and_excessive_depth() {
        let invalid = KernelSpec::Elementwise {
            len: 7,
            inputs: 1,
            expression: Expr::Input { binding: 1 },
        };
        assert!(invalid.metal_source().is_err());
        assert!(invalid.corex_source().is_err());
        let mut deep = Expr::Input { binding: 0 };
        for _ in 0..34 {
            deep = Expr::Negate {
                value: Box::new(deep),
            };
        }
        assert!(KernelSpec::Elementwise {
            len: 7,
            inputs: 1,
            expression: deep
        }
        .validate()
        .is_err());
    }
    #[test]
    fn invalid_semantics_and_binding_aliases_are_rejected() {
        assert!(KernelSpec::RmsNorm {
            rows: 1,
            cols: 17,
            epsilon: 0.0
        }
        .validate()
        .is_err());
        assert!(KernelSpec::Axpy {
            len: 1,
            alpha: f32::NAN
        }
        .validate()
        .is_err());
        assert!(KernelSpec::Gemm {
            m: u32::MAX,
            n: 2,
            k: 1,
            tiled: true
        }
        .validate()
        .is_err());
        let spec = KernelSpec::Add { len: 17 };
        assert!(spec.validate_bindings(&[17, 17, 17], &[1, 2, 1]).is_err());
        assert!(spec.validate_bindings(&[17, 16, 17], &[1, 2, 3]).is_err());
        assert!(spec.validate_bindings(&[17, 17, 17], &[1, 1, 3]).is_ok());
    }
}
