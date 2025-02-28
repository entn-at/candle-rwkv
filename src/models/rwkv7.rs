use candle::{Result, Tensor};
use candle_nn::{
    embedding, layer_norm, linear_no_bias as linear, Embedding, GroupNorm, LayerNorm, Linear,
    Module, VarBuilder,
};

pub use crate::models::rwkv5::{Config, State, Tokenizer};

#[derive(Debug, Clone)]
struct SelfAttention {
    x_r: Tensor,
    x_w: Tensor,
    x_k: Tensor,
    x_v: Tensor,
    x_a: Tensor,
    x_g: Tensor,
    r_k: Tensor,
    w0: Tensor,
    w1: Tensor,
    w2: Tensor,
    a0: Tensor,
    a1: Tensor,
    a2: Tensor,
    g1: Tensor,
    g2: Tensor,
    v0: Option<Tensor>,
    v1: Option<Tensor>,
    v2: Option<Tensor>,
    k_k: Tensor,
    k_a: Tensor,
    receptance: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    ln_x: candle_nn::GroupNorm,
    layer_id: usize,
    // n_attn_heads: usize,
}

impl SelfAttention {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let attn_hidden_size = cfg.attention_hidden_size;

        let receptance = linear(hidden_size, attn_hidden_size, vb.pp("receptance"))?;
        let key = linear(hidden_size, attn_hidden_size, vb.pp("key"))?;
        let value = linear(hidden_size, attn_hidden_size, vb.pp("value"))?;
        let output = linear(attn_hidden_size, hidden_size, vb.pp("output"))?;

        let vb_x = vb.pp("ln_x");
        let ln_x_weight = vb_x.get(hidden_size, "weight")?;
        let ln_x_bias = vb_x.get(hidden_size, "bias")?;

        let ln_x = GroupNorm::new(ln_x_weight, ln_x_bias, hidden_size, 64, 1e-5)?;

        let x_r = vb.get((1, 1, cfg.hidden_size), "x_r")?;
        let x_w = vb.get((1, 1, cfg.hidden_size), "x_w")?;
        let x_k = vb.get((1, 1, cfg.hidden_size), "x_k")?;
        let x_v = vb.get((1, 1, cfg.hidden_size), "x_v")?;
        let x_a = vb.get((1, 1, cfg.hidden_size), "x_a")?;
        let x_g = vb.get((1, 1, cfg.hidden_size), "x_g")?;
        let r_k = vb.get((cfg.hidden_size / 64, 64), "r_k")?;
        let w0 = vb.get((1, 1, cfg.hidden_size), "w0")?;
        let w1 = vb.get((cfg.hidden_size, 64), "w1")?;
        let w2 = vb.get((64, cfg.hidden_size), "w2")?;
        let a0 = vb.get((1, 1, cfg.hidden_size), "a0")?;
        let a1 = vb.get((cfg.hidden_size, 64), "a1")?;
        let a2 = vb.get((64, cfg.hidden_size), "a2")?;
        let g1 = vb.get((cfg.hidden_size, 128), "g1")?;
        let g2 = vb.get((128, cfg.hidden_size), "g2")?;

        let v0 = if layer_id == 0 {
            None
        } else {
            Some(vb.get((1, 1, cfg.hidden_size), "v0")?)
        };
        let v1 = if layer_id == 0 {
            None
        } else {
            Some(vb.get((cfg.hidden_size, 32), "v1")?)
        };
        let v2 = if layer_id == 0 {
            None
        } else {
            Some(vb.get((32, cfg.hidden_size), "v2")?)
        };

        let k_k = vb.get((1, 1, cfg.hidden_size), "k_k")?;
        let k_a = vb.get((1, 1, cfg.hidden_size), "k_a")?;

        Ok(Self {
            key,
            value,
            receptance,
            output,
            ln_x,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            r_k,
            w0,
            w1,
            w2,
            a0,
            a1,
            a2,
            g1,
            g2,
            v0,
            v1,
            v2,
            k_k,
            k_a,
            layer_id,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let xx = state.per_layer[self.layer_id]
            .extract_key_value
            .broadcast_sub(xs)?;

        let (xr, xw, xk, xv, xa, xg) = (
            (xs + &xx * &self.x_r)?,
            (xs + &xx * &self.x_w)?,
            (xs + &xx * &self.x_k)?,
            (xs + &xx * &self.x_v)?,
            (xs + &xx * &self.x_a)?,
            (xs + &xx * &self.x_g)?,
        );
        let r = self.receptance.forward(&xr)?;
        let w = xw
            .broadcast_matmul(&self.w1)?
            .tanh()?
            .broadcast_matmul(&self.w2)?;
        let k = self.key.forward(&xk)?;
        let v = self.value.forward(&xv)?;
        let a = candle_nn::ops::sigmoid(
            &(&self.a0 + xa.broadcast_matmul(&self.a1)?.broadcast_matmul(&self.a2)?)?,
        )?;
        let kk = (k * &self.k_k)?;
        //

        let w = (&self.w0 + w)?;

        Ok(Tensor::zeros(
            (1, 1, 768),
            candle::DType::F32,
            &candle::Device::Cpu,
        )?)
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    x_k: Tensor,
    key: Linear,
    value: Linear,
    layer_id: usize,
}

impl FeedForward {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let int_size = cfg.intermediate_size.unwrap_or(cfg.hidden_size * 4);
        let key = linear(cfg.hidden_size, int_size, vb.pp("key"))?;
        let value = linear(int_size, cfg.hidden_size, vb.pp("value"))?;
        let x_k = vb.get((1, 1, cfg.hidden_size), "x_k")?;
        Ok(Self {
            key,
            value,
            x_k,
            layer_id,
        })
    }

    fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let xx = state.per_layer[self.layer_id]
            .feed_forward
            .broadcast_sub(xs)?;
        let k = (xs + (&xx * &self.x_k)?)?;
        let k = (xs + &xx.broadcast_mul(&self.x_k)?)?;
        let k = (self.key.forward(&k)?).relu()?.powf(2.0)?;
        let xs = self.value.forward(&k)?;
        // state.per_layer[self.layer_id].feed_forward = xs.i((.., xs.dim(1)? - 1))?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct Block {
    pre_ln: Option<LayerNorm>,
    ln1: LayerNorm,
    ln2: LayerNorm,
    attention: SelfAttention,
    feed_forward: FeedForward,
}

impl Block {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln1 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("ln1"))?;
        let ln2 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("ln2"))?;
        let pre_ln = if layer_id == 0 {
            let ln = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("pre_ln"))?;
            Some(ln)
        } else {
            None
        };
        let attention = SelfAttention::new(layer_id, cfg, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(layer_id, cfg, vb.pp("feed_forward"))?;
        Ok(Self {
            pre_ln,
            ln1,
            ln2,
            attention,
            feed_forward,
        })
    }

    fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let xs = match self.pre_ln.as_ref() {
            None => xs.clone(),
            Some(pre_ln) => xs.apply(pre_ln)?,
        };
        let attention = self.attention.forward(&xs.apply(&self.ln1)?, state)?;
        let xs = (xs + attention)?;
        let feed_forward = self.feed_forward.forward(&xs.apply(&self.ln2)?, state)?;
        let xs = (xs + feed_forward)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embeddings: Embedding,
    blocks: Vec<Block>,
    ln_out: LayerNorm,
    head: Linear,
    rescale_every: usize,
    layers_are_rescaled: bool,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("rwkv");
        let embeddings = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embeddings"))?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_b = vb_m.pp("blocks");
        for block_index in 0..cfg.num_hidden_layers {
            let block = Block::new(block_index, cfg, vb_b.pp(block_index))?;
            blocks.push(block)
        }
        let ln_out = layer_norm(cfg.hidden_size, 1e-5, vb_m.pp("ln_out"))?;
        let head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("head"))?;
        Ok(Self {
            embeddings,
            blocks,
            ln_out,
            head,
            rescale_every: cfg.rescale_every,
            layers_are_rescaled: false, // This seem to only happen for the f16/bf16 dtypes.
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let (_b_size, _seq_len) = xs.dims2()?;
        let mut xs = xs.apply(&self.embeddings)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            xs = block.forward(&xs, state)?;
            if self.layers_are_rescaled && (block_idx + 1) % self.rescale_every == 0 {
                xs = (xs / 2.)?
            }
        }
        let xs = xs.apply(&self.ln_out)?.apply(&self.head)?;
        state.pos += 1;
        Ok(xs)
    }
}
