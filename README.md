# candle-rwkv

### Supports

- [x] RWKV5 Eagle
- [x] RWKV6 Finch
- [x] RWKV7 Goose

### Examples

If you just want to have a try. Run command below.

```bash
# run rwkv7 (developing)
cargo run --release --example rwkv -- --which "v7-0b4" --prompt "User: why is the sky blue?\n\nAssistant: "

# run rwkv6
cargo run --release --example rwkv -- --which "v6-3b" --prompt "User: 我在深圳，我要去埃及金字塔，我要怎么走?\n\nAssistant: "

# run quantized rwkv6
cargo run --release --example rwkv -- --quantized --which "v6-3b" --prompt "User: 我在深圳，我要去埃及金字塔，我要怎么走?\n\nAssistant: "

# run state-tuned rwkv6
cargo run --release --example rwkv -- --state-tuned --which "v6-3b" --prompt "我在深圳，我要去埃及金字塔，我要怎么走?"

# run quantized state-tuned rwkv6
cargo run --release --example rwkv -- --quantized --state-tuned --which "v6-3b" --prompt "我在深圳，我要去埃及金字塔，我要怎么走?"
```

If you want to use local model file. First, download pth file from [Hugging Face](https://huggingface.co/BlinkDL). Then run command below.

```bash
# convert model file to safetensors
cargo run --release --example convert -- --input ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth

# convert state file to safetensors
cargo run --release --example convert -- --input ./rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.pth

# run state tuned rwkv6
cargo run --release --example rwkv -- --which "v6-1b6" --state-tuned --weight-files ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.safetensors --state-files ./rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.safetensors --prompt "我在深圳，我要去埃及金字塔，我要怎么走?"


# quanzited model

# quantize pth to gguf
cargo run --release --example quantize -- --input ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth
# run quantized rwkv6
cargo run --release --example rwkv -- --quantized --which "v6-1b6" --weight-files ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096-q4k.gguf --prompt "User: 我在深圳，我要去埃及金字塔，我要怎么走?\n\nAssistant: "
```

### Others

All PRs are welcome

Powered by [candle](https://github.com/huggingface/candle)