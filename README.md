# llama-qwen36-vulkan

Run `llama.cpp` in Docker with Vulkan acceleration for AMD GPUs, serving Qwen3.6 27B
with MTP (multi-token prediction) speculative decoding.

This setup targets a clean, headless Ubuntu server with Docker and an AMD Radeon
RX 7900 XTX (24 GB). Everything below was measured on that machine.

## Goals

- Keep the host system clean
- Avoid compiling `llama.cpp` on the host
- Use Vulkan GPU acceleration through `/dev/dri`
- Run `llama.cpp` in server mode
- Use server-level sampling defaults
- Support large context inference for coding agents and terminal clients

## Models

Two mutually exclusive profiles, both on port 8080:

| Profile | Model | Quant | File size | Context |
|---|---|---|---|---|
| `stock` (default) | [unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) | `UD-Q4_K_XL` | 17.9 GB | 163840 |
| `fable` | [DavidAU/Qwen3.6-27B-Fable-Fusion-711-...-MTP-GGUF](https://huggingface.co/DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF) | `MTP-Q4_K_M` | 18.5 GB | 147456 |

The `fable` model is an uncensored multi-stage fine-tune of Qwen3.6-27B. Its repo has
no `UD-Q4_K_XL`, so the config pins `MTP-Q4_K_M` — the closest match in both size and
quality that still fits with all layers on the GPU. That repo mixes MTP and non-MTP
files under non-standard quant names, so the exact filename is pinned via `--hf-file`
rather than a `:QUANT` suffix.

Both profiles use MTP builds. Standard Qwen3.6 GGUFs do not contain the prediction
heads and will not work with `--spec-type draft-mtp`.

## Switching models

Controlled by `COMPOSE_PROFILES` in `.env` (defaults to `stock`):

```bash
docker compose up -d                      # stock Unsloth build
docker compose --profile fable up -d      # DavidAU Fable-Fusion fine-tune
```

Both services are profiled deliberately, so a stray `docker compose up` can never try
to load two 18 GB models onto the same 24 GB card. Stop one before starting the other.

The two profiles use different `--alias` values (`qwen3.6-27b` and `qwen3.6-27b-fable`)
so `/v1/models` tells you which is live. `llama-server` does not enforce the `model`
field on requests, so existing client configs keep working across a swap.

## Hardware target

- Ubuntu 26.04 minimal server (headless, no desktop or window manager)
- AMD Radeon RX 7900 XTX, Navi 31, 24560 MiB VRAM
- Mesa RADV Vulkan driver
- Docker Compose
- `ghcr.io/ggml-org/llama.cpp:server-vulkan`

MTP support was merged upstream in [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)
(May 2026), so the official image is used directly — no third-party build needed.

## Host prerequisites

Install Docker and make sure your user can access the GPU render device.

Check that `/dev/dri` exists:

```bash
ls -l /dev/dri
```

Expected output should include something like:

```text
card1
renderD128
```

Add your user to the `render` and `video` groups:

```bash
sudo usermod -aG render,video "$USER"
```

Then log out and back in, or reboot.

Verify Vulkan sees the AMD GPU:

```bash
sudo apt update
sudo apt install vulkan-tools
vulkaninfo --summary
```

You want to see your AMD GPU listed as a discrete GPU, for example:

```text
deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
deviceName = AMD Radeon RX 7900 XTX
driverName = radv
```

It is normal for `llvmpipe` to also appear as a CPU fallback device.

## Important: disable GPU runtime power management

On a headless box the GPU runtime-suspends when idle, and **amdgpu evicts all VRAM to
system RAM when it does**. The model then has to be paged back over PCIe on the next
request. Measured impact with the model loaded and idle for ~75 s:

| `power/control` | Idle VRAM | Idle GTT (system RAM) | First request after idle | Generation |
|---|---|---|---|---|
| `auto` (default) | 26 MiB | 15507 MiB | 10.0 s | 68.0 tok/s |
| `on` | 22768 MiB | 361 MiB | **4.5 s** | **72.5 tok/s** |

This is the single highest-impact tuning change for a bursty agent workload. Apply it
for the current boot:

```bash
echo on | sudo tee /sys/class/drm/card0/device/power/control
```

Confirm it held:

```bash
cat /sys/class/drm/card0/device/power/runtime_status   # want: active
```

To make it persist across reboots, add a udev rule (adjust the PCI slot if yours
differs — check with `lspci -nn | grep -i vga`):

```bash
echo 'ACTION=="add", SUBSYSTEM=="pci", KERNEL=="0000:03:00.0", ATTR{power/control}="on"' \
  | sudo tee /etc/udev/rules.d/99-amdgpu-no-runtime-pm.rules
```

This keeps the GPU powered up continuously, which costs idle watts. That is the right
trade for a dedicated inference server, but not for a laptop or workstation.

## Environment

Create a `.env` file:

```bash
cp env.sample .env
```

Then edit it:

```env
HF_TOKEN=hf_your_read_only_token_here
COMPOSE_PROFILES=stock
```

A Hugging Face token may not be required for public models, but it is useful for more
reliable downloads and for gated/private repos. Use a read-only token.

## Start the server

For the first run, start in the foreground so you can watch the download and load:

```bash
docker compose up
```

Once everything works, run in the background:

```bash
docker compose up -d
docker logs -f llama-qwen36
```

## Test the API

Check server properties:

```bash
curl -s http://localhost:8080/props | jq
```

Send a chat completion request, and check MTP is actually engaging:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "stream": false
  }' | jq '.timings | {
      predicted_per_second,
      draft_n,
      draft_n_accepted,
      acceptance: (.draft_n_accepted / .draft_n * 100)
    }'
```

## Measured performance

RX 7900 XTX, `fable` profile (`MTP-Q4_K_M`) unless noted, short prompt, warm.

### MTP draft length

`--spec-draft-n-max` is the main MTP knob. Measured at ctx 131072:

| Setting | Generation | Draft acceptance |
|---|---|---|
| `--spec-type none` | 36.9 tok/s | — |
| `n-max 2` | **65.1 tok/s** | 69.8% |
| `n-max 3` | 65.0 tok/s | 58.4% |
| `n-max 4` | 37.6 tok/s | 51.3% |

MTP is worth **~1.8x** here. `n-max 2` wins: 3 is no faster and 4 collapses back to
non-MTP speed because acceptance falls off. Note the upstream *default* is 3, so this
is set explicitly.

Acceptance is prompt-dependent — 70-87% observed across different prompts. DavidAU's
guidance is that below 50% you should switch to the non-MTP quants, since the draft
overhead stops paying for itself. Keep `temperature <= 1.0`; higher temperatures
degrade acceptance.

### Context size and the layer-eviction cliff

`--gpu-layers auto` with `--fit on` silently moves layers to CPU rather than failing
when a context does not fit. One evicted layer costs a PCIe round-trip per token:

| Profile | Context | Layers on GPU | Idle VRAM | Generation |
|---|---|---|---|---|
| stock | 147456 | 66/66 | 22843 MiB (93.0%) | 76.3 tok/s |
| stock | **163840** | **66/66** | **23483 MiB (95.6%)** | **77.5 tok/s** |
| stock | 180224 | 64/66 | 23608 MiB (96.1%) | 44.6 tok/s |
| fable | **147456** | **66/66** | **23409 MiB (95.3%)** | **71.9 tok/s** |
| fable | 155648 | 66/66 | 23729 MiB (96.6%) | 69.0 tok/s |
| fable | 163840 | 65/66 | 23847 MiB (97.1%) | 49.1 tok/s |

Each profile ships with the largest context that still keeps 66/66 layers resident.
**More VRAM used is not better** — the cliff matters far more than the last GB.

If you change quant or context, always confirm full residency in the load log:

```bash
docker logs llama-qwen36 2>&1 | grep "offloaded"
# want: offloaded 66/66 layers to GPU
```

Note the default log level hides this line; add `-lv 4` to the command to see the
fit decisions and buffer sizes.

### Long context

At the shipped settings, a 107k-token prompt on the `fable` profile:

```text
prefill      421.8 tok/s   (762 tok/s on short prompts)
generation    35.5 tok/s   (vs 71.9 tok/s at short context)
peak VRAM     23449 MiB    (+46 MiB over idle — compute buffers barely grow)
```

Generation slows at deep context because attention cost grows with the KV cache, not
because anything spilled. The ~1.1 GB margin left at 147456 is comfortable.

### Batch sizes

Stock defaults (`--batch-size 2048`, `--ubatch-size 512`) are best. Raising them hurts
both, badly:

| Setting | Prefill | Generation |
|---|---|---|
| defaults | **762.5 tok/s** | **65.2 tok/s** |
| `--ubatch-size 1024` | 749.8 tok/s | 39.1 tok/s |
| `--batch-size 4096 --ubatch-size 2048` | 703.7 tok/s | 26.5 tok/s |

The larger buffers take VRAM that the KV cache and weights need. Leave them alone.

## Why 160k context fits in 24 GB

Qwen3.6-27B is a **hybrid attention** model. Only 16 of its 64 layers use full
attention (every 4th); the other 48 use linear attention with a fixed-size recurrent
state that does not grow with context. So the KV cache is roughly a quarter of what a
conventional 27B dense model would need.

Measured buffer breakdown at ctx 131072, `q8_0` K/V:

```text
model weights (Vulkan0)      16949 MiB
KV cache, 16 full-attn layers 4352 MiB   (K 2176 + V 2176)
recurrent state, 64 layers      449 MiB   (fixed, independent of context)
MTP draft KV cache              512 MiB
compute buffers                 260 MiB
```

KV cost scales at roughly **37 MiB per 1000 tokens** of context, counting the MTP
draft cache. Useful for estimating a different context size before trying it.

## Vision is disabled

Qwen3.6 is multimodal and `--hf-repo` auto-downloads the vision projector. Both
profiles pass `--no-mmproj`, which frees roughly 0.9 GB of VRAM for weights and
context. If you want image input, drop that flag and reduce `--ctx-size` to
compensate — otherwise the fitter will evict layers to CPU.

## Sampling defaults

Both profiles set the Qwen3.6 thinking / precise-coding preset:

```text
temperature        = 0.6
top_p              = 0.95
top_k              = 20
min_p              = 0.0
presence_penalty   = 0.0
repeat_penalty     = 1.0
```

DavidAU's recommended settings for the `fable` model match these for coding. For
general/creative use that author suggests `temperature 1.0`; keep it at or below 1.0
so MTP acceptance holds up.

Clients that do not send their own sampling parameters use these server-level
defaults. This is not a hard policy — a client that explicitly sends different values
overrides them. For hard enforcement, put a reverse proxy in front and strip those
fields.

## Thinking mode

Qwen3.6 returns reasoning in a separate `reasoning_content` field. Both profiles pass
`--reasoning-preserve`, which carries reasoning across turns; `llama-server` itself
recommends this when the chat template advertises the capability, and it helps on
iterative coding tasks.

For coding agents it is usually best to keep thinking enabled and configure the client
to hide or collapse `reasoning_content`. Note that a low `max_tokens` can be consumed
entirely by the thinking block, returning empty `content` — give it room, or cap
thinking with `--reasoning-budget N`.

## Monitoring

```bash
sudo apt install radeontop
sudo radeontop
```

For exact VRAM accounting, read the kernel directly — this is what the tables above
use, and it is more reliable than Vulkan's reported free memory:

```bash
echo $(( $(cat /sys/class/drm/card0/device/mem_info_vram_used) / 1048576 )) MiB
echo $(( $(cat /sys/class/drm/card0/device/mem_info_gtt_used)  / 1048576 )) MiB
```

`gtt_used` is the number to watch: a large value means buffers have spilled to system
RAM over PCIe. It should stay a few hundred MiB. Also note that a high host-RAM figure
is expected and mostly harmless — `llama.cpp` mmaps the GGUF, so the model file sits in
reclaimable page cache even when fully offloaded.

## If you run out of VRAM

In order of preference:

1. **Reduce `--ctx-size`.** Cheapest and most predictable, ~37 MiB per 1000 tokens.
2. **Quantize the K cache harder** (`--cache-type-k q4_0`, leave V at `q8_0`).
3. **Quantize both** (`q4_0` / `q4_0`) — more savings, more quality impact.
4. **Drop to a smaller quant.** The `fable` repo has `MTP-Q4_K_S` (17.5 GB) and
   `MTP-IQ4_XS` (17.0 GB) below the shipped `MTP-Q4_K_M`, plus `LOW-MTP-IQ4_XS`
   (15.1 GB) if you need significantly more room.

Always re-check `offloaded 66/66 layers` afterward.

## Security notes

- Use a read-only Hugging Face token
- Be careful exposing port `8080` publicly
- `llama.cpp` server has no authentication by default and defaults to permissive CORS
- Use a firewall, VPN, SSH tunnel, or authenticating reverse proxy for remote access

## License

MIT
