# llama-qwen36-vulkan

Run `llama.cpp` in Docker with Vulkan acceleration for AMD GPUs.

This setup is designed for a clean Ubuntu server host with Docker, OpenSSH, and an AMD Radeon RX 7900 XTX GPU. It runs the official `llama.cpp` Vulkan server image and serves the Unsloth Qwen3.6 27B GGUF model.

## Goals

- Keep the host system clean
- Avoid compiling `llama.cpp` on the host
- Use Vulkan GPU acceleration through `/dev/dri`
- Run `llama.cpp` in server mode
- Use server-level sampling defaults
- Support large context inference for coding agents and terminal clients

## Model

Default model:

- Source: <https://huggingface.co/unsloth/Qwen3.6-27B-GGUF>
- Quantization: `UD-Q4_K_XL`
- Format: GGUF

## Hardware target

This configuration was tested with:

- Ubuntu 26.04 minimal server
- AMD Radeon RX 7900 XTX, Navi 31
- Mesa RADV Vulkan driver
- Docker Compose
- `ghcr.io/ggml-org/llama.cpp:server-vulkan`

It should also work on other AMD GPUs with working Vulkan support, but VRAM requirements will vary.

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

Verify group membership:

```bash
id
groups
```

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

## Monitoring AMD GPU usage

Install `radeontop`:

```bash
sudo apt install radeontop
```

Run it while the server is loading or generating:

```bash
sudo radeontop
```

This is useful for checking VRAM usage and GPU activity, similar to how `nvidia-smi` is commonly used on NVIDIA systems.

## Environment

Create a `.env` file:

```bash
cp env.sample .env
```

Then edit it:

```env
HF_TOKEN=hf_your_read_only_token_here
```

A Hugging Face token may not be required for public models, but it is useful for more reliable downloads and for gated/private repos.

Use a read-only token.

## Docker Compose

Example `compose.yml`:

```yaml
services:
  llama-qwen36:
    image: ghcr.io/ggml-org/llama.cpp:server-vulkan
    container_name: llama-qwen36
    restart: unless-stopped

    env_file:
      - .env

    devices:
      - /dev/dri:/dev/dri

    ports:
      - "8080:8080"

    volumes:
      - ./models:/models
      - ./hf-cache:/root/.cache/huggingface

    command:
      - --hf-repo
      - unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL
      - --host
      - 0.0.0.0
      - --port
      - "8080"

      # Vulkan / AMD GPU offload
      - --device
      - Vulkan0
      - --gpu-layers
      - auto
      - --fit
      - "on"
      - --fit-target
      - "1024"

      # Runtime config
      - --ctx-size
      - "131072"
      - --flash-attn
      - "on"
      - --cache-type-k
      - q8_0
      - --cache-type-v
      - q8_0

      # Server behavior
      - --parallel
      - "1"
      - --predict
      - "32768"

      # Sampling defaults
      - --temp
      - "0.6"
      - --top-p
      - "0.95"
      - --top-k
      - "20"
      - --min-p
      - "0.0"
      - --presence-penalty
      - "0.0"
      - --repeat-penalty
      - "1.0"

      # Useful for terminal agents / OpenAI-compatible clients
      - --alias
      - qwen3.6-27b
      - --jinja
```

## Start the server

For the first run, start in the foreground so you can watch the logs:

```bash
docker compose up
```

Once everything works, run in the background:

```bash
docker compose up -d
```

Follow logs:

```bash
docker logs -f llama-qwen36
```

## Test the API

Check server properties:

```bash
curl -s http://localhost:8080/props | jq
```

Check the default generation settings:

```bash
curl -s http://localhost:8080/props \
  | jq '.default_generation_settings.params | {
      temperature,
      top_p,
      top_k,
      min_p,
      presence_penalty,
      repeat_penalty,
      n_predict
    }'
```

Send a chat completion request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one sentence."
      }
    ],
    "stream": false
  }' | jq
```

## Sampling defaults

This server config sets the following defaults globally:

```text
temperature        = 0.6
top_p              = 0.95
top_k              = 20
min_p              = 0.0
presence_penalty   = 0.0
repeat_penalty     = 1.0
```

Clients that do not send their own sampling parameters will use these server-level defaults.

Note that this is not a hard security policy. A client that explicitly sends different sampling parameters may override these values. If hard enforcement is needed, place a reverse proxy in front of the server and strip or reject those fields.

## Context and VRAM notes

This setup uses:

```text
context size:       131072
KV cache K type:    q8_0
KV cache V type:    q8_0
flash attention:    enabled
parallel slots:     1
max prediction:     32768
```

On a 24 GB RX 7900 XTX, this configuration leaves limited but usable VRAM headroom.

With `--parallel 1`, observed VRAM usage was around 91% after model load. Without limiting parallelism, llama.cpp initialized multiple slots and VRAM usage was higher.

If you run out of VRAM, try the following in order:

### Option 1: Quantize K cache only

```yaml
- --cache-type-k
- q4_0
- --cache-type-v
- q8_0
```

This saves memory while keeping the value cache higher precision.

### Option 2: Quantize both K and V cache

```yaml
- --cache-type-k
- q4_0
- --cache-type-v
- q4_0
```

This saves more memory but may affect quality more.

### Option 3: Reduce context

```yaml
- --ctx-size
- "65536"
```

## Thinking mode

Qwen thinking models may return reasoning in a separate `reasoning_content` field.

For coding agents, it is often best to keep thinking enabled for quality, but configure the client to hide or collapse `reasoning_content` in the terminal UI.

If your client displays the reasoning content and you do not want that, consider adding a system instruction such as:

```text
Use reasoning when helpful, but do not include reasoning traces in the final answer. Keep terminal output concise.
```

For trivial prompts, Qwen-style models may also respond to `/no_think`, while harder tasks may benefit from `/think`.

## Security notes

- Use a read-only Hugging Face token
- Be careful exposing port `8080` publicly
- `llama.cpp` server does not provide authentication by default
- Use a firewall, VPN, SSH tunnel, or reverse proxy with authentication for remote access

## License

MIT
