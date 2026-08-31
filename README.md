<div align="center">
  <img src="docs/arbuz-mark.svg" width="76" alt="ArbuzDiffusion">
  <h1>ArbuzDiffusion</h1>
  <p><b>Stable Diffusion WebUI Forge &mdash; Neo, squeezed.</b></p>
  <p>
    <a href="https://github.com/Riyozaki/sd-webui-forge-NEO/tree/neo/tests"><img src="https://img.shields.io/badge/tests-passing-16a96a" alt="tests passing"></a>
    <img src="https://img.shields.io/badge/no-torch%20or%20GPU%20needed-16a96a" alt="no torch or GPU needed">
    <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11-16a96a" alt="python">
    <img src="https://img.shields.io/badge/gradio-4.40.0-16a96a" alt="gradio">
  </p>
</div>

<p align="center"><sup>
[ <a href="https://github.com/Haoming02/sd-webui-forge-classic/tree/classic#stable-diffusion-webui-forge---classic">Classic</a> |
<a href="https://github.com/Haoming02/sd-webui-forge-classic/tree/neo">Neo</a> |
<b>ArbuzDiffusion</b> ]
</sup></p>

<p align="center"><img src="html/ui.webp" width=512 alt="UI"></p>
<p align="center"><sub>The screenshot is upstream's and shows the stock Gradio theme &mdash; this fork ships its own, see <a href="#the-look">The look</a>.</sub></p>

<blockquote><i>
<b>Stable Diffusion WebUI Forge</b> is a platform on top of the original <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">Stable Diffusion WebUI</a> by <ins>AUTOMATIC1111</ins>, to make development easier, optimize resource management, speed up inference, and study experimental features.<br>
The name "Forge" is inspired by "Minecraft Forge". This project aims to become the Forge of Stable Diffusion WebUI.<br>
<p align="right">- <b>lllyasviel</b><br>
<sup>(paraphrased)</sup></p>
</i></blockquote>

<br>

"**Neo**" mainly serves as an continuation for the "`latest`" version of Forge, which was built on [Gradio](https://github.com/gradio-app/gradio) `4.40.0` before lllyasviel became too busy... Additionally, this fork is focused on optimization and usability, with the main goal of being the lightest WebUI without any bloatwares.

**ArbuzDiffusion** is that fork with three things pushed further: the sampling
and decode paths are measured and trimmed (see [Optimizations](#optimizations)),
models can be found and pulled from [Civitai](#civitai) without leaving the UI,
and the interface has its own palette instead of the stock Gradio one
(see [The look](#the-look)).

> [!Tip]
> [How to Install](#installation)

<br>

## Features [Sep. 03]
> Most base features of the original [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) should still function

#### New Features

- [X] **Civitai / Civitai Red browser**
    - search, inspect and download models from [civitai.com](https://civitai.com) *and* the age restricted [civitai.red](https://civitai.red)
    - see [Civitai](#civitai) below
- [X] Support [Wan 2.2](https://github.com/Wan-Video/Wan2.2)
    - `txt2img`, `img2img`, `txt2vid`, `img2vid`

> [!Important]
> To export a video, you need to have **[FFmpeg](https://ffmpeg.org/)** installed

- [X] Support [Nunchaku](https://github.com/nunchaku-tech/nunchaku) (`SVDQ`) Models
    - `dev`, `krea`, `kontext`, `t5`
- [X] Support [Flux Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
    - `img2img`, `inpaint`

> [!Note]
> Since the `state_dict` between **Flux-Dev**, **Flux-Krea**, and **Flux-Kontext** are exactly the same, to be properly detected as a **Kontext** model, the model needs to include "`kontext`" in its path, either the file or folder name.

- [X] Support [Chroma](https://huggingface.co/lodestones/Chroma)
    - special thanks: [@croquelois](https://github.com/lllyasviel/stable-diffusion-webui-forge/pull/2925)
- [X] Rewrite Preset System
    - now actually remember the checkpoint/modules selections for each preset
- [X] Support [uv](https://github.com/astral-sh/uv) package manager
    - requires **manually** installing [uv](https://github.com/astral-sh/uv/releases)
    - drastically speed up installation
    - see [Commandline](#by-neo)
- [X] Support [SageAttention](https://github.com/thu-ml/SageAttention), [FlashAttention](https://github.com/Dao-AILab/flash-attention), and fast `fp16_accumulation`
    - see [Commandline](#by-neo)
- [X] Implement RescaleCFG
    - reduce burnt colors; mainly for `v-pred` checkpoints
    - enable in **Settings/UI Alternatives**
- [X] Implement MaHiRo
    - alternative CFG calculation; improve prompt adherence
    - enable in **Settings/UI Alternatives**
- [X] Support loading upscalers in `half` precision
    - speed up; reduce quality
    - enable in **Settings/Upscaling**
- [X] Support running tile composition on GPU
    - enable in **Settings/Upscaling**
- [X] Update `spandrel`
    - support new Upscaler architectures
- [X] Add `pillow-heif` package
    - support `.avif` and `.heif` images

#### TODO

- [ ] Improve Memory Management during Generation
    - currently, even when using the same models you could run in [ComfyUI](https://github.com/comfyanonymous/ComfyUI), you might still get **Out of Memory** error...
- [ ] Support [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)

#### Removed Features

- [X] SD2
- [X] SD3
- [X] Forge Spaces
- [X] Hypernetworks
- [X] CLIP Interrogator
- [X] Deepbooru Interrogator
- [X] Textual Inversion Training
- [X] Most built-in Extensions
- [X] Some built-in Scripts
- [X] Some Samplers
- [X] Sampler in RadioGroup
- [X] Unix `.sh` launch scripts
    - You can still use this WebUI by simply copying a launch script from other working WebUI

#### Optimizations

- [X] Remove the per-step **device synchronisation** in the sampling loop
    - `cond_mark` used to be rebuilt and copied from pageable CPU memory to the GPU on *every* step, which forced a full sync
- [X] Skip the mask accumulators of `calc_cond_uncond_batch` for the common case
    - one condition + one unconditional condition covering the whole latent: six latent-sized allocations and ~8 kernels less per step
- [X] Auto-tune **cuDNN** convolutions while sampling
    - disabled automatically in low VRAM mode
- [X] Live preview is decoded on its own CUDA stream
- [X] Cache the `xformers` flash-attention lookup
    - it was re-done for every attention layer of every VAE tile
- [X] Choose the **SageAttention** kernel (incl. the int8/fp8 `++` variant)
- [X] Scan `models/` with a thread pool
    - startup and every *refresh* are much faster with large checkpoint/LoRA collections
- [X] Richer progress readout: `12/30 · 3.4 s/it · ETA 1:02`
- [X] Configurable **PNG compression level** *(lossless, purely speed vs. file size)*
- [X] Optional **TeaCache**
    - reuses the previous UNet residual when the timestep embedding has barely moved
    - **off by default** - it is an approximation; see [TeaCache](#teacache)
- [X] Optional **CUDA graph** replay of the UNet and optional **TorchInductor** compilation
    - **off by default**; see [UNet acceleration](#unet-acceleration)
- [X] No longer `git` `clone` any repository on fresh install
- [X] Remove unused `cmd_args`
- [X] Remove unused `args_parser`
- [X] Remove unused `shared_options`
- [X] Remove legacy codes
- [X] Fix some typos
- [X] Remove redundant upscaler codes
    - put every upscaler inside the `ESRGAN` folder
- [X] Optimize upscaler logics
- [X] Optimize certain operations in `Spandrel`
- [X] Improve color correction
- [X] Revamp settings
    - improve formatting
    - update descriptions
- [X] Check for Extension updates in parallel
- [X] Move `embeddings` folder into `models` folder
- [X] Disable Refiner by default
    - enable again in **Settings/UI Alternatives**
- [X] Lint & Format
- [X] Update `Pillow`
    - faster image processing
- [X] Update `protobuf`
    - faster `insightface` loading
- [X] Update to latest PyTorch
    - `torch==2.8.0+cu128`
    - `xformers==0.0.32`

> [!Note]
> If your GPU does not support the latest PyTorch, manually [install](#install-older-pytorch) older version of PyTorch

- [X] No longer install `open-clip` twice
- [X] Update some packages to newer versions
- [X] Update recommended Python to `3.11.9`
- [X] many more... :tm:

<br>

## Commandline
> These flags can be added after the `set COMMANDLINE_ARGS=` line in the `webui-user.bat` *(separate each flag with space)*

#### A1111 built-in

- `--xformers`: Install the `xformers` package to speed up generation
- `--port`: Specify a server port to use
    - defaults to `7860`
- `--api`: Enable [API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API) access

<br>

- Once you have successfully launched the WebUI, you can add the following flags to bypass some validation steps in order to improve the Startup time
    - `--skip-prepare-environment`
    - `--skip-install`
    - `--skip-python-version-check`
    - `--skip-torch-cuda-test`
    - `--skip-version-check`

> [!Important]
> Remove them if you are installing an Extension, as those also block Extension from installing requirements

#### by. Forge

- For RTX **30** and above, you can add the following flags to slightly increase the performance; but in rare occurrences, they may cause `OutOfMemory` errors or even crash the WebUI; and in certain configurations, they may even lower the speed instead
    - `--cuda-malloc`
    - `--cuda-stream`
    - `--pin-shared-memory`

- `--forge-ref-a1111-home`: Point to an Automatic1111 installation to load its `models` folders
    - **i.e.** `Stable-diffusion`, `text_encoder`

#### by. Neo

- `--uv`: Replace the `python -m pip` calls with `uv pip` to massively speed up package installation
    - requires **uv** to be installed first *(see [Installation](#installation))*
- `--uv-symlink`: Same as above; but additionally pass `--link-mode symlink` to the commands
    - significantly reduces installation size (`~7 GB` to `~100 MB`)

> [!Important]
> Using `symlink` means it will directly access the packages from the cache folders; refrain from clearing the cache when setting this option

- `--forge-ref-comfy-home`: Point to an ComfyUI installation to load its `models` folders
    - **i.e.** `diffusion_models`, `clip`

- `--model-ref`: Points to a central `models` folder that contains all your models
    - said folder should contain subfolders like `Stable-diffusion`, `Lora`, `VAE`, `ESRGAN`, etc.

> [!Important]
> This simply **replaces** the `models` folder, rather than adding on top of it

- `--sage`: Install the `sageattention` package to speed up generation
    - will also attempt to install `triton` automatically

> [!Note]
> For RTX **50** users, you may need to manually [install](#install-sageattention-2) `sageattention 2` instead

- `--flash`: Install the `flash_attn` package to speed up generation
- `--fast-fp16`: Enable the `allow_fp16_accumulation` option
    - requires PyTorch **2.7.0** +

<br>

## The look

A fork with its own name should have its own face. **ArbuzDiffusion** ships two
Gradio themes built from one palette, and the palette is a watermelon read from
the outside in:

| Ramp | Role | Use |
| - | - | - |
| **bark** | deep rind green | the *neutral* ramp &mdash; every surface carries a hint of it instead of being grey |
| **flesh** | red-pink | the *primary* ramp: the one accent, for Generate, focus rings and active states |
| **rind** | bright green | the *secondary* ramp: progress, success, anything that should feel alive |

| Setting (Settings &rarr; User Interface) | Meaning |
| - | - |
| `Gradio Theme` | `Arbuz Dark` (default) or `Arbuz Light` |
| `Interface density` (Settings &rarr; ArbuzDiffusion) | `Compact` / `Cozy` / `Spacious` &mdash; drives the theme's own spacing and text scales, so every component follows |
| `Show the ArbuzDiffusion header` | the mark and wordmark above the tabs |

Everything hand-written in this fork &mdash; the extra networks cards, the
Civitai tab, the progress bar &mdash; reads the same `--arbuz-*` tokens, so a
single theme change recolours the lot. Both the theme and the density ask for a
UI reload; that is expected.

<br>

## Civitai

A **Civitai** tab lets you find and download models without leaving the WebUI.

Since 15.04.2026 Civitai is split across two sites, and both are supported:

| Site | Content |
| - | - |
| [civitai.com](https://civitai.com) | SFW |
| [civitai.red](https://civitai.red) | age restricted |

- search by name/tag/creator, and filter by **type** (`Checkpoint`, `LORA`, `LoCon`, `DoRA`, `TextualInversion`, `Controlnet`, `VAE`, ...), **base model** (`SDXL 1.0`, `Flux.1 D`, `Pony`, `Illustrious`, ...), **sort order** and **period**
- *Open URL / hash*: paste a Civitai link, a model id, or the `AutoV2`/`SHA256` hash of a local file to jump straight to it
- downloads run in the background with **live progress** (bytes, speed, ETA), support **resume** and use multiple connections
- every download is written into the matching folder under `models/` and gets a `.json`
  sidecar plus a preview image, so it shows up in **Extra Networks** with a thumbnail,
  its description and its trigger words
- trigger words can be copied into the positive prompt with one click

**Updates for installed LoRAs** (in the tab, collapsed by default) answers "is one
of mine out of date?" without touching the disk: the AutoV2 value Forge already
keeps for every network is the same one Civitai indexes, so one lookup plus one
look at the model's version list is enough. The check runs in a thread, two
throttled requests per network, and every row can pull the new version into the
folder the old one lives in. Networks Forge has not hashed yet are skipped and
counted rather than hashed &mdash; an update check must not read gigabytes.

The sidecar next to a downloaded model records `civitai.hashes`, which is what the
LoRA scanner reads: a model that came from Civitai is never hashed again.

API keys (needed for models that require an account) are set in
**Settings &rarr; Neo Optimizations &rarr; Civitai**; each site needs its own key.

<br>

## TeaCache

*Settings &rarr; Neo Optimizations &rarr; Performance*

TeaCache skips UNet evaluations whose result would barely change: while the
timestep embedding keeps moving smoothly, the denoiser output of a step is
reconstructed as `x + previous_residual` instead of being computed again.
Typical skip rates are 20-50 % of the steps.

> [!Warning]
> **This is an approximation.** The default threshold of `0` leaves every result
> bit-exact. Higher values are faster and progressively less faithful; the console
> prints the achieved skip rate (`TeaCache: skipped 12/30 steps (40%)`) after every
> generation so the threshold can be tuned from evidence rather than from hope.

| Setting | Meaning |
| - | - |
| `TeaCache threshold` | `0` disables it; start around `0.05` - `0.15` |
| `TeaCache warmup` | fraction of the first steps that are always computed |

Two safety properties worth knowing: the cache resets at the beginning and the
end of every sampling call (so a hires pass or a new batch can never reuse a
stale residual), and when the conditional/unconditional pass has to be split
into two forward passes only the first one is cached - the two halves can never
be mixed up.

<br>

## UNet acceleration

*Settings &rarr; Neo Optimizations &rarr; Performance*

Two experiments that leave the maths alone - apart from fp16 rounding, the
results are the same numbers you would get without them.

**CUDA graph replay** records a single UNet forward and replays it for every
following step.  Sampling calls the denoiser with identical shapes each step, so
the per-step launch overhead of hundreds of kernels is replaced by one replay.
It is the most useful on small/fast models (SD1.5) and at high step counts, and
the least useful where a single step already takes hundreds of milliseconds.

> [!Warning]
> A recording keeps a private memory pool reserved until the shape changes, so
> on an 8 GB card try it at a modest resolution first.

| Setting | Meaning |
| - | - |
| `Replay the UNet with CUDA graphs` | off by default |

It refuses to record whenever it cannot be sure that replaying is correct:

- a ControlNet is active (its tensors change every step)
- gradients are enabled, or something else is already capturing
- an extension put a tensor into `transformer_options`
- the model has no ordinary parameters (quantised models)

and before a recording is used it is **compared against the normal result**.  If
the replayed output differs by more than 2 % of the peak value, replay is
switched off for the rest of the session and the console says so.  The same
happens when recording runs out of memory.  A recording is also thrown away when
the weights change - Forge applies LoRAs by replacing parameter objects, and a
recording made before that would keep pointing at the old storage.

**TorchInductor compilation** hands the UNet forward to `torch.compile`.  The
first generation after enabling it takes minutes while the model is compiled,
which is why it happens on the first UNet call and not while loading.  Errors
inside the compiler fall back to the normal UNet instead of breaking a
generation.

| Setting | Meaning |
| - | - |
| `disabled` | the default |
| `dynamic shapes` | compiled once, works for every resolution and prompt length |
| `static shapes` | better kernels, but recompiles whenever the resolution, batch size or prompt length changes |

If either of them misbehaves badly enough that the settings page will not load,
launch with `--neo-no-cuda-graph` or `--neo-no-compile`.

<br>

## Benchmarking

`benchmark_neo.py` runs the real pipeline several times and reports timings, so the
settings in **Settings &rarr; Neo Optimizations** can be measured instead of guessed at:

```bash
python benchmark_neo.py --steps 20 --width 1024 --height 1024 --runs 3
python benchmark_neo.py --ab neo_cudnn_benchmark
```

While you are just using the UI, every generation prints where its time went:

```
[NEO] 4 image(s) 1024x1024, 20 steps: 5.02s total - sampling 4.31s, decode 0.29s,
post-processing 0.31s, the rest 0.11s
```

That split is the fastest way to read a machine: low VRAM pushes cost into
*sampling*, a slow disk or a high PNG compression level shows up in
*post-processing*, and a decode that silently fell back to tiled VAE (the console
says `Encountered Out of Memory during VAE decoding`) shows up as *decode*
ballooning. Turn the line off in **Settings &rarr; Neo Optimizations**.

> [!Note]
> `--ab` only works for boolean settings. TeaCache's threshold and the compile
> mode are not, so set them in the UI and compare two plain runs.

<br>

## Tests

The suite runs without torch, gradio or numpy &mdash; every test either stubs the
dependency or loads the module under test from source &mdash; so it finishes in
about a second and passes on a machine with no GPU:

```bash
python -m unittest discover -s tests
```

It covers the CUDA graph logic (when a recording is allowed, how it is verified,
how it is torn down), the Civitai update-check rendering and the sidecar reader,
the theme builder, and the failure cleanup paths.

The matching GitHub Actions workflow is checked in as
[`docs/github-tests-workflow.yml`](docs/github-tests-workflow.yml) &mdash; copy it
to `.github/workflows/tests.yml` (or commit it from a token that carries the
`workflows` permission) to have it run on every push and pull request against
Python 3.10 and 3.11. It installs nothing, so a run finishes in seconds.

<br>

## Installation

### One click (Windows)

Neither git nor Python has to be installed first.

1. Put **`install.bat`** in the folder you want the fork to live in &mdash; everything the
   installer downloads stays inside that folder, and the system drive is never written to. If the
   rest of the repository is not next to it, `install.bat` fetches the repository itself first.
2. Double-click it and press Enter until it reports that it is done.
3. From then on, start with **`run.bat`**.

`install.bat` fetches a portable Python 3.11, builds a venv in `installer_files/env`, installs
torch, delegates everything else to `launch.py --exit` (so the package list can never drift from
`requirements.txt`), and writes a `webui.settings.bat` that points the launcher at that environment
and redirects every cache &mdash; pip, HuggingFace, torch, temp &mdash; into the folder. Running it
again is safe: it only fills in what is missing. If torch is already installed on the machine, the
installer offers to reuse it instead of pulling another three gigabytes.

`console.bat` opens a console with the environment activated. The full guide, including the offline
path and troubleshooting, is in **[INSTALL.md](INSTALL.md)**.

### Manual installation

0. Install **[git](https://git-scm.com/downloads)**
1. Clone the Repo
    ```bash
    git clone https://github.com/Riyozaki/sd-webui-forge-NEO.git
    ```

2. Setup Python

<details>
<summary>Recommended Method</summary>

- Install **[uv](https://github.com/astral-sh/uv#installation)**
- Set up **venv**
    ```bash
    cd sd-webui-forge-neo
    uv venv venv --python 3.11 --seed
    ```
- Add the `--uv` flag to `webui-user.bat`

</details>

<details>
<summary>Standard Method</summary>

- Install **[Python 3.11.9](https://www.python.org/downloads/release/python-3119/)**
    - Remember to enable `Add Python to PATH`

</details>

3. **(Optional)** Configure [Commandline](#commandline)
4. Launch the WebUI via `webui-user.bat`
5. During the first launch, it will automatically install all the requirements
6. Once the installation is finished, the WebUI will start in a browser automatically

<br>

### Install sageattention 2

<details>
<summary>Expand</summary>

0. Ensure the WebUI can properly launch already, by following the [installation](#installation) steps first
1. Open the console in the WebUI directory
    ```bash
    cd sd-webui-forge-neo
    ```
2. Start the virtual environment
    ```bash
    venv\scripts\activate
    ```
3. Create a new folder
    ```bash
    mkdir repo
    cd repo
    ```
4. Clone the repo
    ```bash
    git clone https://github.com/thu-ml/SageAttention
    cd SageAttention
    ```
5. Install the library
    ```
    pip install -e . --no-build-isolation
    ```

    - If you installed `uv`, use `uv pip install` instead
    - The installation takes a few minutes

<br>

### Alternatively
> for **Windows**

- Download the pre-built `.whl` package from https://github.com/woct0rdho/SageAttention/releases
```bash
pip install sageattention...win_amd64.whl
```
- If you installed `uv`, use `uv pip install` instead
- **Important:** Download the correct `.whl` for your PyTorch version

</details>

### Install older PyTorch

<details>
<summary>Expand</summary>

0. Navigate to the WebUI directory
1. Edit the `webui-user.bat` file
2. Add a new line to specify an older version:
```bash
set TORCH_COMMAND=pip install torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cu121
```

</details>

### Install FFmpeg

<details>
<summary>Expand</summary>

> for Windows

1. Download the FFmpeg [.7z](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z)
2. Extract the contents to a folder of choice
3. Add the `bin` folder within to the system **PATH**
    - `Edit the System Environment Variables` > `Environment Variables` > `Path`
4. Verify the installation by entering `ffmpeg` in a command prompt

</details>

<br>

## Attention

> [!Important]
> The `--xformers` and `--sage` args are only responsible for installing the packages, **not** whether its respective attention is used *(this also means you can remove them once the packages are successfully installed)*

**Forge Neo** tries to import the packages and automatically choose the first available attention function in the following order:

1. `SageAttention`
2. `FlashAttention`
3. `xformers`
4. `PyTorch`
5. `Basic`

> [!Tip]
> To skip a specific attention, add the respective disable arg such as `--disable-sage`

> [!Note]
> The **VAE** only checks for `xformers`, so `--xformers` is still recommended even if you already have `--sage`

In my experience, the speed of each attention function for SDXL is ranked in the following order:

- `SageAttention` ≥ `FlashAttention` > `xformers` > `PyTorch` >> `Basic`

> [!Note]
> `SageAttention` is based on quantization, so its quality might be slightly worse than others

> [!Important]
> When using `SageAttention 2`, both positive prompts and negative prompts are required; omitting negative prompts can cause `NaN` issues

<br>

## Issues & Requests

> [!Tip]
> When reporting something slow, paste the `[NEO] ...s total - sampling ...` line
> from the console along with your settings. It says which part of the pipeline
> the time went into, which is usually most of the answer.

- **Issues** about removed features will simply be ignored
- **Issues** regarding installation will be ignored if it's obviously user-error
- Non-Windows platforms will not be supported, as I cannot verify nor maintain them

</details>

<hr>

<p align="center">
Special thanks to <b>AUTOMATIC1111</b>, <b>lllyasviel</b>, and <b>comfyanonymous</b>, <b>kijai</b>, <b>city96</b>, <br>
along with the rest of the contributors, <br>
for their invaluable efforts in the open-source image generation community
</p>
