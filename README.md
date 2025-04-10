# ComfyUI Batch Video Generator (for Wan 2.1)

This repository contains a Python script (`wan_video_generator.py`) specifically adapted to run batch video generation tasks using the **Wan 2.1 model** within ComfyUI. It utilizes configurations defined in an external JSON file (`experiments.json`) to allow for programmatic experimentation and queuing of multiple video generations without manual intervention.

This script was originally generated from a Wan 2.1 workflow using the [ComfyUI-to-Python-Extension](https://github.com/pydn/ComfyUI-to-Python-Extension) by Peyton DeNiro and subsequently modified to support batch processing from a configuration file and model caching.

## Benefits

*   **Batch Processing:** Run hundreds or thousands of video generation experiments defined in a simple JSON file (`experiments.json`).
*   **Automation:** Avoids the need to manually queue each generation in the ComfyUI interface.
*   **Parameter Sweeping:** Easily test variations in prompts, seeds, CFG, steps, models, LoRAs, and other parameters systematically.
*   **Efficiency:** Includes model caching to avoid reloading the same models (CLIP, UNet, VAE) for consecutive runs, saving time and potentially improving stability.
*   **Reproducibility:** Keep a record of your experiments and their configurations in the `experiments.json` file.

## Prerequisites

*   **ComfyUI:** You must have a working installation of ComfyUI. This script is designed to be run from within the root directory of your ComfyUI installation.

## Setup

1.  **Clone this Repository:**
    ```bash
    # Navigate to your desired workspace directory
    cd /workspace/
    git clone https://github.com/learn2train/comfyui-workflows.git
    ```

2.  **Download Base Models:** Download the required Wan 2.1 models (Text Encoder, VAE, Diffusion Model) into your workspace. Note that we use the 14B text to video version here, but you can adapt the following to use the 1.3B parameters version instead.
    ```bash
    cd /workspace/
    # Text Encoder
    wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
    # VAE
    wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors
    # Diffusion Model
    wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors
    ```

3.  **Prepare LoRAs (Optional):** If your experiments use LoRAs, ensure they (e.g., `epoch10.safetensors`) and any associated configuration files (e.g., `adapter_config.json`) are available in your workspace. You might need to upload them if using a remote environment (e.g., using `runpodctl send <file>`).

4.  **Prepare Configuration:** Create or upload your `experiments.json` file to your workspace. This file defines the parameters for each video generation run (see Configuration section below).

5.  **Organize ComfyUI Models:** Move the downloaded and prepared models/LoRAs into the correct subdirectories within your ComfyUI installation folder.
    ```bash
    # Define your ComfyUI path
    COMFYUI_PATH=/workspace/ComfyUI

    # Create directories if they don't exist
    mkdir -p $COMFYUI_PATH/models/text_encoders/
    mkdir -p $COMFYUI_PATH/models/vae/
    mkdir -p $COMFYUI_PATH/models/diffusion_models/
    mkdir -p $COMFYUI_PATH/models/loras/

    # Move models (adjust paths based on your workspace)
    mv /workspace/umt5_xxl_fp8_e4m3fn_scaled.safetensors $COMFYUI_PATH/models/text_encoders/
    mv /workspace/wan_2.1_vae.safetensors $COMFYUI_PATH/models/vae/
    mv /workspace/wan2.1_t2v_14B_bf16.safetensors $COMFYUI_PATH/models/diffusion_models/

    # Move LoRAs (adjust paths and filenames)
    mv /workspace/epoch*.safetensors $COMFYUI_PATH/models/loras/
    mv /workspace/adapter_config.json $COMFYUI_PATH/models/loras/
    ```

6.  **Place Script and Config:** Copy the generator script and your experiments configuration file into the root of your ComfyUI installation.
    ```bash
    cp /workspace/comfyui-workflows/wan_video_generator.py $COMFYUI_PATH/
    cp /workspace/experiments.json $COMFYUI_PATH/
    ```
    *(Note: Ensure the `experiments.json` path inside `wan_video_generator.py` matches its location*

## Configuration (`experiments.json`)

This file should contain a JSON list (`[...]`) where each element is an object (`{...}`) defining one experiment. Key parameters include:

*   `prompt` (string): The positive text prompt.
*   `negative_prompt` (string, optional): The negative text prompt. Defaults to a standard one if omitted.
*   `seed` (integer, optional): The random seed. If omitted or `null`, a random seed will be generated.
*   `filename_prefix` (string): Base name for the output video file (saved in ComfyUI's default output directory).
*   `cfg` (float): CFG scale (e.g., `7.0`).
*   `steps` (integer): Number of sampling steps (e.g., `30`).
*   `latent_width` (integer): Width of the latent video (e.g., `1280`).
*   `latent_height` (integer): Height of the latent video (e.g., `720`).
*   `latent_length` (integer): Number of frames (e.g., `33`).
*   `clip_name` (string): Filename of the CLIP model (must exist in `ComfyUI/models/text_encoders/`).
*   `unet_name` (string): Filename of the UNet/Diffusion model (must exist in `ComfyUI/models/diffusion_models/`).
*   `vae_name` (string): Filename of the VAE model (must exist in `ComfyUI/models/vae/`).
*   `lora_name` (string, optional): Filename of the LoRA model (must exist in `ComfyUI/models/loras/`). Use `null` or omit if no LoRA.

**Example `experiments.json` entry:**

```json
{
  "prompt": "A cat riding a bicycle on the moon, cinematic",
  "negative_prompt": "blurry, low quality, text, watermark",
  "seed": 123456789,
  "filename_prefix": "cat_moon_bike_s123_epoch10",
  "cfg": 6.0,
  "steps": 30,
  "latent_width": 1280,
  "latent_height": 720,
  "latent_length": 33,
  "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
  "unet_name": "wan2.1_t2v_14B_bf16.safetensors",
  "vae_name": "wan_2.1_vae.safetensors",
  "lora_name": "epoch10.safetensors",
}
```

## Usage

1.  Navigate to your ComfyUI root directory.
    ```bash
    cd /workspace/ComfyUI
    ```
2.  Run the script using Python.
    ```bash
    python wan_video_generator.py
    ```
    *(Consider using `tmux` or `screen` for long runs, especially on remote machines: `tmux new -s inference`)*

The script will load experiments from `experiments.json`, execute them sequentially, cache models where possible, and save outputs to the default ComfyUI output directory.

## License

This project uses the MIT License. See the `LICENSE` file for details.

## Credits

*   The initial Python script structure was generated using the [ComfyUI-to-Python-Extension](https://github.com/pydn/ComfyUI-to-Python-Extension) by Peyton DeNiro.
*   ComfyUI by Comfyanonymous.
