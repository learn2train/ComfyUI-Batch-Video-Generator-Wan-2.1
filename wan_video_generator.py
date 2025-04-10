import os
import random
import json # Added import
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()

    # --- Configuration ---
    config_path = "experiments.json"
    default_negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    # --- Load Experiments ---
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            experiments = json.load(f)
        if not isinstance(experiments, list):
            print(f"Error: Configuration file {config_path} must contain a JSON list (array).")
            sys.exit(1)
        print(f"Loaded {len(experiments)} experiments from {config_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading configuration file {config_path}: {e}")
        sys.exit(1)

    if not experiments:
        print("No experiments defined in the configuration file. Exiting.")
        sys.exit(0)

    # --- Main Execution Logic ---
    with torch.inference_mode():
        # Instantiate node classes that don't depend on config per run
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        emptyhunyuanlatentvideo = NODE_CLASS_MAPPINGS["EmptyHunyuanLatentVideo"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveanimatedwebp = NODE_CLASS_MAPPINGS["SaveAnimatedWEBP"]()
        # Loaders need to be instantiated inside the loop as models change
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()

        # --- Initialize Model Cache ---
        loaded_clip_name = None
        loaded_clip_model = None
        loaded_unet_name = None
        loaded_unet_model = None # Base UNet cache
        loaded_vae_name = None
        loaded_vae_model = None

        for i, config in enumerate(experiments):
            print(f"\n--- Starting Run {i+1}/{len(experiments)} ---")
            print(f"Config: {config}")

            # Get parameters for this run, using defaults if necessary
            try:
                prompt = config["prompt"]
                # Get seed, default to None if missing or null in JSON
                seed_value = config.get("seed")
                if seed_value is None:
                    seed = random.randint(1, 2**64)
                    print(f"Seed not specified or null, using random seed: {seed}")
                else:
                    seed = int(seed_value) # Ensure seed is an integer

                filename_prefix = config.get('filename_prefix', 'Video')
                cfg = config.get("cfg", 6.0)
                steps = config.get("steps", 30)
                latent_width = config.get("latent_width", 832)
                latent_height = config.get("latent_height", 480)
                latent_length = config.get("latent_length", 33)
                clip_name = config["clip_name"]
                unet_name = config["unet_name"]
                vae_name = config["vae_name"]
                # Optional parameters with defaults
                negative_prompt = config.get("negative_prompt", default_negative_prompt)
                lora_name = config.get("lora_name") # Can be None

            except KeyError as e:
                print(f"!!! ERROR: Missing required key in config for run {i+1}: {e}")
                print("!!! Skipping this run.")
                continue

            # --- Model Loading (with Caching) ---
            try:
                # Load CLIP (Cache Check)
                if loaded_clip_name != clip_name or loaded_clip_model is None:
                    print(f"Loading CLIP: {clip_name}...")
                    cliploader_result = cliploader.load_clip(clip_name=clip_name, type="wan", device="default")
                    loaded_clip_model = get_value_at_index(cliploader_result, 0)
                    loaded_clip_name = clip_name
                    # Optional: Clear previous model from memory if needed
                    # del cliploader_result
                else:
                    print(f"Reusing cached CLIP: {loaded_clip_name}")
                clip_model = loaded_clip_model # Use the cached or newly loaded model

                # Load UNet (Cache Check) - Base UNet
                if loaded_unet_name != unet_name or loaded_unet_model is None:
                    print(f"Loading UNet: {unet_name}...")
                    unetloader_result = unetloader.load_unet(unet_name=unet_name, weight_dtype="default")
                    loaded_unet_model = get_value_at_index(unetloader_result, 0)
                    loaded_unet_name = unet_name
                    # del unetloader_result
                else:
                    print(f"Reusing cached UNet: {loaded_unet_name}")
                # NOTE: We use loaded_unet_model as the base for LoRA application below

                # Load VAE (Cache Check)
                if loaded_vae_name != vae_name or loaded_vae_model is None:
                    print(f"Loading VAE: {vae_name}...")
                    vaeloader_result = vaeloader.load_vae(vae_name=vae_name)
                    loaded_vae_model = get_value_at_index(vaeloader_result, 0)
                    loaded_vae_name = vae_name
                    # del vaeloader_result
                else:
                    print(f"Reusing cached VAE: {loaded_vae_name}")
                vae_model = loaded_vae_model # Use the cached or newly loaded model

                # Apply LoRA (if specified) - Always applied to the current base UNet (cached or new)
                if lora_name:
                    # TODO: Add strength_model to config? Currently hardcoded to 1
                    lora_strength_model = config.get("lora_strength", 1.0)
                    print(f"Applying LoRA: {lora_name} (Strength: {lora_strength_model}) to UNet: {loaded_unet_name}")
                    loraloadermodelonly_result = loraloadermodelonly.load_lora_model_only(
                        lora_name=lora_name,
                        strength_model=lora_strength_model,
                        model=loaded_unet_model # Apply LoRA to the potentially cached base UNet
                    )
                    model_for_sampling = get_value_at_index(loraloadermodelonly_result, 0)
                    # del loraloadermodelonly_result # Clean up intermediate result
                else:
                    print("No LoRA specified for this run.")
                    model_for_sampling = loaded_unet_model # Use the base UNet directly

            except Exception as e:
                print(f"!!! ERROR loading/applying models for run {i+1}: {e}")
                print("!!! Skipping this run.")
                continue # Skip to the next experiment

            # --- Workflow Execution ---
            try:
                print("Encoding prompts...")
                positive_cond = cliptextencode.encode(text=prompt, clip=clip_model)
                negative_cond = cliptextencode.encode(text=negative_prompt, clip=clip_model)

                print(f"Generating empty latent ({latent_width}x{latent_height}x{latent_length})...")
                latent_image = emptyhunyuanlatentvideo.generate(
                    width=latent_width,
                    height=latent_height,
                    length=latent_length, # Use configured length
                    batch_size=1
                )

                print(f"Sampling ({steps} steps, CFG {cfg})...")
                ksampler_result = ksampler.sample(
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name="uni_pc", # Fixed sampler
                    scheduler="simple",   # Fixed scheduler
                    denoise=1,
                    model=model_for_sampling,
                    positive=get_value_at_index(positive_cond, 0),
                    negative=get_value_at_index(negative_cond, 0),
                    latent_image=get_value_at_index(latent_image, 0),
                )

                print("Decoding...")
                vaedecode_result = vaedecode.decode(
                    samples=get_value_at_index(ksampler_result, 0),
                    vae=vae_model,
                )

                print(f"Saving output: {filename_prefix}...")
                saveanimatedwebp.save_images(
                    filename_prefix=filename_prefix,
                    fps=16, # Fixed FPS
                    lossless=False, quality=90, method="default", # Fixed save params
                    images=get_value_at_index(vaedecode_result, 0),
                )
                print(f"--- Finished Run {i+1}/{len(experiments)} ---")

            except Exception as e:
                print(f"!!! ERROR during generation for run {i+1}: {e}")
                print("!!! Continuing to next run if possible.")

            # --- Cleanup (Optional but recommended for long runs) ---
            # Consider uncommenting if memory becomes an issue
            # try:
            #     del clip_model, unet_model, vae_model, model_for_sampling
            #     del cliploader_result, unetloader_result, vaeloader_result
            #     if lora_name: del loraloadermodelonly_result
            #     del positive_cond, negative_cond, latent_image, ksampler_result, vaedecode_result
            #     if torch.cuda.is_available():
            #         torch.cuda.empty_cache()
            # except NameError: # Handle cases where variables might not be defined due to errors
            #     pass

        print("\n=== All experiments finished ===")


if __name__ == "__main__":
    main()
