import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import argparse
import json


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
    except TypeError: # Handle cases where obj might not be subscriptable as expected
        # Attempt to access 'result' if it's a dictionary-like object
        if hasattr(obj, '__getitem__') and "result" in obj:
             try:
                 return obj["result"][index]
             except KeyError:
                 raise IndexError(f"Index {index} not found in obj or obj['result']")
        raise IndexError(f"Object is not a sequence or mapping, or index {index} is out of bounds.")


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    if path is None:
        path = os.getcwd()

    abs_path = os.path.abspath(path)

    if name in os.listdir(abs_path):
        path_name = os.path.join(abs_path, name)
        print(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(abs_path)

    if parent_directory == abs_path:
        return None

    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        # Check if the path is already in sys.path
        if comfyui_path not in sys.path:
            sys.path.append(comfyui_path)
            print(f"'{comfyui_path}' added to sys.path")
        else:
            print(f"'{comfyui_path}' already in sys.path")
    else:
        print("Could not find ComfyUI directory.")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config # Moved import here
    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


# --- ComfyUI Setup ---
# Ensure setup runs only once
if "NODE_CLASS_MAPPINGS" not in globals():
    print("Initializing ComfyUI...")
    add_comfyui_directory_to_sys_path()
    # It's crucial to import main AFTER ComfyUI is in sys.path
    try:
        add_extra_model_paths()
    except ImportError as e:
        print(f"Could not import from main.py or utils.extra_config: {e}")
        print("Proceeding without loading extra model paths.")
    except Exception as e:
        print(f"An error occurred during extra model paths setup: {e}")
        print("Proceeding without loading extra model paths.")

    def import_custom_nodes() -> None:
        """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS"""
        import asyncio
        import execution
        from nodes import init_extra_nodes
        import server

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)
        init_extra_nodes()

    try:
        import_custom_nodes()
        from nodes import NODE_CLASS_MAPPINGS
        print("ComfyUI initialized successfully.")
    except ImportError as e:
        print(f"Failed to import ComfyUI components: {e}")
        print("Please ensure ComfyUI is correctly installed and accessible.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during ComfyUI initialization: {e}")
        sys.exit(1)
else:
    print("ComfyUI already initialized.")
    from nodes import NODE_CLASS_MAPPINGS # Ensure it's available if script is re-run in same env


# --- Video Generation Function ---
def generate_video_from_config(config: dict):
    """Generates a video based on the provided configuration."""
    print(f"\n--- Generating video for prompt: {config['prompt'][:50]}... ---")
    print(f"Config: {config}") # Print the specific config for this run

    with torch.inference_mode():
        # Loaders - Consider caching these if running multiple generations
        # to avoid reloading models repeatedly. For now, load per call.
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()

        # Load Models based on config
        try:
            cliploader_result = cliploader.load_clip(
                clip_name=config["clip_name"], type="wan", device="default"
            )
            unetloader_result = unetloader.load_unet(
                unet_name=config["unet_name"], weight_dtype="default"
            )
            vaeloader_result = vaeloader.load_vae(vae_name=config["vae_name"])

            # Apply LoRA
            lora_model_result = loraloadermodelonly.load_lora_model_only(
                lora_name=config["lora_name"],
                strength_model=1, # Assuming strength is fixed, could be a param
                model=get_value_at_index(unetloader_result, 0),
            )
            loaded_model = get_value_at_index(lora_model_result, 0)

        except Exception as e:
            print(f"Error loading models: {e}")
            print(f"Check if model names are correct and files exist:")
            print(f"  CLIP: {config['clip_name']}")
            print(f"  UNET: {config['unet_name']}")
            print(f"  VAE: {config['vae_name']}")
            print(f"  LoRA: {config['lora_name']}")
            return # Skip generation if models fail to load

        # Encoders
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        positive_clip = get_value_at_index(cliploader_result, 0)
        positive_cond = cliptextencode.encode(
            text=config["prompt"], clip=positive_clip
        )
        negative_cond = cliptextencode.encode(
            text=config["negative_prompt"], clip=positive_clip
        )

        # Latent Image
        emptyhunyuanlatentvideo = NODE_CLASS_MAPPINGS["EmptyHunyuanLatentVideo"]()
        latent_image = emptyhunyuanlatentvideo.generate(
            width=config["width"],
            height=config["height"],
            length=config["length"],
            batch_size=config["batch_size"],
        )

        # Sampler
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        sampled_latent = ksampler.sample(
            seed=config['seed'] if config.get('seed') is not None else random.randint(1, 2**64),
            steps=config["steps"],
            cfg=config["cfg"],
            sampler_name="uni_pc", # Hardcoded for now, could be param
            scheduler="simple",   # Hardcoded for now, could be param
            denoise=1,            # Hardcoded for now, could be param
            model=loaded_model,
            positive=get_value_at_index(positive_cond, 0),
            negative=get_value_at_index(negative_cond, 0),
            latent_image=get_value_at_index(latent_image, 0),
        )

        # Decoder
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        decoded_images = vaedecode.decode(
            samples=get_value_at_index(sampled_latent, 0),
            vae=get_value_at_index(vaeloader_result, 0),
        )

        # Save Output
        saveanimatedwebp = NODE_CLASS_MAPPINGS["SaveAnimatedWEBP"]()
        saveanimatedwebp.save_images(
            filename_prefix=config["filename_prefix"],
            fps=config["fps"],
            lossless=False, # Hardcoded, could be param
            quality=90,     # Hardcoded, could be param
            method="default",# Hardcoded, could be param
            images=get_value_at_index(decoded_images, 0),
        )
        print(f"--- Video saved with prefix: {config['filename_prefix']} ---")


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate videos using ComfyUI workflow via command line.")

    # Video Generation Parameters
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--length", type=int, default=33, help="Video length in frames")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")

    # Prompts
    parser.add_argument("--prompt", type=str, help="Positive prompt text (required if no config file)")
    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", help="Negative prompt text")

    # Model Names (pointing to files) - Paths are required either via CLI or config file
    parser.add_argument("--vae_name", type=str, help="Path to VAE model file")
    parser.add_argument("--lora_name", type=str, help="Path to LoRA model file")
    parser.add_argument("--clip_name", type=str, help="Path to CLIP model file")
    parser.add_argument("--unet_name", type=str, help="Path to UNET model file")

    # KSampler Parameters
    parser.add_argument("--steps", type=int, default=30, help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=None, help="KSampler seed (random if not specified)")
    parser.add_argument("--cfg", type=float, default=6.0, help="CFG scale")

    # Output Parameters
    parser.add_argument("--filename_prefix", type=str, default="Video", help="Prefix for the output WEBP file")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the output WEBP file")

    # Configuration File
    parser.add_argument("--config_file", type=str, help="Path to a .jsonl file with generation configurations, one JSON object per line.")

    args = parser.parse_args()

    # --- Argument Validation ---
    model_args_provided = all([args.vae_name, args.lora_name, args.clip_name, args.unet_name])
    prompt_provided = args.prompt is not None

    if not args.config_file:
        # If no config file, all model paths and prompt are required via CLI
        if not model_args_provided:
            parser.error("When not using --config_file, --vae_name, --lora_name, --clip_name, and --unet_name are required.")
        if not prompt_provided:
            parser.error("--prompt is required when not using --config_file")

    # --- Execution Flow ---
    if args.config_file:
        print(f"Processing config file: {args.config_file}")
        try:
            with open(args.config_file, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): # Skip empty lines and comments
                        continue
                    try:
                        # Load config from the line
                        line_config = json.loads(line)
                        count += 1

                        # Create the final config for this run:
                        # Create the final config for this run:
                        # Start with command-line args as base (includes defaults like steps, cfg, etc. and potentially seed)
                        current_config = vars(args).copy()
                        # Update/override with values from the JSON line
                        current_config.update(line_config)
                        # Remove config_file key as it's not needed for generation func
                        if 'config_file' in current_config:
                            del current_config['config_file'] # Should be removed anyway by vars(args) but belt-and-suspenders

                        # Validate required fields for config file line
                        required_keys = ['prompt', 'vae_name', 'lora_name', 'clip_name', 'unet_name']
                        missing_keys = [key for key in required_keys if key not in current_config or not current_config[key]]

                        if missing_keys:
                            print(f"Skipping line {count}: Missing or empty required keys: {', '.join(missing_keys)}")
                            continue

                        generate_video_from_config(current_config)

                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line {count}: {line}")
                    except Exception as e:
                        print(f"Error processing line {count}: {line} - Error: {e}")
                print(f"Finished processing {count} configurations from {args.config_file}")

        except FileNotFoundError:
            print(f"Error: Config file not found at {args.config_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading or processing config file {args.config_file}: {e}")
            sys.exit(1)

    elif prompt_provided and model_args_provided:
        # Single run using command-line arguments (already validated above)
        config = vars(args)
        # Remove config_file key as it's not needed for generation func
        if 'config_file' in config:
            del config['config_file'] # Should be None here anyway

        print("Starting single generation run with command-line arguments...")
        generate_video_from_config(config)
        print("Single generation run finished.")


if __name__ == "__main__":
    main()
