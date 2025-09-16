import os
import gc
import time
import torch
import types
import logging
from packaging.version import parse

from torch import nn
from .flash_quantization import get_quantize_fn

# Set up logger
logger = logging.getLogger(__name__)

keys_to_overload = [
    'load_format',
    'quantization',
    # TODO: Check if sglang needs to determine distributed_executor_backend
    # 'distributed_executor_backend',
]

def load_flashrl_config(config):

    config_path = config.strip()

    if config_path in ['bf16', 'fp8', 'fp8_vllm', 'fp8_fast', 'fp8_vllm_fast']:
        logger.info(f"Using profile-free default for: {config_path}")

        from .configs import get_default_config
        from dataclasses import asdict
        config_data = {'configs': [asdict(get_default_config(config_path))]}
    else:
        logger.info(f"Loading flash_rl config from: {config_path}")

        if not os.path.exists(config_path):
            from huggingface_hub import hf_hub_download
            config_path = config_path.split('/')
            assert len(config_path) >= 3, f'Invalid flash_rl config path: {config_path}'
            config_path = hf_hub_download(repo_id='/'.join(config_path[:2]), filename='/'.join(config_path[2:]))

        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

    return config_data

def patch_sglang_Engine():
    try:
        from sglang.srt.entrypoints.engine import Engine
        if not hasattr(Engine, 'beforeflashrl__init__'):
            # Store the original LLM init function
            original_init = Engine.__init__
            Engine.beforeflashrl__init__ = original_init

            def hacked_init_(
                self,
                **kwargs
            ) -> None:
                # Patch the sampler class (TODO: patch sglang sampler)
                # sampler_patch_status = patch_vllm_logprob_compute()
                # logger.debug(f"Patching vllm Sampler... status: {sampler_patch_status}")

                config = os.environ.get("FLASHRL_CONFIG", None)

                # TODO: Check if sglang needs to determine distributed_executor_backend
                # if 'distributed_executor_backend' not in kwargs or kwargs['distributed_executor_backend'] != 'external_launcher':
                #     logger.error("flash_rl only supports external_launcher for now")
                assert 'RANK' in os.environ and 'WORLD_SIZE' in os.environ, \
                    'flash_rl only supports external_launcher for now'

                rank = int(os.environ.get("RANK", None))
                mp_size = kwargs.get('tensor_parallel_size', 1) * kwargs.get('pipeline_parallel_size', 1)
                dp_rank = rank // mp_size

                if config is not None:
                    # Load the config file and set the model
                    # Assuming config is a JSON file, you can use json.load() to read it
                    logger.info(f"flash_rl config detected.")
                    config_data = load_flashrl_config(config)

                    config_count = len(config_data['configs'])
                    config_index = dp_rank % config_count
                    logger.info(f"Using config {config_index} of {config_count}")
                    config_data = config_data['configs'][config_index]

                    for k, v in config_data.items():
                        logger.info(f"rank {rank} flash_rl config: {k}: {v}")

                    for key in keys_to_overload:
                        if key in config_data:
                            logger.debug(f"Overloading {key} with {config_data[key]}")
                            kwargs[key] = config_data.get(key)
                    model = config_data.get('model', kwargs.get('model_path'))
                    kwargs['model_path'] = model
                    if config_data.get('fn', 'int8') != 'bf16':

                        # TODO: Check sglang version here.

                        if config_data.get('fn', 'int8') in ['fp8_vllm', 'fp8', 'fp8_fast', 'fp8_vllm_fast']:
                            if 'profile' in config_data:
                                logger.warning(f"flash_rl fp8 profile is not needed, but set as {config_data['profile']}")
                            self.flash_rl_profile = None
                            kwargs['quantization'] = "w8a8_fp8"
                        else:
                            quant_profile = config_data.get('profile', os.path.join(model, 'profile.pt'))
                            logger.debug(f"Loading flash_rl profile from: {quant_profile}")

                            quant_profile_path = quant_profile.strip()
                            if not os.path.exists(quant_profile_path):
                                from huggingface_hub import hf_hub_download
                                quant_profile_path = quant_profile_path.split('/')
                                assert len(quant_profile_path) >= 3, f'Invalid flash_rl profile path: {quant_profile_path}'
                                quant_profile_path = hf_hub_download(repo_id='/'.join(quant_profile_path[:2]), filename='/'.join(quant_profile_path[2:]))

                            self.flash_rl_profile = torch.load(quant_profile_path)
                            kwargs['quantization'] = "w8a8_int8"

                    if 'module_attribute_to_preserve' in config_data:
                        logger.debug(f"flash_rl module_attribute_to_preserve: {config_data['module_attribute_to_preserve']}")
                        self.flash_rl_module_attribute_to_preserve = config_data.get('module_attribute_to_preserve')
                    else:
                        self.flash_rl_module_attribute_to_preserve = []

                else:
                    logger.info(f"flash_rl config not detected.")
                    logger.info(f"Using the original model: {kwargs.get('model')}")

                # Call the parent's __init__ with the custom model
                init_return = original_init(
                    self,
                    **kwargs,
                )

                # TODO: Quantize weights before loading

                return init_return

            # Patch the LLM init function
            Engine.__init__ = hacked_init_

            logger.debug("Successfully patched sglang Engine")
        else:
            logger.debug("sglang Engine already patched")
        return True

    except Exception as e:
        logger.error(f"Error patching sglang Engine: {e}")
        return False
