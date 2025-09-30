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

def bond_method_to_cls(func, obj):
    if hasattr(func, '__self__') or not callable(func):
        # If the function is already bound to an instance, return it as is
        return func
    else:
        return types.MethodType(func, obj)

recorded_loader_keys = [
    'weight_loader',
    'load_qkv_weight',
    'load_row_parallel_weight',
    'load_merged_column_weight',
    'output_dim',
    'input_dim',
    '_assert_and_load',
]

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

def get_config_data_and_flash_rl_profile():
    config = os.environ.get("FLASHRL_CONFIG", None)
    rank = int(os.environ.get("RANK", None))
    mp_size = int(os.environ.get("MP_SIZE", None))
    dp_rank = rank // mp_size

    if config is not None:
        config_data = load_flashrl_config(config)
        config_count = len(config_data['configs'])
        config_index = dp_rank % config_count
        logger.info(f"Using config {config_index} of {config_count}")
        config_data = config_data['configs'][config_index]
        if config_data.get('fn', 'int8') != 'bf16':
            if config_data.get('fn', 'int8') in ['fp8_vllm', 'fp8', 'fp8_fast', 'fp8_vllm_fast']:
                flash_rl_profile = None
            else:
                model = config_data.get('model', None)
                quant_profile = config_data.get('profile', os.path.join(model, 'profile.pt'))
                logger.debug(f"Loading flash_rl profile from: {quant_profile}")
                flash_rl_profile = torch.load(quant_profile)
        return config_data, flash_rl_profile
    return None, None

@staticmethod
def hacked_load_weights_and_postprocess(
    model,
    weights,
    target_device,
    hacked_data_dict = None,
):
    # Hack model.load_weights first.
    config_data, flash_rl_profile = get_config_data_and_flash_rl_profile()
    if (not hasattr(model, 'beforeflashrl_load_weights')) and (config_data.get('fn', 'int8') != 'bf16'):
        quant_fn = config_data.get('fn', 'int8')
        model.flashrl_quant_fn = quant_fn
        logger.debug(f"flash_rl quantization function: {quant_fn}")
        flash_quantize_fn = get_quantize_fn(quant_fn)

        # Store the original load_weights function
        original_load_weights = model.load_weights
        model.beforeflashrl_load_weights = original_load_weights
        def hacked_load_weights(
            weights,
        ):
            # Skip the case: When reload weights, hacked_load_weights calls load_weights_and_postprocess, it loads weights repeatedly.
            if weights is None:
                return
            
            print("Run hacked_load_weights")
            start_time = time.time()
            setattr(model, 'hacked_not_need_process_weights_after_loading', False)

            if not hasattr(model, "hacked_original_weights_rebuild_keys"):
                print("First time load weights, call original_load_weights")
                return original_load_weights(weights)

            if 'module_attribute_to_preserve' in config_data:
                logger.debug(f"flash_rl module_attribute_to_preserve: {config_data['module_attribute_to_preserve']}")
                flash_rl_module_attribute_to_preserve = config_data.get('module_attribute_to_preserve')
            else:
                flash_rl_module_attribute_to_preserve = []

            if len(flash_rl_module_attribute_to_preserve) > 0:
                for _, module in model.named_modules():
                    for attr in flash_rl_module_attribute_to_preserve:
                        if torch.is_tensor(getattr(module, attr, None)):
                            setattr(module, f'hacked_{attr}', getattr(module, attr))

            existing_params = dict(model.named_parameters())

            hacked_data_dict = {}
            for name, p in existing_params.items():
                hacked_data_dict[name] = p.data

            for name, (shape, stride, dtype, nbytes) in model.hacked_original_weights_rebuild_keys.items():
                if name in existing_params:
                    existing_params[name].data = torch.empty(shape, dtype=dtype)

            for k, loader_k in model.hacked_recorded_loader.items():
                for n, loader in loader_k.items():
                    if not hasattr(existing_params[n], k):
                        setattr(existing_params[n], k, bond_method_to_cls(loader, existing_params[n]))

            del existing_params

            end_time = time.time()
            logger.debug(f"flash_rl load_weights preparation took {end_time - start_time:.2f} seconds")
            start_time = end_time

            logger.debug("Second time load weights")
            original_load_weights(
                flash_quantize_fn(weights, flash_rl_profile)
            )

            end_time = time.time()
            logger.debug(f"flash_rl original_load_weights took {end_time - start_time:.2f} seconds")
            start_time = end_time

            del weights
            # if hasattr(model, 'hacked_target_device'):
            from sglang.srt.model_loader.loader import DefaultModelLoader
            DefaultModelLoader.load_weights_and_postprocess(model, None, None, hacked_data_dict=hacked_data_dict)
            setattr(model, 'hacked_not_need_process_weights_after_loading', True)
            # else:
            #     setattr(model, 'hacked_not_need_process_weights_after_loading', False)
            #     for name, p in model.named_parameters():
            #         strided_data = torch.as_strided(p.data, hacked_data_dict[name].shape, hacked_data_dict[name].stride())
            #         hacked_data_dict[name].copy_(strided_data)

            #         tmp_data = p.data
            #         p.data = hacked_data_dict[name]
            #         del tmp_data

            del hacked_data_dict
            gc.collect()
            torch.cuda.empty_cache()

            if len(flash_rl_module_attribute_to_preserve) > 0:
                for _, module in model.named_modules():
                    for attr in flash_rl_module_attribute_to_preserve:
                        if torch.is_tensor(getattr(module, attr, None)):
                            assert hasattr(module, f'hacked_{attr}'), f"module {module} does not have attribute hacked_{attr}"
                            setattr(module, attr, getattr(module, f'hacked_{attr}'))
                            delattr(module, f'hacked_{attr}')

            end_time = time.time()
            logger.debug(f"flash_rl load_weights process_weights_after_loading took {end_time - start_time:.2f} seconds")
            return

        model.load_weights = hacked_load_weights
        logger.debug("Successfully patched the load_weights function of sglang")
    else:
        logger.debug("sglang load_weights patching skipped")

    # Load weights.
    print("Run hacked_load_weights_and_postprocess")
    print("model.load_weights", model.load_weights)
    model.load_weights(weights)

    # Postprocess.
    if target_device is None:
        target_device = getattr(model, 'hacked_target_device', None)
    else:
        setattr(model, 'hacked_target_device', target_device)

    # if getattr(model, 'hacked_not_need_process_weights_after_loading', False):
    #     logger.debug("vllm process_weights_after_loading already processed")
    #     return

    original_weights = dict(model.named_parameters())

    # this can be optimized for better memory usage, leave for future work...
    if not hasattr(model, 'hacked_original_weights_rebuild_keys'):
        model.hacked_original_weights_rebuild_keys = {}
        for name, p in original_weights.items():
            model.hacked_original_weights_rebuild_keys[name] = (p.shape, p.stride(), p.dtype, p.untyped_storage().nbytes())

    # record weight_loader
    if not hasattr(model, 'hacked_recorded_loader'):
        recorded_loader = {k: dict() for k in recorded_loader_keys}
        for name, p in original_weights.items():
            for k in recorded_loader.keys():
                if hasattr(p, k):
                    attr = getattr(p, k)
                    if not callable(attr):
                        recorded_loader[k][name] = attr
                    elif p is attr.__self__:
                        recorded_loader[k][name] = attr.__func__
                    else:
                        recorded_loader[k][name] = attr
        model.hacked_recorded_loader = recorded_loader

    # Original process_weights_after_loading.
    if not getattr(model, 'hacked_not_need_process_weights_after_loading', False):
        from sglang.srt.model_loader.loader import device_loading_context
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

    # Restore stride and move data back.
    if hacked_data_dict is not None:
        for name, p in model.named_parameters():
            strided_data = torch.as_strided(p.data, hacked_data_dict[name].shape, hacked_data_dict[name].stride())
            hacked_data_dict[name].copy_(strided_data)

            tmp_data = p.data
            p.data = hacked_data_dict[name]
            del tmp_data

def patch_sglang_load_weights_and_postprocess():
    try:
        from sglang.srt.model_loader.loader import DefaultModelLoader
        if not hasattr(DefaultModelLoader, 'beforeflashrl_load_weights_and_postprocess'):

            original_load_weights_and_postprocess = DefaultModelLoader.load_weights_and_postprocess
            DefaultModelLoader.beforeflashrl_load_weights_and_postprocess = original_load_weights_and_postprocess
            DefaultModelLoader.load_weights_and_postprocess = hacked_load_weights_and_postprocess

            logger.debug("Successfully patched the load_weights_and_postprocess function of sglang")
        else:
            logger.debug("sglang load_weights_and_postprocess already patched")
    except ImportError:
        logger.error(f"Error patching sglang load_weights_and_postprocess: {e}")
        return False
    return True

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
                os.environ['MP_SIZE'] = str(mp_size)
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
                            flash_rl_profile = None
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

                            flash_rl_profile = torch.load(quant_profile_path)
                            kwargs['quantization'] = "w8a8_int8"

                else:
                    logger.info(f"flash_rl config not detected.")
                    logger.info(f"Using the original model: {kwargs.get('model')}")

                # Call the parent's __init__ with the custom model
                print(f"kwargs given to sglang Engine: {kwargs}")
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
