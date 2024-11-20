# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import (
    FluxPipeline,
    AutoencoderKL,
    FluxTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)


import os
import json
import torch
import numpy as np
import PIL
from PIL import Image
from IPython.display import HTML
import sys
sys.path.append('/scratch/aseth7/dag/Pyramid-Flow')
from pyramid_dit import PyramidDiTForVideoGeneration
from IPython.display import Image as ipython_image
from diffusers.utils import load_image, export_to_video, export_to_gif

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class RewardPyramidDiTForVideoGeneration(PyramidDiTForVideoGeneration):
    r"""
    The Flux pipeline for text-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        model_path: str,
        model_dtype: torch.dtype,
        model_name: str = "pyramid_flux",
        model_variant: str = "diffusion_transformer_768p",
        memsave: bool = False,
    ):
        super().__init__(model_path,
                        model_dtype,
                        model_name=model_name,
                        model_variant=model_variant)
        # optionally enable memsave_torch
        # if memsave:
        #     import memsave_torch.nn

        #     vae = memsave_torch.nn.convert_to_memory_saving(vae)
        #     transformer = memsave_torch.nn.convert_to_memory_saving(transformer)
        #     text_encoder = memsave_torch.nn.convert_to_memory_saving(text_encoder)
        #     text_encoder_2 = memsave_torch.nn.convert_to_memory_saving(text_encoder_2)
        # enable checkpointing
        # self.text_encoder.gradient_checkpointing_enable()
        # text_encoder_2.gradient_checkpointing_enable()
        # self.dit.enable_gradient_checkpointing()
        self.vae.eval()
        self.text_encoder.eval()
        # text_encoder_2.eval()
        self.dit.eval()

        # freeze diffusion parameters
        freeze_params(self.vae.parameters())
        freeze_params(self.dit.parameters())
        freeze_params(self.text_encoder.parameters())
        # freeze_params(text_encoder_2.parameters())


    def apply(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 7.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]]="cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False, # If true, reload device will be cuda.
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
    ):
        if self.sequential_offload_enabled and not cpu_offloading:
            print("Warning: overriding cpu_offloading set to false, as it's needed for sequential cpu offload")
            cpu_offloading=True
        device = self.device if not cpu_offloading else torch.device("cuda")
        dtype = self.dtype
        if cpu_offloading:
            # skip caring about the text encoder here as its about to be used anyways.
            if not self.sequential_offload_enabled:
                if str(self.dit.device) != "cpu":
                    print("(dit) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                    self.dit.to("cpu")
                    torch.cuda.empty_cache()
            if str(self.vae.device) != "cpu":
                print("(vae) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                self.vae.to("cpu")
                torch.cuda.empty_cache()


        assert (temp - 1) % self.frame_per_unit == 0, "The frames should be divided by frame_per unit"

        if isinstance(prompt, str):
            batch_size = 1
            prompt = prompt + ", hyper quality, Ultra HD, 8K"        # adding this prompt to improve aesthetics
        else:
            assert isinstance(prompt, list)
            batch_size = len(prompt)
            prompt = [_ + ", hyper quality, Ultra HD, 8K" for _ in prompt]

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * len(self.stages)

        if isinstance(video_num_inference_steps, int):
            video_num_inference_steps = [video_num_inference_steps] * len(self.stages)

        negative_prompt = negative_prompt or ""

        # Get the text embeddings
        if cpu_offloading and not self.sequential_offload_enabled:
            self.text_encoder.to("cuda")
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)
        if cpu_offloading:
            if not self.sequential_offload_enabled:
                self.text_encoder.to("cpu")
                self.dit.to("cuda")
            torch.cuda.empty_cache()

        if use_linear_guidance:
            max_guidance_scale = guidance_scale
            # guidance_scale_list = torch.linspace(max_guidance_scale, min_guidance_scale, temp).tolist()
            guidance_scale_list = [max(max_guidance_scale - alpha * t_, min_guidance_scale) for t_ in range(temp)]
            print(guidance_scale_list)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        if is_sequence_parallel_initialized():
            # sync the prompt embedding across multiple GPUs
            sp_group_rank = get_sequence_parallel_group_rank()
            global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
            torch.distributed.broadcast(prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(pooled_prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(prompt_attention_mask, global_src_rank, group=get_sequence_parallel_group())

        # Create the initial random noise
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else  self.dit.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            temp,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        # by default, we needs to start from the block noise
        for _ in range(len(self.stages)-1):
            height //= 2;width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2
        
        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        num_units = 1 + (temp - 1) // self.frame_per_unit
        stages = self.stages

        generated_latents_list = []    # The generated results
        last_generated_latents = None

        for unit_index in tqdm(range(num_units)):
            gc.collect()
            torch.cuda.empty_cache()
            
            if callback:
                callback(unit_index, num_units)
            
            if use_linear_guidance:
                self._guidance_scale = guidance_scale_list[unit_index]
                self._video_guidance_scale = guidance_scale_list[unit_index]

            if unit_index == 0:
                past_condition_latents = [[] for _ in range(len(stages))]
                intermed_latents = self.generate_one_unit(
                    latents[:,:,:1],
                    past_condition_latents,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    num_inference_steps,
                    height,
                    width,
                    1,
                    device,
                    dtype,
                    generator,
                    is_first_frame=True,
                )
            else:
                # prepare the condition latents
                past_condition_latents = []
                clean_latents_list = self.get_pyramid_latent(torch.cat(generated_latents_list, dim=2), len(stages) - 1)
                
                for i_s in range(len(stages)):
                    last_cond_latent = clean_latents_list[i_s][:,:,-(self.frame_per_unit):]

                    stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
            
                    # pad the past clean latents
                    cur_unit_num = unit_index
                    cur_stage = i_s
                    cur_unit_ptx = 1

                    while cur_unit_ptx < cur_unit_num:
                        cur_stage = max(cur_stage - 1, 0)
                        if cur_stage == 0:
                            break
                        cur_unit_ptx += 1
                        cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                    if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                        cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
                
                    stage_input = list(reversed(stage_input))
                    past_condition_latents.append(stage_input)

                intermed_latents = self.generate_one_unit(
                    latents[:,:, 1 + (unit_index - 1) * self.frame_per_unit:1 + unit_index * self.frame_per_unit],
                    past_condition_latents,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    video_num_inference_steps,
                    height,
                    width,
                    self.frame_per_unit,
                    device,
                    dtype,
                    generator,
                    is_first_frame=False,
                )

            generated_latents_list.append(intermed_latents[-1])
            last_generated_latents = intermed_latents

        generated_latents = torch.cat(generated_latents_list, dim=2)

        if output_type == "latent":
            image = generated_latents
        else:
            if cpu_offloading:
                if not self.sequential_offload_enabled:
                    self.dit.to("cpu")
                self.vae.to("cuda")
                torch.cuda.empty_cache()
            image = self.decode_latent(generated_latents, save_memory=save_memory, inference_multigpu=inference_multigpu)
            if cpu_offloading:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
                # not technically necessary, but returns the pipeline to its original state

        return image
