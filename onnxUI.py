import argparse
import functools
import gc
import os
import re
import cv2
import time
import colortrans
from clip_interrogator import Config, Interrogator
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from typing import Optional, Tuple
from math import ceil
import tempfile
import signal
import shutil

from diffusers import (
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
)
from diffusers import __version__ as _df_version
import gradio as gr
import numpy as np
from packaging import version
import PIL
from PIL import Image

import lpw_pipe

# We want to safe data to PNG
from PIL import Image, PngImagePlugin


# gradio function
def run_diffusers(
    prompt: str,
    neg_prompt: Optional[str],
    init_image: Optional[PIL.Image.Image],
    init_mask: Optional[PIL.Image.Image],
    iteration_count: int,
    batch_size: int,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    eta: float,
    denoise_strength: Optional[float],
    seed: str,
    image_format: str,
    legacy: bool,
    savemask: bool,
    video: bool,
    fps: float,
    firststep: int,
    laststep: int,
    loopback: bool,
    loopback_halving: bool,
    colortransfer: bool,
    transfer_methods: str,
    transfer_amounts: str,
) -> Tuple[list, str]:
    global model_name
    global current_pipe
    global pipe

    prompt.strip("\n")
    neg_prompt.strip("\n")

    # generate seeds for iterations
    if seed == "":
        rng = np.random.default_rng()
        seed = rng.integers(np.iinfo(np.uint32).max)
    else:
        try:
            seed = int(seed) & np.iinfo(np.uint32).max
        except ValueError:
            seed = hash(seed) & np.iinfo(np.uint32).max

    # use given seed for the first iteration
    seeds = np.array([seed], dtype=np.uint32)

    if iteration_count > 1:
        seed_seq = np.random.SeedSequence(seed)
        seeds = np.concatenate(
            (seeds, seed_seq.generate_state(iteration_count - 1))
        )

    # create and parse output directory
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)
    dir_list = os.listdir(output_path)
    if video is False:
        if len(dir_list) and video is False:
            pattern = re.compile(
                r"([0-9][0-9][0-9][0-9][0-9][0-9])-([0-9][0-9])\..*"
            )
            match_list = [pattern.match(f) for f in dir_list]
            next_index = max([int(m[1]) if m else -1 for m in match_list]) + 1
        else:
            next_index = 0
    else:
        next_index = 0

    sched_name = pipe.scheduler.__class__.__name__
    if sched_name == "DPMSolverMultistepScheduler":
        sched_short_name = "DPMSM"
    elif sched_name == "DPMSolverSinglestepScheduler":
        sched_short_name = "DPMSS"
    elif sched_name == "EulerAncestralDiscreteScheduler":
        sched_short_name = "EulerA"
    elif sched_name == "EulerDiscreteScheduler":
        sched_short_name = "Euler"
    elif sched_name == "DDPMScheduler":
        sched_short_name = "DDPM"
    elif sched_name == "DDIMScheduler":
        sched_short_name = "DDIM"
    elif sched_name == "LMSDiscreteScheduler":
        sched_short_name = "LMS"
    elif sched_name == "PNDMScheduler":
        sched_short_name = "PNDM"
    elif sched_name == "KDPM2DiscreteScheduler":
        sched_short_name = "KDPM2"
    elif sched_name == "KDPM2AncestralDiscreteScheduler":
        sched_short_name = "KDPM2A"
    elif sched_name == "DEISMultistepScheduler":
        sched_short_name = "DEIS"
    elif sched_name == "HeunDiscreteScheduler":
        sched_short_name = "Heun"

    neg_prompt = None if neg_prompt == "" else neg_prompt
    images = []
    time_taken = 0

    # image
    if video is False:
        for i in range(iteration_count):
            print(f"iteration {i + 1}/{iteration_count}")

            info = (
                f"{next_index + i:06} | "
                f"prompt: {prompt} "
                f"negative prompt: {neg_prompt} | "
                f"scheduler: {sched_name} "
                f"model: {model_name} "
                f"iteration size: {iteration_count} "
                f"batch size: {batch_size} "
                f"steps: {steps} "
                f"scale: {guidance_scale} "
                f"width: {width} "
                f"height: {height} "
                f"eta: {eta} "
                f"seed: {seeds[i]}"
            )
            if current_pipe == "img2img":
                info = info + f" denoise: {denoise_strength}"
            with open(os.path.join(output_path, "history.txt"), "a") as log:
                log.write(info + "\n")

            # create generator object from seed
            rng = np.random.RandomState(seeds[i])

            if current_pipe == "txt2img":
                start = time.time()
                batch_images = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=batch_size,
                    generator=rng,
                ).images
                finish = time.time()
            elif current_pipe == "img2img":
                start = time.time()

                if loopback is True:
                    try:
                        loopback_image
                    except UnboundLocalError:
                        loopback_image = None

                    if loopback_image is not None:
                        if loopback_halving is True:
                            denoise_strength = denoise_strength * 0.5
                            if denoise_strength < 0.01:
                                denoise_strength = 0.01
                                print("limited denoise to 0.01")
                            print(f"denoise adjusted to {denoise_strength}")
                            if denoise_strength == 0.01:
                                steps = steps
                                print(f"steps not adjusted")
                            elif denoise_strength != 0.01:
                                steps = steps * 2

                            if (steps > 1000) and (
                                sched_name == "DPMSM" or "DPMSS" or "DEIS"
                            ):
                                steps = 1000
                                print("limited steps to 1000")

                        batch_images = pipe(
                            prompt,
                            negative_prompt=neg_prompt,
                            image=loopback_image,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            eta=eta,
                            strength=denoise_strength,
                            num_images_per_prompt=batch_size,
                            generator=rng,
                        ).images
                    elif loopback_image is None:
                        batch_images = pipe(
                            prompt,
                            negative_prompt=neg_prompt,
                            image=init_image,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            eta=eta,
                            strength=denoise_strength,
                            num_images_per_prompt=batch_size,
                            generator=rng,
                        ).images
                elif loopback is False:
                    batch_images = pipe(
                        prompt,
                        negative_prompt=neg_prompt,
                        image=init_image,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        strength=denoise_strength,
                        num_images_per_prompt=batch_size,
                        generator=rng,
                    ).images
                finish = time.time()
            elif current_pipe == "inpaint":
                start = time.time()
                if legacy is True:
                    batch_images = pipe(
                        prompt,
                        negative_prompt=neg_prompt,
                        image=init_image,
                        mask_image=init_mask,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        num_images_per_prompt=batch_size,
                        generator=rng,
                    ).images
                else:
                    batch_images = pipe(
                        prompt,
                        negative_prompt=neg_prompt,
                        image=init_image,
                        mask_image=init_mask,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        num_images_per_prompt=batch_size,
                        generator=rng,
                    ).images
                finish = time.time()

            short_prompt = prompt.strip('<>:"/\\|?*\n\t')
            short_prompt = re.sub(r'[\\/*?:"<>|\n\t]', "", short_prompt)
            short_prompt = (
                short_prompt[:64] if len(short_prompt) > 64 else short_prompt
            )

            metadata = PngImagePlugin.PngInfo()

            metadata.add_text("Prompt: ", str(prompt))
            metadata.add_text("Negative prompt: ", str(neg_prompt))
            metadata.add_text("Steps: ", str(steps))
            metadata.add_text("Sampler: ", str(sched_name))
            metadata.add_text("CFG scale: ", str(guidance_scale))
            metadata.add_text("Seed: ", str(seeds[i]))
            metadata.add_text("Size: ", str(f"{width}x{height}"))
            metadata.add_text("Model: ", str(model_name))
            metadata.add_text("Iteration size: ", str(iteration_count))
            metadata.add_text("Batch Size: ", str(batch_size))
            metadata.add_text("Eta: ", str(eta))
            if current_pipe == "img2img":
                metadata.add_text("Denoise: ", str(denoise_strength))

            # img2img color transfer
            if (
                colortransfer is True
                and loopback is False
                and current_pipe == "img2img"
            ):
                for j in range(batch_size):
                    batch_images[j] = transfer_colour(
                        init_image, batch_images[j], transfer_methods
                    )
            elif (
                colortransfer is True
                and loopback is True
                and current_pipe == "img2img"
            ):
                for j in range(batch_size):
                    batch_images[j] = transfer_colour(
                        init_image, batch_images[j], transfer_methods
                    )
                loopback_image = batch_images[0]
            elif (
                colortransfer is False
                and loopback is True
                and current_pipe == "img2img"
            ):
                loopback_image = batch_images[0]

            # inpaint color transfer
            elif colortransfer is True and current_pipe == "inpaint":
                for j in range(batch_size):
                    batch_images[j] = transfer_colour(
                        init_image, batch_images[j], transfer_methods
                    )

            if savemask is True and current_pipe == "inpaint":
                saved_mask = PIL.ImageOps.invert(init_mask)
                saved_mask.save(
                    os.path.join(
                        output_path + f"/masks/",
                        f"{next_index + i:06}-"
                        f"00."
                        f"{short_prompt}_"
                        f"{seeds[i]}_"
                        f"{guidance_scale}g_"
                        f"{width}x"
                        f"{height}_"
                        f"{steps}s_"
                        f"{sched_short_name} "
                        f"mask."
                        f"{image_format}",
                    ),
                    optimize=True,
                    pnginfo=metadata,
                )

            if loopback is True:
                # png output
                if image_format == "png":
                    loopback_image.save(
                        os.path.join(
                            output_path,
                            f"{next_index + i:06}-"
                            f"00."
                            f"{short_prompt}_"
                            f"{seeds[i]}_"
                            f"{guidance_scale}g_"
                            f"{width}x"
                            f"{height}_"
                            f"{steps}s_"
                            f"{sched_short_name}."
                            f"{image_format}",
                        ),
                        optimize=True,
                        pnginfo=metadata,
                    )
                # jpg output
                elif image_format == "jpg":
                    loopback_image.save(
                        os.path.join(
                            output_path,
                            f"{next_index + i:06}-"
                            f"00."
                            f"{short_prompt}_"
                            f"{seeds[i]}_"
                            f"{guidance_scale}g_"
                            f"{width}x"
                            f"{height}_"
                            f"{steps}s_"
                            f"{sched_short_name}."
                            f"{image_format}",
                        ),
                        quality=95,
                        subsampling=0,
                        optimize=True,
                        progressive=True,
                    )
            elif loopback is False:
                # png output
                if image_format == "png":
                    for j in range(batch_size):
                        batch_images[j].save(
                            os.path.join(
                                output_path,
                                f"{next_index + i:06}-"
                                f"{j:02}."
                                f"{short_prompt}_"
                                f"{seeds[i]}_"
                                f"{guidance_scale}g_"
                                f"{width}x"
                                f"{height}_"
                                f"{steps}s_"
                                f"{sched_short_name}."
                                f"{image_format}",
                            ),
                            optimize=True,
                            pnginfo=metadata,
                        )
                # jpg output
                elif image_format == "jpg":
                    for j in range(batch_size):
                        batch_images[j].save(
                            os.path.join(
                                output_path,
                                f"{next_index + i:06}-"
                                f"{j:02}."
                                f"{short_prompt}_"
                                f"{seeds[i]}_"
                                f"{guidance_scale}g_"
                                f"{width}x"
                                f"{height}_"
                                f"{steps}s_"
                                f"{sched_short_name}."
                                f"{image_format}",
                            ),
                            quality=95,
                            subsampling=0,
                            optimize=True,
                            progressive=True,
                        )
            images.extend(batch_images)
            time_taken = time_taken + (finish - start)

    # video
    elif video is True:
        if firststep > laststep:
            step_direction = -1
            ffmpeg_start = laststep
            reversed_or_not = " -vf reverse"
        else:
            step_direction = 1
            ffmpeg_start = firststep
            reversed_or_not = ""
        for step in range(
            firststep, (laststep + step_direction), step_direction
        ):
            print(f"step {step}/{laststep} for video frames")

            short_prompt = prompt.strip('<>:"/\\|?*\n\t')
            short_prompt = re.sub(r'[\\/*?:"<>|\n\t]', "", short_prompt)
            short_prompt = (
                short_prompt[:64] if len(short_prompt) > 32 else short_prompt
            )
            frames_path = (
                output_path + f"/videoframes/"
                f"{short_prompt}_"
                f"{seed}_"
                f"{guidance_scale}g_"
                f"{width}x"
                f"{height}_"
                f"{firststep}-"
                f"{laststep}s_"
                f"{sched_short_name}_"
                f"{fps}fps"
            )
            os.makedirs(frames_path, exist_ok=True)

            try:
                image_number

            except UnboundLocalError:
                image_number = None

            if image_number is None:
                image_number = 1

            info = (
                f"{next_index + image_number:06} | "
                f"prompt: {prompt} "
                f"negative prompt: {neg_prompt} | "
                f"scheduler: {sched_name} "
                f"model: {model_name} "
                f"iteration size: {iteration_count} "
                f"batch size: {batch_size} "
                f"steps: {step} "
                f"scale: {guidance_scale} "
                f"width: {width} "
                f"height: {height} "
                f"eta: {eta} "
                f"seed: {seed}"
            )
            if current_pipe == "img2img":
                info = info + f" denoise: {denoise_strength}"
            with open(os.path.join(frames_path, "history.txt"), "a") as log:
                log.write(info + "\n")

            # create generator object from seed
            rng = np.random.RandomState(seed)

            if current_pipe == "txt2img":
                start = time.time()
                batch_images = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=step,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=batch_size,
                    generator=rng,
                ).images
                finish = time.time()
            elif current_pipe == "img2img":
                start = time.time()
                batch_images = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    image=init_image,
                    num_inference_steps=step,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    strength=denoise_strength,
                    num_images_per_prompt=batch_size,
                    generator=rng,
                ).images
                finish = time.time()
            elif current_pipe == "inpaint":
                start = time.time()
                batch_images = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    image=init_image,
                    mask_image=init_mask,
                    height=height,
                    width=width,
                    num_inference_steps=step,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=batch_size,
                    generator=rng,
                ).images
                finish = time.time()

            short_prompt = prompt.strip('<>:"/\\|?*\n\t')
            short_prompt = re.sub(r'[\\/*?:"<>|\n\t]', "", short_prompt)
            short_prompt = (
                short_prompt[:64] if len(short_prompt) > 32 else short_prompt
            )

            metadata = PngImagePlugin.PngInfo()

            metadata.add_text("Prompt: ", str(prompt))
            metadata.add_text("Negative prompt: ", str(neg_prompt))
            metadata.add_text("Steps: ", str(step))
            metadata.add_text("Sampler: ", str(sched_name))
            metadata.add_text("CFG scale: ", str(guidance_scale))
            metadata.add_text("Seed: ", str(seed))
            metadata.add_text("Size: ", str(f"{width}x{height}"))
            metadata.add_text("Model: ", str(model_name))
            metadata.add_text("Iteration size: ", str(iteration_count))
            metadata.add_text("Batch Size: ", str(batch_size))
            metadata.add_text("Eta: ", str(eta))
            if current_pipe == "img2img":
                metadata.add_text("Denoise: ", str(denoise_strength))

            # png output
            if image_format == "png":
                for j in range(batch_size):
                    batch_images[j].save(
                        os.path.join(
                            frames_path,
                            f"{next_index + image_number:06}-"
                            f"{j:02}."
                            f"{short_prompt}_"
                            f"{seed}_"
                            f"{guidance_scale}g_"
                            f"{width}x"
                            f"{height}_"
                            f"{sched_short_name}."
                            f"{image_format}",
                        ),
                        optimize=True,
                        pnginfo=metadata,
                    )
                    image_number = image_number + 1
            # jpg output
            elif image_format == "jpg":
                for j in range(batch_size):
                    batch_images[j].save(
                        os.path.join(
                            frames_path,
                            f"{next_index + image_number:06}-"
                            f"{j:02}."
                            f"{short_prompt}_"
                            f"{seed}_"
                            f"{guidance_scale}g_"
                            f"{width}x"
                            f"{height}_"
                            f"{sched_short_name}."
                            f"{image_format}",
                        ),
                        quality=95,
                        subsampling=0,
                        optimize=True,
                        progressive=True,
                    )
                    image_number = image_number + 1

            images.extend(batch_images)
            time_taken = time_taken + (finish - start)

    time_taken = time_taken / 60.0
    if iteration_count > 1 or video is True:
        status = (
            f"Run indexes {next_index:06} "
            f"to {next_index + iteration_count - 1:06} "
            f"took {time_taken:.1f} minutes "
            f"to generate {iteration_count} "
            f"iterations with batch size of {batch_size}. "
            f"seeds: " + np.array2string(seeds, separator=",")
        )
    else:
        status = (
            f"Run index {next_index:06} "
            f"took {time_taken:.1f} minutes "
            f"to generate a batch size of {batch_size}. "
            f"seed: {seeds[0]}"
        )
    short_prompt = prompt.strip('<>:"/\\|?*\n\t')
    short_prompt = re.sub(r'[\\/*?:"<>|\n\t]', "", short_prompt)
    short_prompt = (
        short_prompt[:64] if len(short_prompt) > 32 else short_prompt
    )
    frames_path = (
        output_path + f"/videoframes/"
        f"{short_prompt}_"
        f"{seed}_"
        f"{guidance_scale}g_"
        f"{width}x"
        f"{height}_"
        f"{firststep}-"
        f"{laststep}s_"
        f"{sched_short_name}_"
        f"{fps}fps"
    )

    if video is True:
        os.system(
            f"ffmpeg "
            f"-f image2 "
            f"-r "
            f"{fps} "
            f"-start_number "
            f"{ffmpeg_start} "
            f'-i "'
            f"{frames_path}/%06d-00."
            f"{short_prompt}_"
            f"{seed}_"
            f"{guidance_scale}g_"
            f"{width}x"
            f"{height}_"
            f"{sched_short_name}."
            f'{image_format}" '
            f"-vcodec libx264 "
            f"-crf 7.5 "
            f"-preset veryslow"
            f"{reversed_or_not} "
            f'"videooutput/'
            f"{short_prompt}_"
            f"{seed}_"
            f"{guidance_scale}g_"
            f"{firststep}-"
            f"{laststep}s_"
            f"{sched_short_name}_"
            f'{fps}fps.mp4"'
        )
        print(
            f"ffmpeg "
            f"-f image2 "
            f"-r "
            f"{fps} "
            f"-start_number "
            f"{ffmpeg_start} "
            f'-i "'
            f"{frames_path}/%06d-00."
            f"{short_prompt}_"
            f"{seed}_"
            f"{guidance_scale}g_"
            f"{width}x"
            f"{height}_"
            f"{sched_short_name}."
            f'{image_format}" '
            f"-vcodec libx264 "
            f"-crf 7.5 "
            f"-preset veryslow"
            f"{reversed_or_not} "
            f'"videooutput/'
            f"{short_prompt}_"
            f"{seed}_"
            f"{guidance_scale}g_"
            f"{firststep}-"
            f"{laststep}s_"
            f"{sched_short_name}_"
            f'{fps}fps.mp4"'
        )

    return images, status


def resize_and_crop(input_image: PIL.Image.Image, height: int, width: int):
    input_width, input_height = input_image.size
    if height / width > input_height / input_width:
        adjust_width = int(input_width * height / input_height)
        input_image = input_image.resize((adjust_width, height))
        left = (adjust_width - width) // 2
        right = left + width
        input_image = input_image.crop((left, 0, right, height))
    else:
        adjust_height = int(input_height * width / input_width)
        input_image = input_image.resize((width, adjust_height))
        top = (adjust_height - height) // 2
        bottom = top + height
        input_image = input_image.crop((0, top, width, bottom))
    return input_image


def tagger_predict(image, score_threshold):
    # tagger_model_path = "deepdanbooru.onnx"
    tagger_model_path = hf_hub_download(
        repo_id="skytnt/deepdanbooru_onnx", filename="deepdanbooru.onnx"
    )
    eprovider = "CPUExecutionProvider"
    # eprovider="DmlExecutionProvider"
    tagger_model = ort.InferenceSession(
        tagger_model_path, providers=[eprovider]
    )
    tagger_model_meta = tagger_model.get_modelmeta().custom_metadata_map
    tagger_tags = eval(tagger_model_meta["tags"])
    s = 512
    h, w = image.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(
        image,
        ph // 2,
        ph - ph // 2,
        pw // 2,
        pw - pw // 2,
        cv2.BORDER_REPLICATE,
    )
    image = image.astype(np.float32) / 255
    image = image[np.newaxis, :]
    probs = tagger_model.run(None, {"input_1": image})[0][0]
    probs = probs.astype(np.float32)
    tags = []
    probabilities = []
    for prob, label in zip(probs.tolist(), tagger_tags):
        if prob < score_threshold:
            continue
        tags.append(label)
        probabilities.append(prob)
    del tagger_model
    del tagger_model_meta
    del tagger_tags
    gc.collect()
    return tags, probabilities


def danbooru_click(extras_image):
    img = cv2.cvtColor(np.array(extras_image), cv2.COLOR_RGB2BGR)
    img = img[:, :, ::-1].copy()
    dh, dw = img.shape[:-1]
    tags, probs = tagger_predict(img, 0.25)
    newprompt = ""
    for x in tags:
        if not "rating" in x:
            newprompt += x + ", "
    newprompt = newprompt.strip(", ")
    repdict = {"\\": "\\\\", "(": "\\(", ")": "\\)"}
    for key, value in repdict.items():
        newprompt = newprompt.replace(key, value)
    print(newprompt)
    global current_tab
    if current_tab == 0:
        return {interrogate_prompt: newprompt}
    elif current_tab == 1:
        return {interrogate_prompt: newprompt}
    elif current_tab == 2:
        return {interrogate_prompt: newprompt}


def clip_interrogator_click(extras_image):
    global current_tab
    config = Config(
        clip_model_path="cache",
        cache_path="cache",
        clip_model_name="ViT-L-14/openai",
        download_cache=False,
        chunk_size=384,
        blip_image_eval_size=512,
    )
    ci_vitl = Interrogator(config)
    ci_vitl.clip_model = ci_vitl.clip_model.to("cpu")
    ci = ci_vitl

    newprompt = ci.interrogate(extras_image)
    print(newprompt)
    gc.collect()
    if current_tab == 0:
        return {interrogate_prompt: newprompt}
    elif current_tab == 1:
        return {interrogate_prompt: newprompt}
    elif current_tab == 2:
        return {interrogate_prompt: newprompt}


def clip_interrogator_negative_click(extras_image):
    global current_tab
    config = Config(
        clip_model_path="cache",
        clip_model_name="ViT-L-14/openai",
        download_cache=False,
        chunk_size=384,
        blip_image_eval_size=512,
    )
    ci_vitl = Interrogator(config)
    ci_vitl.clip_model = ci_vitl.clip_model.to("cpu")
    ci = ci_vitl

    newnegativeprompt = ci.interrogate_negative(extras_image)
    print(newnegativeprompt)
    gc.collect()
    if current_tab == 0:
        return {interrogate_negative_prompt: newnegativeprompt}
    elif current_tab == 1:
        return {interrogate_negative_prompt: newnegativeprompt}
    elif current_tab == 2:
        return {interrogate_negative_prompt: newnegativeprompt}


def release_click():
    global pipe
    global scheduler

    scheduler = None
    pipe = None
    gc.collect()

    print("pipe and scheduler released from memory")


def transfer_colour(input_image, output_image, transfer_methods):
    input_image_array = np.array(input_image)
    output_image_array = np.array(output_image)

    if transfer_methods == "lhm":
        print("applying lhm colour transfer")
        image_transfer_array = colortrans.transfer_lhm(
            output_image_array, input_image_array
        )
    if transfer_methods == "reinhard":
        print("applying reinhard colour transfer")
        image_transfer_array = colortrans.transfer_reinhard(
            output_image_array, input_image_array
        )
    if transfer_methods == "pccm":
        print("applying pccm colour transfer")
        image_transfer_array = colortrans.transfer_pccm(
            output_image_array, input_image_array
        )

    image_transfer = Image.fromarray(image_transfer_array)

    return image_transfer


def video_parameter_select(video_parameter):
    print("placeholder")


def clear_click():
    global current_tab
    if current_tab == 0:
        return {
            prompt_t0: "",
            neg_prompt_t0: "",
            sch_t0: "DEIS",
            iter_t0: 1,
            batch_t0: 1,
            steps_t0: 16,
            guid_t0: 3.5,
            height_t0: 512,
            width_t0: 512,
            eta_t0: 0.0,
            seed_t0: "",
            fmt_t0: "png",
            video_t0: False,
            fps_t0: 5,
            firststep_t0: 1,
            laststep_t0: 32,
        }
    elif current_tab == 1:
        return {
            prompt_t1: "",
            neg_prompt_t1: "",
            sch_t1: "DEIS",
            image_t1: None,
            iter_t1: 1,
            batch_t1: 1,
            steps_t1: 16,
            guid_t1: 3.5,
            height_t1: 512,
            width_t1: 512,
            eta_t1: 0.0,
            denoise_t1: 0.75,
            seed_t1: "",
            fmt_t1: "png",
            video_t1: False,
            fps_t1: 5,
            firststep_t1: 1,
            laststep_t1: 32,
            loopback_t1: False,
            loopback_halving_t1: False,
            colortransfer_t1: False,
            transfer_methods_t1: "lhm",
            transfer_amounts_t1: "each",
        }
    elif current_tab == 2:
        return {
            prompt_t2: "",
            neg_prompt_t2: "",
            sch_t2: "DEIS",
            legacy_t2: False,
            savemask_t2: False,
            image_t2: None,
            mask_t2: None,
            iter_t2: 1,
            batch_t2: 1,
            steps_t2: 16,
            guid_t2: 3.5,
            height_t2: 512,
            width_t2: 512,
            eta_t2: 0.0,
            seed_t2: "",
            fmt_t2: "png",
            video_t2: False,
            fps_t2: 5,
            firststep_t2: 1,
            laststep_t2: 32,
            colortransfer_t2: False,
            transfer_methods_t2: "lhm",
            transfer_amounts_t2: "each",
        }


def generate_click(
    model_drop,
    prompt_t0,
    neg_prompt_t0,
    sch_t0,
    iter_t0,
    batch_t0,
    steps_t0,
    guid_t0,
    height_t0,
    width_t0,
    eta_t0,
    seed_t0,
    fmt_t0,
    video_t0,
    fps_t0,
    firststep_t0,
    laststep_t0,
    prompt_t1,
    neg_prompt_t1,
    image_t1,
    sch_t1,
    iter_t1,
    batch_t1,
    steps_t1,
    guid_t1,
    height_t1,
    width_t1,
    eta_t1,
    denoise_t1,
    seed_t1,
    fmt_t1,
    video_t1,
    fps_t1,
    firststep_t1,
    laststep_t1,
    loopback_t1,
    loopback_halving_t1,
    colortransfer_t1,
    transfer_methods_t1,
    transfer_amounts_t1,
    prompt_t2,
    neg_prompt_t2,
    sch_t2,
    legacy_t2,
    savemask_t2,
    image_t2,
    mask_t2,
    iter_t2,
    batch_t2,
    steps_t2,
    guid_t2,
    height_t2,
    width_t2,
    eta_t2,
    seed_t2,
    fmt_t2,
    video_t2,
    fps_t2,
    firststep_t2,
    laststep_t2,
    colortransfer_t2,
    transfer_methods_t2,
    transfer_amounts_t2,
):
    global model_name
    global provider
    global current_tab
    global current_pipe
    global current_legacy
    global release_memory_after_generation
    global release_memory_on_change
    global scheduler
    global pipe

    # reset scheduler and pipeline if model is different
    if model_name != model_drop:
        model_name = model_drop
        scheduler = None
        pipe = None
        gc.collect()
    model_path = os.path.join("model", model_name)

    # select which scheduler depending on current tab
    if current_tab == 0:
        sched_name = sch_t0
    elif current_tab == 1:
        sched_name = sch_t1
    elif current_tab == 2:
        sched_name = sch_t2
    else:
        raise Exception("Unknown tab")

    if sched_name == "PNDM" and type(scheduler) is not PNDMScheduler:
        scheduler = PNDMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "LMS" and type(scheduler) is not LMSDiscreteScheduler:
        scheduler = LMSDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "DDIM" and type(scheduler) is not DDIMScheduler:
        scheduler = DDIMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "DDPM" and type(scheduler) is not DDPMScheduler:
        scheduler = DDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif (
        sched_name == "Euler" and type(scheduler) is not EulerDiscreteScheduler
    ):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif (
        sched_name == "EulerA"
        and type(scheduler) is not EulerAncestralDiscreteScheduler
    ):
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif (
        sched_name == "DPMSM"
        and type(scheduler) is not DPMSolverMultistepScheduler
    ):
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif (
        sched_name == "DPMSS"
        and type(scheduler) is not DPMSolverSinglestepScheduler
    ):
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif (
        sched_name == "DEIS" and type(scheduler) is not DEISMultistepScheduler
    ):
        scheduler = DEISMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif (
        sched_name == "KDPM2" and type(scheduler) is not KDPM2DiscreteScheduler
    ):
        scheduler = KDPM2DiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif (
        sched_name == "KDPM2A"
        and type(scheduler) is not KDPM2AncestralDiscreteScheduler
    ):
        scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "Heun" and type(scheduler) is not HeunDiscreteScheduler:
        scheduler = HeunDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

    # select which pipeline depending on current tab
    if current_tab == 0:
        if (
            current_pipe == ("img2img" or "inpaint")
            and release_memory_on_change
        ):
            pipe = None
            gc.collect()
        if current_pipe != "txt2img" or pipe is None:
            if textenc_on_cpu and vae_on_cpu:
                print("Using CPU Text Encoder")
                print("Using CPU VAE")
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder"
                )
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder"
                )
                pipe = OnnxStableDiffusionPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc,
                    vae_decoder=cpuvaedec,
                    vae_encoder=None,
                )
            elif textenc_on_cpu:
                print("Using CPU Text Encoder")
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder"
                )
                pipe = OnnxStableDiffusionPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc,
                )
            elif vae_on_cpu:
                print("Using CPU VAE")
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder"
                )
                pipe = OnnxStableDiffusionPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    vae_decoder=cpuvaedec,
                    vae_encoder=None,
                )
            else:
                pipe = OnnxStableDiffusionPipeline.from_pretrained(
                    model_path, provider=provider, scheduler=scheduler
                )
        current_pipe = "txt2img"
    elif current_tab == 1:
        if (
            current_pipe == ("txt2img" or "inpaint")
            and release_memory_on_change
        ):
            pipe = None
            gc.collect()
        if current_pipe != "img2img" or pipe is None:
            if textenc_on_cpu and vae_on_cpu:
                print("Using CPU Text Encoder")
                print("Using CPU VAE")
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder"
                )
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder"
                )
                cpuvaeenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_encoder"
                )
                pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc,
                    vae_decoder=cpuvaedec,
                    vae_encoder=cpuvaeenc,
                )
            elif textenc_on_cpu:
                print("Using CPU Text Encoder")
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder"
                )
                pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc,
                )
            elif vae_on_cpu:
                print("Using CPU VAE")
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder"
                )
                cpuvaeenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_encoder"
                )
                pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    vae_decoder=cpuvaedec,
                    vae_encoder=cpuvaeenc,
                )
            else:
                pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path, provider=provider, scheduler=scheduler
                )
        current_pipe = "img2img"
    elif current_tab == 2:
        if (
            current_pipe == ("txt2img" or "img2img")
            and release_memory_on_change
        ):
            pipe = None
            gc.collect()
        if (
            current_pipe != "inpaint"
            or pipe is None
            or current_legacy != legacy_t2
        ):
            if legacy_t2:
                if textenc_on_cpu and vae_on_cpu:
                    print("Using CPU Text Encoder")
                    print("Using CPU VAE")
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder"
                    )
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder"
                    )
                    cpuvaeenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_encoder"
                    )
                    pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc,
                        vae_decoder=cpuvaedec,
                        vae_encoder=cpuvaeenc,
                    )
                elif textenc_on_cpu:
                    print("Using CPU Text Encoder")
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder"
                    )
                    pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc,
                    )
                elif vae_on_cpu:
                    print("Using CPU VAE")
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder"
                    )
                    cpuvaeenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_encoder"
                    )
                    pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        vae_decoder=cpuvaedec,
                        vae_encoder=cpuvaeenc,
                    )
                else:
                    pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                        model_path, provider=provider, scheduler=scheduler
                    )
            else:
                if textenc_on_cpu and vae_on_cpu:
                    print("Using CPU Text Encoder")
                    print("Using CPU VAE")
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder"
                    )
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder"
                    )
                    cpuvaeenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_encoder"
                    )
                    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc,
                        vae_decoder=cpuvaedec,
                        vae_encoder=cpuvaeenc,
                    )
                elif textenc_on_cpu:
                    print("Using CPU Text Encoder")
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder"
                    )
                    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc,
                    )
                elif vae_on_cpu:
                    print("Using CPU VAE")
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder"
                    )
                    cpuvaeenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_encoder"
                    )
                    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        vae_decoder=cpuvaedec,
                        vae_encoder=cpuvaeenc,
                    )
                else:
                    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                        model_path, provider=provider, scheduler=scheduler
                    )
        current_pipe = "inpaint"
        current_legacy = legacy_t2

    # manual garbage collection
    gc.collect()

    # modifying the methods in the pipeline object
    if type(pipe.scheduler) is not type(scheduler):
        pipe.scheduler = scheduler
    if version.parse(_df_version) >= version.parse("0.8.0"):
        safety_checker = None
    else:
        safety_checker = lambda images, **kwargs: (
            images,
            [False] * len(images),
        )
    pipe.safety_checker = safety_checker
    pipe._encode_prompt = functools.partial(lpw_pipe._encode_prompt, pipe)

    # run the pipeline with the correct parameters
    if current_tab == 0:
        images, status = run_diffusers(
            prompt_t0,
            neg_prompt_t0,
            None,
            None,
            iter_t0,
            batch_t0,
            steps_t0,
            guid_t0,
            height_t0,
            width_t0,
            eta_t0,
            0,
            seed_t0,
            fmt_t0,
            False,
            video_t0,
            fps_t0,
            firststep_t0,
            laststep_t0,
            False,
            False,
            False,
            transfer_methods_t1,
            transfer_amounts_t1,
        )
    elif current_tab == 1:
        # input image resizing
        input_image = image_t1.convert("RGB")
        input_image = resize_and_crop(input_image, height_t1, width_t1)

        # adjust steps to account for denoise
        steps_t1_old = steps_t1
        steps_t1 = ceil(steps_t1 / denoise_t1)
        if (steps_t1 > 1000) and (sch_t1 == "DPMSM" or "DPMSS" or "DEIS"):
            steps_t1_unreduced = steps_t1
            steps_t1 = 1000
            print()
            print(
                f"Adjusting steps to account for denoise. From {steps_t1_old} "
                f"to {steps_t1_unreduced} steps internally."
            )
            print(
                f"Without adjustment the actual step count would be "
                f"~{ceil(steps_t1_old * denoise_t1)} steps."
            )
            print()
            print(
                f"INTERNAL STEP COUNT EXCEEDS 1000 MAX FOR DPMSM, DPMSS, "
                f"or DEIS. INTERNAL STEPS WILL BE REDUCED TO 1000."
            )
            print()
        else:
            print()
            print(
                f"Adjusting steps to account for denoise. From {steps_t1_old} "
                f"to {steps_t1} steps internally."
            )
            print(
                f"Without adjustment the actual step count would be "
                f"~{ceil(steps_t1_old * denoise_t1)} steps."
            )
            print()

        images, status = run_diffusers(
            prompt_t1,
            neg_prompt_t1,
            input_image,
            None,
            iter_t1,
            batch_t1,
            steps_t1,
            guid_t1,
            height_t1,
            width_t1,
            eta_t1,
            denoise_t1,
            seed_t1,
            fmt_t1,
            False,
            video_t1,
            fps_t1,
            firststep_t1,
            laststep_t1,
            loopback_t1,
            loopback_halving_t1,
            colortransfer_t1,
            transfer_methods_t1,
            transfer_amounts_t1,
        )
    elif current_tab == 2:
        input_image = image_t2["image"].convert("RGB")
        input_image = resize_and_crop(input_image, height_t2, width_t2)

        if mask_t2 is not None:
            print("using uploaded mask")
            input_mask = mask_t2.convert("RGB")
            input_mask = resize_and_crop(input_mask, height_t2, width_t2)
        else:
            print("using painted mask")
            input_mask = image_t2["mask"].convert("RGB")
            input_mask = resize_and_crop(input_mask, height_t2, width_t2)

        # adjust steps to account for legacy inpaint only using ~80% of set steps
        if legacy_t2 is True:
            steps_t2_old = steps_t2
            if steps_t2 < 5:
                steps_t2 = steps_t2 + 1
            elif steps_t2 >= 5:
                steps_t2 = int((steps_t2 / 0.7989) + 1)
            print()
            print(
                f"Adjusting steps for legacy inpaint. From {steps_t2_old} "
                f"to {steps_t2} internally."
            )
            print(
                f"Without adjustment the actual step count would be "
                f"~{int(steps_t2_old * 0.8)} steps."
            )
            print()

        images, status = run_diffusers(
            prompt_t2,
            neg_prompt_t2,
            input_image,
            input_mask,
            iter_t2,
            batch_t2,
            steps_t2,
            guid_t2,
            height_t2,
            width_t2,
            eta_t2,
            0,
            seed_t2,
            fmt_t2,
            legacy_t2,
            savemask_t2,
            video_t2,
            fps_t2,
            firststep_t2,
            laststep_t2,
            False,
            False,
            colortransfer_t2,
            transfer_methods_t2,
            transfer_amounts_t2,
        )

    gc.collect()
    if release_memory_after_generation:
        pipe = None
        gc.collect()

    return images, status


def select_tab0():
    global current_tab
    current_tab = 0


def select_tab1():
    global current_tab
    current_tab = 1


def select_tab2():
    global current_tab
    current_tab = 2


def choose_sch(sched_name: str):
    if sched_name == "DDIM":
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)


def make_video1(video: bool):
    if video is True:
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)


def make_video2(video: bool):
    if video is True:
        return gr.update(interactive=False)
    else:
        return gr.update(interactive=True)


def make_loopback(loopback: bool):
    if loopback is True:
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)


def clear_temp_files(sig, frame):
    print(f"Cleaning temporary files...", flush=True)
    shutil.rmtree(tempfile.gettempdir(), ignore_errors=True, onerror=None)
    exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="gradio interface for ONNX based Stable Diffusion"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="run ONNX with CPU",
    )
    parser.add_argument(
        "--release-memory-after-generation",
        action="store_true",
        default=False,
        help="de-allocate the pipeline and release memory after generation",
    )
    parser.add_argument(
        "--release-memory-on-change",
        action="store_true",
        default=False,
        help="de-allocate the pipeline and release memory allocated when "
        "changing pipelines.",
    )
    parser.add_argument(
        "--cpu-textenc",
        action="store_true",
        default=False,
        help="Run Text Encoder on CPU, saves VRAM by running Text Encoder on CPU",
    )
    parser.add_argument(
        "--cpu-vae",
        action="store_true",
        default=False,
        help="Run VAE on CPU, saves VRAM by running VAE on CPU",
    )
    args = parser.parse_args()

    # variables for ONNX pipelines
    model_name = None
    provider = (
        "CPUExecutionProvider" if args.cpu_only else "DmlExecutionProvider"
    )
    current_tab = 0
    current_pipe = "txt2img"
    current_legacy = False
    release_memory_after_generation = args.release_memory_after_generation
    release_memory_on_change = args.release_memory_on_change
    textenc_on_cpu = args.cpu_textenc
    vae_on_cpu = args.cpu_vae

    # diffusers objects
    scheduler = None
    pipe = None

    # check versions
    is_v_0_12 = version.parse(_df_version) >= version.parse("0.12.0")
    is_v_dev = version.parse(_df_version).is_prerelease

    # prerelease version use warning
    if is_v_dev:
        print(
            "You are using diffusers "
            + str(version.parse(_df_version))
            + " (prerelease)\n"
            + "If you experience unexpected errors please run "
            + "`pip install diffusers --force-reinstall`."
        )

    # custom css
    custom_css = """
    #gen_button {height: 90px}
    #image_init {min-height: 400px}
    #image_init [data-testid="image"], #image_init [data-testid="image"] > div {min-height: 400px}
    #image_inpaint {min-height: 400px}
    #image_inpaint [data-testid="image"], #image_inpaint [data-testid="image"] > div {min-height: 400px}
    #image_inpaint .touch-none {display: flex}
    #image_inpaint img {display: block; max-width: 84%}
    #image_inpaint canvas {max-width: 84%; object-fit: contain}
    """

    # search the model folder
    model_dir = "model"
    model_list = []
    with os.scandir(model_dir) as scan_it:
        for entry in scan_it:
            if entry.is_dir():
                model_list.append(entry.name)

    default_model = model_list[0] if len(model_list) > 0 else None

    if is_v_0_12:
        from diffusers import (
            OnnxStableDiffusionInpaintPipeline,
            OnnxRuntimeModel,
            OnnxStableDiffusionInpaintPipelineLegacy,
            DDPMScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            KDPM2DiscreteScheduler,
            KDPM2AncestralDiscreteScheduler,
            HeunDiscreteScheduler,
            DPMSolverSinglestepScheduler,
            DEISMultistepScheduler,
        )

        sched_list = [
            "DEIS",
            "DPMSM",
            "DPMSS",
            "Euler",
            "EulerA",
            "Heun",
            "KDPM2",
            "KDPM2A",
            "DDIM",
            "LMS",
            "PNDM",
            "DDPM",
        ]
    else:
        sched_list = ["DDIM", "LMS", "PNDM"]

    transfer_methods_list = ["lhm", "reinhard", "pccm"]
    transfer_amounts_list = ["each", "final"]

    # create gradio block
    title = "Stable Diffusion ONNX"
    with gr.Blocks(title=title, css=custom_css) as demo:
        with gr.Row():
            with gr.Column(scale=13, min_width=650):
                model_drop = gr.Dropdown(
                    model_list,
                    value=default_model,
                    label="model folder",
                    interactive=True,
                )
            with gr.Column(scale=11, min_width=550):
                with gr.Row():
                    gen_btn = gr.Button(
                        "Generate", variant="primary", elem_id="gen_button"
                    )
                    clear_btn = gr.Button("Clear", elem_id="gen_button")
                    release_btn = gr.Button(
                        "Release Memory", elem_id="gen_button"
                    )
        with gr.Row():
            with gr.Column(scale=13, min_width=650):
                with gr.Tab(label="txt2img") as tab0:
                    prompt_t0 = gr.Textbox(value="", lines=2, label="prompt")
                    neg_prompt_t0 = gr.Textbox(
                        value="",
                        lines=2,
                        label="negative prompt",
                    )
                    sch_t0 = gr.Radio(
                        sched_list, value="DEIS", label="scheduler"
                    )
                    with gr.Row():
                        iter_t0 = gr.Slider(
                            1, 300, value=1, step=1, label="iteration count"
                        )
                        batch_t0 = gr.Slider(
                            1, 4, value=1, step=1, label="batch size"
                        )
                    steps_t0 = gr.Slider(
                        1, 300, value=16, step=1, label="steps"
                    )
                    guid_t0 = gr.Slider(
                        1.01, 50, value=3.5, step=0.01, label="guidance"
                    )
                    width_t0 = gr.Slider(
                        256, 2048, value=512, step=64, label="width"
                    )
                    height_t0 = gr.Slider(
                        256, 2048, value=512, step=64, label="height"
                    )
                    eta_t0 = gr.Slider(
                        0,
                        1,
                        value=0.0,
                        step=0.01,
                        label="DDIM eta",
                        interactive=False,
                    )
                    seed_t0 = gr.Textbox(value="", max_lines=1, label="seed")
                    fmt_t0 = gr.Radio(
                        ["png", "jpg"], value="png", label="image format"
                    )
                    with gr.Row():
                        video_t0 = gr.Checkbox(
                            value=False, label="create video"
                        )
                    fps_t0 = gr.Slider(
                        1,
                        120,
                        value=5,
                        step=0.01,
                        label="framerate",
                        interactive=False,
                    )
                    with gr.Row():
                        firststep_t0 = gr.Slider(
                            1,
                            300,
                            value=1,
                            step=1,
                            label="first step",
                            interactive=False,
                        )
                        laststep_t0 = gr.Slider(
                            1,
                            300,
                            value=32,
                            step=1,
                            label="last step",
                            interactive=False,
                        )
                with gr.Tab(label="img2img") as tab1:
                    prompt_t1 = gr.Textbox(value="", lines=2, label="prompt")
                    neg_prompt_t1 = gr.Textbox(
                        value="",
                        lines=2,
                        label="negative prompt",
                    )
                    sch_t1 = gr.Radio(
                        sched_list, value="DEIS", label="scheduler"
                    )
                    image_t1 = gr.Image(
                        label="input image", type="pil", elem_id="image_init"
                    )
                    with gr.Row():
                        iter_t1 = gr.Slider(
                            1, 300, value=1, step=1, label="iteration count"
                        )
                        batch_t1 = gr.Slider(
                            1, 4, value=1, step=1, label="batch size"
                        )
                    with gr.Row():
                        loopback_t1 = gr.Checkbox(
                            value=False, label="loopback (use iteration count)"
                        )
                        loopback_halving_t1 = gr.Checkbox(
                            value=False,
                            label="halve denoise each " "loopback",
                            interactive=False,
                        )
                    with gr.Row():
                        colortransfer_t1 = gr.Checkbox(
                            value=False,
                            label="colour transfer from base",
                        )
                        transfer_methods_t1 = gr.Radio(
                            transfer_methods_list,
                            value="lhm",
                            label="colour transfer method",
                        )
                        transfer_amounts_t1 = gr.Radio(
                            transfer_amounts_list,
                            value="each",
                            label="amount of colour transfers",
                        )
                    steps_t1 = gr.Slider(
                        1, 300, value=16, step=1, label="steps"
                    )
                    guid_t1 = gr.Slider(
                        1.01, 50, value=3.5, step=0.01, label="guidance"
                    )
                    width_t1 = gr.Slider(
                        256, 2048, value=512, step=64, label="width"
                    )
                    height_t1 = gr.Slider(
                        256, 2048, value=512, step=64, label="height"
                    )
                    denoise_t1 = gr.Slider(
                        0, 1, value=0.75, step=0.01, label="denoise strength"
                    )
                    eta_t1 = gr.Slider(
                        0,
                        1,
                        value=0.0,
                        step=0.01,
                        label="DDIM eta",
                        interactive=False,
                    )
                    seed_t1 = gr.Textbox(value="", max_lines=1, label="seed")
                    fmt_t1 = gr.Radio(
                        ["png", "jpg"], value="png", label="image format"
                    )
                    with gr.Row():
                        video_t1 = gr.Checkbox(
                            value=False, label="create video"
                        )
                    fps_t1 = gr.Slider(
                        1,
                        120,
                        value=5,
                        step=0.01,
                        label="framerate",
                        interactive=False,
                    )
                    with gr.Row():
                        firststep_t1 = gr.Slider(
                            1,
                            300,
                            value=1,
                            step=1,
                            label="first step",
                            interactive=False,
                        )
                        laststep_t1 = gr.Slider(
                            1,
                            300,
                            value=32,
                            step=1,
                            label="last step",
                            interactive=False,
                        )
                with gr.Tab(label="inpainting") as tab2:
                    prompt_t2 = gr.Textbox(value="", lines=2, label="prompt")
                    neg_prompt_t2 = gr.Textbox(
                        value="",
                        lines=2,
                        label="negative prompt",
                    )
                    sch_t2 = gr.Radio(
                        sched_list, value="DEIS", label="scheduler"
                    )
                    legacy_t2 = gr.Checkbox(
                        value=False, label="legacy inpaint"
                    )
                    savemask_t2 = gr.Checkbox(
                        value=False, label="save painted mask"
                    )
                    image_t2 = gr.Image(
                        source="upload",
                        tool="sketch",
                        label="input image",
                        type="pil",
                        elem_id="image_inpaint",
                    )
                    mask_t2 = gr.Image(
                        source="upload",
                        label="input mask",
                        type="pil",
                        invert_colors=True,
                        elem_id="mask_inpaint",
                    )
                    with gr.Row():
                        iter_t2 = gr.Slider(
                            1, 300, value=1, step=1, label="iteration count"
                        )
                        batch_t2 = gr.Slider(
                            1, 4, value=1, step=1, label="batch size"
                        )
                    with gr.Row():
                        colortransfer_t2 = gr.Checkbox(
                            value=False,
                            label="colour transfer from base",
                        )
                        transfer_methods_t2 = gr.Radio(
                            transfer_methods_list,
                            value="lhm",
                            label="colour transfer method",
                        )
                        transfer_amounts_t2 = gr.Radio(
                            transfer_amounts_list,
                            value="each",
                            label="amount of colour transfers",
                        )
                    steps_t2 = gr.Slider(
                        1, 300, value=16, step=1, label="steps"
                    )
                    guid_t2 = gr.Slider(
                        1.01, 50, value=3.5, step=0.01, label="guidance"
                    )
                    width_t2 = gr.Slider(
                        256, 2048, value=512, step=64, label="width"
                    )
                    height_t2 = gr.Slider(
                        256, 2048, value=512, step=64, label="height"
                    )
                    eta_t2 = gr.Slider(
                        0,
                        1,
                        value=0.0,
                        step=0.01,
                        label="DDIM eta",
                        interactive=False,
                    )
                    seed_t2 = gr.Textbox(value="", max_lines=1, label="seed")
                    fmt_t2 = gr.Radio(
                        ["png", "jpg"], value="png", label="image format"
                    )
                    with gr.Row():
                        video_t2 = gr.Checkbox(
                            value=False, label="create video"
                        )
                    fps_t2 = gr.Slider(
                        1,
                        120,
                        value=5,
                        step=0.01,
                        label="framerate",
                        interactive=False,
                    )
                    with gr.Row():
                        firststep_t2 = gr.Slider(
                            1,
                            300,
                            value=1,
                            step=1,
                            label="first step",
                            interactive=False,
                        )
                        laststep_t2 = gr.Slider(
                            1,
                            300,
                            value=32,
                            step=1,
                            label="last step",
                            interactive=False,
                        )
            with gr.Column(scale=11, min_width=550):
                image_out = gr.Gallery(value=None, label="output images")
                status_out = gr.Textbox(value="", label="status")
                extras_image = gr.Image(
                    label="input image", type="pil", elem_id="image_extras"
                )
                with gr.Row():
                    danbooru_btn = gr.Button(
                        "Deepdanbooru", elem_id="deepdb_button"
                    )
                    clip_interrogator_btn = gr.Button(
                        "CLIP Interrogate", elem_id="clip_interrogator_btn"
                    )
                    clip_interrogator_negative_btn = gr.Button(
                        "CLIP Interrogate Negative",
                        elem_id="clip_interrogator_negative_btn",
                    )
                interrogate_prompt = gr.Textbox(
                    value="", lines=2, label="interrogate prompt result"
                )
                interrogate_negative_prompt = gr.Textbox(
                    value="",
                    lines=2,
                    label="interrogate negative prompt " "result",
                )

        # config components
        tab0_inputs = [
            prompt_t0,
            neg_prompt_t0,
            sch_t0,
            iter_t0,
            batch_t0,
            steps_t0,
            guid_t0,
            height_t0,
            width_t0,
            eta_t0,
            seed_t0,
            fmt_t0,
            video_t0,
            fps_t0,
            firststep_t0,
            laststep_t0,
        ]
        tab1_inputs = [
            prompt_t1,
            neg_prompt_t1,
            image_t1,
            sch_t1,
            iter_t1,
            batch_t1,
            steps_t1,
            guid_t1,
            height_t1,
            width_t1,
            eta_t1,
            denoise_t1,
            seed_t1,
            fmt_t1,
            video_t1,
            fps_t1,
            firststep_t1,
            laststep_t1,
            loopback_t1,
            loopback_halving_t1,
            colortransfer_t1,
            transfer_methods_t1,
            transfer_amounts_t1,
        ]
        tab2_inputs = [
            prompt_t2,
            neg_prompt_t2,
            sch_t2,
            legacy_t2,
            savemask_t2,
            image_t2,
            mask_t2,
            iter_t2,
            batch_t2,
            steps_t2,
            guid_t2,
            height_t2,
            width_t2,
            eta_t2,
            seed_t2,
            fmt_t2,
            video_t2,
            fps_t2,
            firststep_t2,
            laststep_t2,
            colortransfer_t2,
            transfer_methods_t2,
            transfer_amounts_t2,
        ]
        all_inputs = [model_drop]
        all_inputs.extend(tab0_inputs)
        all_inputs.extend(tab1_inputs)
        all_inputs.extend(tab2_inputs)
        danbooru_btn.click(
            fn=danbooru_click,
            inputs=[extras_image],
            outputs=interrogate_prompt,
        )
        clip_interrogator_btn.click(
            fn=clip_interrogator_click,
            inputs=[extras_image],
            outputs=interrogate_prompt,
        )
        clip_interrogator_negative_btn.click(
            fn=clip_interrogator_negative_click,
            inputs=[extras_image],
            outputs=interrogate_negative_prompt,
        )

        clear_btn.click(
            fn=clear_click, inputs=None, outputs=all_inputs, queue=False
        )
        gen_btn.click(
            fn=generate_click,
            inputs=all_inputs,
            outputs=[image_out, status_out],
        )
        release_btn.click(
            fn=release_click, inputs=None, outputs=None, queue=False
        )

        tab0.select(fn=select_tab0, inputs=None, outputs=None)
        tab1.select(fn=select_tab1, inputs=None, outputs=None)
        tab2.select(fn=select_tab2, inputs=None, outputs=None)

        sch_t0.change(
            fn=choose_sch, inputs=sch_t0, outputs=eta_t0, queue=False
        )
        sch_t1.change(
            fn=choose_sch, inputs=sch_t1, outputs=eta_t1, queue=False
        )
        sch_t2.change(
            fn=choose_sch, inputs=sch_t2, outputs=eta_t2, queue=False
        )

        # didn't know how to condense these
        video_t0.change(
            fn=make_video1, inputs=video_t0, outputs=fps_t0, queue=False
        )
        video_t0.change(
            fn=make_video1, inputs=video_t0, outputs=firststep_t0, queue=False
        )
        video_t0.change(
            fn=make_video1, inputs=video_t0, outputs=laststep_t0, queue=False
        )
        video_t0.change(
            fn=make_video2, inputs=video_t0, outputs=steps_t0, queue=False
        )
        video_t0.change(
            fn=make_video2, inputs=video_t0, outputs=iter_t0, queue=False
        )
        video_t0.change(
            fn=make_video2, inputs=video_t0, outputs=batch_t0, queue=False
        )

        video_t1.change(
            fn=make_video1, inputs=video_t1, outputs=fps_t1, queue=False
        )
        video_t1.change(
            fn=make_video1, inputs=video_t1, outputs=firststep_t1, queue=False
        )
        video_t1.change(
            fn=make_video1, inputs=video_t1, outputs=laststep_t1, queue=False
        )
        video_t1.change(
            fn=make_video2, inputs=video_t1, outputs=steps_t1, queue=False
        )
        video_t1.change(
            fn=make_video2, inputs=video_t1, outputs=iter_t1, queue=False
        )
        video_t1.change(
            fn=make_video2, inputs=video_t1, outputs=batch_t1, queue=False
        )

        video_t2.change(
            fn=make_video1, inputs=video_t2, outputs=fps_t2, queue=False
        )
        video_t2.change(
            fn=make_video1, inputs=video_t2, outputs=firststep_t2, queue=False
        )
        video_t2.change(
            fn=make_video1, inputs=video_t2, outputs=laststep_t2, queue=False
        )
        video_t2.change(
            fn=make_video2, inputs=video_t2, outputs=steps_t2, queue=False
        )
        video_t2.change(
            fn=make_video2, inputs=video_t2, outputs=iter_t2, queue=False
        )
        video_t2.change(
            fn=make_video2, inputs=video_t2, outputs=batch_t2, queue=False
        )

        # loopback
        loopback_t1.change(
            fn=make_loopback,
            inputs=loopback_t1,
            outputs=loopback_halving_t1,
            queue=False,
        )

        image_out.style(grid=2)
        image_t1.style(height=402)
        image_t2.style(height=402)

    # change the default temp folder and handle cleaning it when stopping the ui
    os.makedirs("temp", exist_ok=True)
    tempfile.tempdir = os.path.abspath(os.path.join("temp"))
    signal.signal(signal.SIGINT, clear_temp_files)

    # start gradio web interface on local host
    demo.launch()

    # use the following to launch the web interface to a private network
    # demo.queue(concurrency_count=1)
    # demo.launch(server_name="0.0.0.0")
