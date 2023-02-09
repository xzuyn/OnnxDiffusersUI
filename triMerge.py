import os
import torch

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def trimerge(model_a, model_b, model_c, model_output, withoutvae=True):
    a = {}
    b = {}
    c = {}

    a_path = f"./models/Stable-diffusion/{model_a}.safetensors"
    b_path = f"./models/Stable-diffusion/{model_b}.safetensors"
    c_path = f"./models/Stable-diffusion/{model_c}.safetensors"

    with safe_open(a_path, framework="pt", device="cpu") as fa:
        for ka in fa.keys():
            a[ka] = fa.get_tensor(ka)
    with safe_open(b_path, framework="pt", device="cpu") as fb:
        for kb in fb.keys():
            b[kb] = fb.get_tensor(kb)
    with safe_open(c_path, framework="pt", device="cpu") as fc:
        for kc in fc.keys():
            c[kc] = fc.get_tensor(kc)
    out_path = f"./models/Stable-diffusion/{model_output}.safetensors"
    if os.path.isfile(out_path):
        resp = input("Output file already exists. Overwrite? (y/n)").lower()
        if resp == "y":
            os.remove(out_path)
        else:
            return False
    for key in tqdm(a.keys(), desc="Stage 1/3"):
        if withoutvae and "first_stage_model" in key:
            continue
        if "model" in key and key in b and key in c:
            a[key] = b[key] * (
                abs(a[key] - b[key]) > abs(a[key] - c[key])
            ) + c[key] * (abs(a[key] - b[key]) <= abs(a[key] - c[key]))
    for key in tqdm(b.keys(), desc="Stage 2/3"):
        if "model" in key and key not in a:
            a[key] = b[key]
    for key in tqdm(c.keys(), desc="Stage 3/3"):
        if "model" in key and key not in a:
            a[key] = c[key]
    print("Saving...")
    save_file(a, out_path)
    return out_path


print("Safetensors Only. Do not include .safetensors in name.")
print("Inputs and Outputs go in ./models/Stable-diffusion/")
print()

input_model_a = input("Model A: ")
input_model_b = input("Model B: ")
input_model_c = input("Model C: ")
input_model_output = input("Output Model: ")
print()

trimerge(input_model_a, input_model_b, input_model_c, input_model_output)
