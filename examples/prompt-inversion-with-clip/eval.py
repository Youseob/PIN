import open_clip
# from optim_utils import * 

import torch
import mediapy as media
import argparse
import json
from PIL import Image
from io import BytesIO
import requests


def read_json(filename: str):
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)

def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


args = argparse.Namespace()
args.__dict__.update(read_json("sample_config.json"))
print(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2-1-base"
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    revision="fp16",
    )

pipe = pipe.to(device)
prompt = ' Lloyd Stephenson Keane Girls Brighton Pony Carroll Gallery' #detailsgirls portraying  #Girls portraying riding holiday paintings Brighton #Girls holiday painting seas ponies pedals
num_images = 4
guidance_scale = 9
num_inference_steps = 25
image_length = 512

images = pipe(
    prompt,
    num_images_per_prompt=num_images,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    height=image_length,
    width=image_length,
    ).images

print(f"prompt: {prompt}")
# media.show_images(images)
media.write_image('test0.png', images[0])
media.write_image('test1.png', images[1])
media.write_image('test2.png', images[2])
media.write_image('test3.png', images[3])


# urls = [
#         "https://a.1stdibscdn.com/alexander-averin-paintings-pony-riding-on-the-beach-post-impressionist-style-oil-painting-for-sale-picture-6/a_7443/a_28523631526593507117/Promenade_detalle_5_master.JPG?disable=upscale&auto=webp&quality=60&width=1318",
#        ]

# orig_images = list(filter(None,[download_image(url) for url in urls]))
# target_images = orig_images
# tokenizer_funct = open_clip.get_tokenizer(args.clip_model)
# with torch.no_grad():
#     curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
#     curr_images = torch.concatenate(curr_images).to(device)
#     image_features = model.encode_image(curr_images)

#     texts = tokenizer_funct(["Hi", "Hello"]).to(device)
#     text_features = model.encode_text(texts)
#     image_features = image_features / image_features.norm(dim=1, keepdim=True)
#     text_features = text_features / text_features.norm(dim=1, keepdim=True)
#     pdb.set_trace()
#     # (image_bs, emb_dim) @ (emb_dim, samples_text)
#     logits_per_image = image_features @ text_features.t() # (1, samples_texts)
#     logits_per_text = logits_per_image.t() # (sample_texts, 1)

