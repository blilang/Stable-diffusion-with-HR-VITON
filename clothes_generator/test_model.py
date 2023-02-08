from diffusers import StableDiffusionPipeline
import torch
from diffusers import DDIMScheduler
from PIL import Image

model_path = "./sd_notextfinetune_clothes_model-Step-12000"  
prompt = "dress, rabbit, casual"
#tshirt, rabbit, [V]
#leaves, tshirt, casual, [V]
#Nike, tshirt, [V]
#painting, tshirt, [V]

pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        scheduler=DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=True,
        ),
        safety_checker=None,
    )

# def dummy(images, **kwargs):
#     return images, False
# pipe.safety_checker = dummy
pipe = pipe.to("cuda")
images = pipe(prompt, num_inference_steps=30, num_images_per_prompt=10,height=512,width=512).images

box=(128,0,896,1024)
for i, image in enumerate(images):
    image.save(f"test512-{i}.jpg")
    image = image.resize((1024,1024))
    temp = image.crop(box)
    new_image = Image.new('RGB', (768,1024),(0,0,0))
    new_image.paste(temp, (0,0))
    new_image.save(f"test-{i}.jpg")