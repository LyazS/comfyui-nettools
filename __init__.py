from __future__ import annotations
from PIL import Image
import numpy as np
import base64
import torch
from io import BytesIO
from server import PromptServer
import json
import random

class LoadImagesBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "external_tooling"
    FUNCTION = "load_images"

    def load_images(self, images):
        imgs = []
        base64_strings = json.loads(images)
        for base64_str in base64_strings:
            imgdata = base64.b64decode(base64_str)
            img = Image.open(BytesIO(imgdata))

            img = img.convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]

            imgs.append(img)
        pass
        stacked_imgs = torch.cat(imgs, dim=0)

        return (stacked_imgs,)


class SendImagesWebSocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "nodelabel": ("STRING", {"multiline": False}),
                "partlen": ("INT", {"default": 1 * 1024 * 1024}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_images"
    OUTPUT_NODE = True
    CATEGORY = "external_tooling"

    def send_images(self, images, nodelabel, partlen):
        results = []
        for img_idx, tensor in enumerate(images):
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            buffered = BytesIO()
            image.save(buffered, format="PNG")
            imageb64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            imageb64len = len(imageb64)
            print(f"Image {img_idx} has size {imageb64len} bytes")
            server = PromptServer.instance
            for parti, b64i in enumerate(range(0, len(imageb64), partlen)):
                server.send_sync(
                    "ntlb64part",
                    {
                        "nodelabel": nodelabel,
                        "img_idx": img_idx,
                        "part_idx": parti,
                        "part_b64": imageb64[b64i : b64i + partlen],
                    },
                    server.client_id,
                )
            results.append(
                # Could put some kind of ID here, but for now just match them by index
                {"source": "websocket", "content-type": "image/png", "type": "output"}
            )

        return {"ui": {"images": results}}

    @classmethod
    def IS_CHANGED(s, images, nodelabel, partlen):
        return random.randint()

NODE_CLASS_MAPPINGS = {
    "NTL_LoadImagesBase64": LoadImagesBase64,
    "NTL_SendImagesWebSocket": SendImagesWebSocket,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NTL_LoadImagesBase64": "ntl Load Images (Base64)",
    "NTL_SendImagesWebSocket": "ntl Send Images (WebSocket)",
}
