from __future__ import annotations
from PIL import Image
import numpy as np
import base64
import torch
from io import BytesIO
from server import PromptServer, BinaryEventTypes


class LoadImageBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "external_tooling"
    FUNCTION = "load_image"

    def load_image(self, image):
        imgdata = base64.b64decode(image)
        img = Image.open(BytesIO(imgdata))

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]

        return (img, mask)


class LoadMaskBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    CATEGORY = "external_tooling"
    FUNCTION = "load_mask"

    def load_mask(self, mask):
        imgdata = base64.b64decode(mask)
        img = Image.open(BytesIO(imgdata))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        if img.dim() == 3:  # RGB(A) input, use red channel
            img = img[:, :, 0]
        return (img.unsqueeze(0),)


class SendImageWebSocket:
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


NODE_CLASS_MAPPINGS = {
    "NTL_LoadImageBase64": LoadImageBase64,
    "NTL_LoadMaskBase64": LoadMaskBase64,
    "NTL_SendImageWebSocket": SendImageWebSocket,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NTL_LoadImageBase64": "ntl Load Image (Base64)",
    "NTL_LoadMaskBase64": "ntl Load Mask (Base64)",
    "NTL_SendImageWebSocket": "ntl Send Image (WebSocket)",
}
