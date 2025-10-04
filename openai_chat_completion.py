import numpy as np
from PIL import Image
import io
import base64
from typing import Literal
from openai import OpenAI
import re

class OpenAIChatCompletion:
    """ComfyUI node for LLM chat completion"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are an expert at expanding and refining text-to-image AI prompts."}),
                "pre_prompt": ("STRING", {"multiline": True, "default": "Please enhance the following image prompt:", "placeholder": "Text to prepend to the user prompt."}),
                "text_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "endpoint": ("STRING", {"multiline": False, "default": "http://localhost:8080/v1"}),
                "api_key": ("STRING", {"multiline": False, "default": "123456"}),
                "model": ("STRING", {"multiline": False, "default": "prompt-enhancer-32b"}),
                "use_temperature": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "use_seed": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "use_max_tokens": ("BOOLEAN", {"default": True}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 1048576}),
                "strip_reasoning": ("BOOLEAN", {"default": True}),
                "reasoning_tag_open": ("STRING", {"multiline": False, "default": "<think>"}),
                "reasoning_tag_close": ("STRING", {"multiline": False, "default": "</think>"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("Text","Reasoning",)
    FUNCTION = "process"
    CATEGORY = "LLM"

    def process(
        self,
        system_prompt: str,
        pre_prompt: str,
        text_prompt: str,
        model: str,
        endpoint: str,
        api_key: str,
        use_temperature: bool,
        temperature: float,
        use_max_tokens: bool,
        max_tokens: int,
        strip_reasoning: bool,
        reasoning_tag_open: str,
        reasoning_tag_close: str,
        use_seed: bool,
        seed: int,
        image: np.ndarray = None,
    ):
        """Process the text prompt with optional image"""
        
        # Prepare messages
        messages = []

        # Add system prompt, if necessary
        if system_prompt is not None and system_prompt != '':
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ]
                }
            )
            
        # Add the pre-prompt to the user prompt, if necessary
        if pre_prompt is not None and pre_prompt != '':
            user_prompt = f"{pre_prompt}\n{text_prompt}"
        else:
            user_prompt = text_prompt

        # If an image was passed, add it to the chat along with the text. Otherwise, only add the text.
        if image is not None:
            # Convert from numpy RGBA to PIL Image
            pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            # Convert to RGB if necessary
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            # Convert to base64
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                    ]
                }
            )
        
        # Connect to OpenAI-compatible endpoint and get chat completion
        client_kwargs = {'base_url': endpoint}
        if api_key is not None and api_key != '':
            client_kwargs['api_key'] = api_key
        client = OpenAI(**client_kwargs)

        completion_kwargs = {'model': model, 'messages': messages}
        if use_seed:
            completion_kwargs['seed'] = seed
        if use_max_tokens:
            completion_kwargs['max_tokens'] = max_tokens
        if use_temperature:
            completion_kwargs['temperature'] = temperature

        completion = client.chat.completions.create(**completion_kwargs)
        result = completion.choices[0].message.content

        # Remove the reasoning content, if necessary
        if strip_reasoning:
            start_delimiter = re.escape(reasoning_tag_open)
            end_delimiter = re.escape(reasoning_tag_close)

            find_pattern = f"{start_delimiter}(.*?){end_delimiter}"
            reasoning = "\n".join(re.findall(find_pattern, result, flags=re.DOTALL)).strip()

            pattern = f"{start_delimiter}.*?{end_delimiter}"
            result = re.sub(pattern, "", result, flags=re.DOTALL).strip()
        else:
            reasoning = ""
            result = result.strip()

        return (result, reasoning,)

