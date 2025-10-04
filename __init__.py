from .openai_chat_completion import OpenAIChatCompletion  # Import your actual node class

NODE_CLASS_MAPPINGS = {
    "OpenAIChatCompletion": OpenAIChatCompletion  # The key is what will show up in ComfyUI interface
}

# Optional: If your node has display names different from class names
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIChatCompletion": "LLM Chat Completion"  # Optional, for better UI display
}

# Required: To properly register your node
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
