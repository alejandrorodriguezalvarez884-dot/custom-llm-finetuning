"""
Convert images extracted from PDFs to text descriptions using the Claude API.
If ANTHROPIC_API_KEY is not set, describe_image() returns an empty string
and the caller should skip that image silently.
"""
import os
import base64
from dotenv import load_dotenv

load_dotenv()

_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

_client = None


def _get_client():
    global _client
    if _client is None and _ANTHROPIC_API_KEY:
        import anthropic
        _client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)
    return _client


def describe_image(image_bytes: bytes, image_ext: str) -> str:
    """
    Return a detailed text description of an image using Claude.
    Returns an empty string if the API key is not configured or the call fails.
    """
    client = _get_client()
    if client is None:
        return ""

    _MEDIA_TYPES = {
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    media_type = _MEDIA_TYPES.get(image_ext.lower(), "image/png")
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        response = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in detail. "
                                "If it contains text, transcribe it fully. "
                                "If it is a diagram, chart, or table, explain its content and structure precisely. "
                                "Be thorough and factual. Do not add information not present in the image."
                            ),
                        },
                    ],
                }
            ],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        print(f"    [image_processor] Warning: could not process image — {exc}")
        return ""


def is_available() -> bool:
    """Return True if the Anthropic API is configured."""
    return bool(_ANTHROPIC_API_KEY)
