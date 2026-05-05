import io
import logging
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


async def download_and_compress_image(
    url: str, max_size_kb: int = 500, max_dimension: int = 1024
) -> bytes:
    """
    Downloads an image from a URL and compresses it to save tokens/bandwidth.

    Args:
        url: The URL of the image to download.
        max_size_kb: Target maximum size in KB.
        max_dimension: Maximum width or height.

    Returns:
        The compressed image bytes (JPEG format).
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            image_data = response.content

        # Open image using Pillow
        img: Any = Image.open(io.BytesIO(image_data))

        # Convert to RGB if it's RGBA or P (to save as JPEG)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Resize if too large
        if img.width > max_dimension or img.height > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        # Compress
        output = io.BytesIO()
        quality = 85
        img.save(output, format="JPEG", quality=quality)

        # Iteratively reduce quality if still too large
        while len(output.getvalue()) > max_size_kb * 1024 and quality > 30:
            quality -= 10
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=quality)

        return output.getvalue()

    except Exception as e:
        logger.warning(f"Failed to download or compress image {url}: {e}")
        # Return empty bytes if failed, the caller should handle it
        return b""
