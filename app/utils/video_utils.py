import logging
import os
import tempfile
from typing import List

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


def extract_keyframes(video_bytes: bytes, num_frames: int = 5) -> List[Image.Image]:
    """
    Extracts a specified number of keyframes from video bytes.

    Args:
        video_bytes: The raw bytes of the video file.
        num_frames: Number of frames to extract (evenly spaced).

    Returns:
        A list of PIL Image objects.
    """
    frames = []
    temp_video = None

    try:
        # Save video bytes to a temporary file because OpenCV needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            temp_video = tmp.name

        cap = cv2.VideoCapture(temp_video)
        if not cap.isOpened():
            logger.error("Could not open video bytes with OpenCV")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.error("Video has no frames")
            return []

        # Calculate frame indices to extract
        indices = [
            int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)
        ]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                # Resize to 1024px max for consistency
                if img.width > 1024 or img.height > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

                frames.append(img)

        cap.release()

    except Exception as e:
        logger.error(f"Error extracting keyframes: {e}")
    finally:
        # Clean up temporary file
        if temp_video and os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except Exception:
                pass

    return frames
