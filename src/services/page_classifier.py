"""Page type classifier for PDF pages using Qwen vision model"""

from typing import Literal
from PIL import Image
import base64
import io
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from src.core.config import settings


PageType = Literal["text", "drawing"]


class PageClassifier:
    """Classify PDF pages as text-based or drawing/legend-based using Qwen vision model"""

    def __init__(self):
        """Initialize the page classifier with Qwen LLM"""
        self._init_llm()

    def classify_page(self, page_image: Image.Image = None) -> PageType:
        """
        Classify a page as 'text' or 'drawing' based on visual analysis.

        Args:
            page_image: PIL Image of the page for visual analysis

        Returns:
            'text' or 'drawing'
        """
        if page_image is None:
            return "text"  # Default if no image provided

        return self._classify_with_llm(page_image)

    def _init_llm(self):
        """Initialize the Qwen LLM for page classification"""
        try:
            self.llm = init_chat_model(
                "qwen3-vl-8b-instruct",
                model_provider="openai",
                base_url=settings.VLM_BASE_URL,
                api_key=settings.DASHSCOPE_API_KEY,
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Qwen LLM: {e}")
            self.llm = None

    def _classify_with_llm(self, page_image: Image.Image) -> PageType:
        """
        Classify page using Qwen LLM vision model.

        Args:
            page_image: PIL Image of the page

        Returns:
            'text' or 'drawing'
        """
        if self.llm is None:
            return "text"  # Default fallback

        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(page_image)

            # Create message for LLM
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Classify the page if it is a full-page drawing? Return true if it is, or false otherwise.

Drawing
- If majority of the page is an architectural plan, engineering schematic, MEP layout, construction drawing, etc
- Contains plan views or schematic layouts — orthogonal top-down or sectional representations of buildings, rooms, or mechanical systems.
- Scale indicators or dimensions
- Return true

Non-drawing
- Contains mostly text 
- Table of materials, schedules, etc.
- Table of schedules are considered non-drawings, even if they contain smaller representations
- Return false

If the page contains mostly text with inline drawings, return false""",
                    },
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": image_base64,
                        "mime_type": "image/png",
                    },
                ],
            }

            # Get LLM response
            response = self.llm.invoke([message])

            # Extract result
            is_drawing = self._extract_drawing_result(response)

            return "drawing" if is_drawing else "text"

        except Exception as e:
            print(f"Warning: LLM classification failed: {e}")
            return "text"  # Default fallback

    def _extract_drawing_result(self, response) -> bool:
        """
        Extract drawing classification result from LLM text response

        Args:
            response: LLM response object with content attribute

        Returns:
            True if classified as drawing, False otherwise
        """
        # Primary: check for content attribute with string parsing
        if hasattr(response, "content") and response.content:
            content_str = str(response.content).strip().lower()
            # Handle various string representations of boolean
            if content_str in ["true", "1", "yes", "drawing"]:
                return True
            if content_str in ["false", "0", "no", "text"]:
                return False

        # Fallback: check raw response structure
        if hasattr(response, "raw"):
            try:
                raw_str = str(response.raw).lower()
                if "true" in raw_str or "drawing" in raw_str:
                    return True
                if "false" in raw_str or "text" in raw_str:
                    return False
            except Exception:
                pass

        # Default: not a drawing
        return False

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 PNG string.
        Resizes image long side to 1024 pixels to optimize for LLM processing.

        Args:
            image: PIL Image

        Returns:
            Base64 encoded PNG string
        """
        # Resize image if larger than max_size
        image = self._resize_image_if_needed(image, max_size=1024)

        # Convert PIL Image to PNG bytes
        png_buffer = io.BytesIO()
        image.save(png_buffer, format="PNG")
        png_bytes = png_buffer.getvalue()

        # Convert PNG bytes to base64
        base64_string = base64.b64encode(png_bytes).decode("utf-8")

        return base64_string

    def _resize_image_if_needed(
        self, image: Image.Image, max_size: int = 1024
    ) -> Image.Image:
        """
        Resize image if its longest side exceeds max_size, maintaining aspect ratio.

        Args:
            image: PIL Image to resize
            max_size: Maximum size for the longest side

        Returns:
            Resized PIL Image (or original if no resize needed)
        """
        width, height = image.size
        longest_side = max(width, height)

        if longest_side <= max_size:
            return image

        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        # Resize using high-quality resampling
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# Global classifier instance
page_classifier = PageClassifier()
