"""TEMP IMPLEMENTATION: Page type classifier for PDF pages"""

from typing import Dict, Any, Literal
from PIL import Image
import numpy as np
from langchain_core.documents import Document


PageType = Literal["text", "drawing"]


class PageClassifier:
    """Classify PDF pages as text-based or drawing/legend-based"""

    def __init__(self):
        # Thresholds for classification
        self.text_density_threshold = 0.05  # Minimum text density ratio
        self.drawing_complexity_threshold = 0.3  # Minimum drawing complexity

    def classify_page(
        self, page_document: Document, page_image: Image.Image = None
    ) -> PageType:
        """
        Classify a page as 'text' or 'drawing' based on content analysis.

        Args:
            page_document: LangChain Document with page content
            page_image: Optional PIL Image of the page for visual analysis

        Returns:
            'text' or 'drawing'
        """
        # Primary classification: based on text content
        text_content = page_document.page_content.strip()

        # If minimal text, likely a drawing page
        if len(text_content) < 100:
            return "drawing"

        # Analyze text density and structure
        text_metrics = self._analyze_text_content(text_content)

        # If image provided, also analyze visual characteristics
        if page_image is not None:
            visual_metrics = self._analyze_visual_content(page_image)
            combined_score = self._combine_metrics(text_metrics, visual_metrics)
        else:
            combined_score = text_metrics.get("drawing_score", 0.0)

        # Classify based on combined score
        # Higher score indicates drawing page
        if combined_score > 0.5:
            return "drawing"
        else:
            return "text"

    def _analyze_text_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze text content characteristics.

        Returns:
            Dictionary with analysis metrics
        """
        # Text length
        text_length = len(text)

        # Line count (indicating structured text vs sparse annotations)
        lines = text.split("\n")
        line_count = len(lines)
        avg_line_length = text_length / max(line_count, 1)

        # Check for common drawing/legend indicators
        drawing_keywords = [
            "legend",
            "symbol",
            "scale",
            "drawing",
            "plan",
            "section",
            "detail",
            "note:",
            "see note",
        ]
        drawing_keyword_count = sum(
            1 for keyword in drawing_keywords if keyword.lower() in text.lower()
        )

        # Calculate drawing score
        # Lower text density, shorter lines, and drawing keywords suggest drawing page
        text_density_score = min(text_length / 2000.0, 1.0)  # Normalize to 0-1
        drawing_score = (
            (1 - text_density_score) * 0.4  # Inverse text density
            + (drawing_keyword_count / len(drawing_keywords)) * 0.3  # Keyword presence
            + (1 - min(avg_line_length / 100.0, 1.0)) * 0.3  # Short lines
        )

        return {
            "text_length": text_length,
            "line_count": line_count,
            "avg_line_length": avg_line_length,
            "drawing_keyword_count": drawing_keyword_count,
            "drawing_score": drawing_score,
        }

    def _analyze_visual_content(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze visual characteristics of the page image.

        Args:
            image: PIL Image of the page

        Returns:
            Dictionary with visual analysis metrics
        """
        # Convert to grayscale for analysis
        gray = image.convert("L")
        img_array = np.array(gray)

        # Calculate edge density (drawings typically have many edges)
        # Simple edge detection using gradient
        edges = np.gradient(img_array.astype(float))
        edge_magnitude = np.sqrt(edges[0] ** 2 + edges[1] ** 2)
        edge_density = np.mean(edge_magnitude > 30) / 255.0  # Threshold and normalize

        # Calculate complexity (variance in pixel values)
        pixel_variance = np.var(img_array) / (255.0**2)  # Normalize

        # Calculate drawing score based on visual characteristics
        # Higher edge density and complexity suggest drawing
        drawing_score = edge_density * 0.5 + pixel_variance * 0.5

        return {
            "edge_density": edge_density,
            "pixel_variance": pixel_variance,
            "drawing_score": drawing_score,
        }

    def _combine_metrics(
        self, text_metrics: Dict[str, Any], visual_metrics: Dict[str, Any]
    ) -> float:
        """
        Combine text and visual metrics into final classification score.

        Args:
            text_metrics: Metrics from text analysis
            visual_metrics: Metrics from visual analysis

        Returns:
            Combined drawing score (0-1, higher = more likely drawing)
        """
        text_score = text_metrics.get("drawing_score", 0.0)
        visual_score = visual_metrics.get("drawing_score", 0.0)

        # Weighted combination (can be adjusted)
        combined = text_score * 0.6 + visual_score * 0.4

        return combined


# Global classifier instance
page_classifier = PageClassifier()
