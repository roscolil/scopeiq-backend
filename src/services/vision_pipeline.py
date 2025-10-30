"""Vision pipeline service for processing PDF pages with drawings and legends"""

import io
import os
import json
import base64
import ast
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from ultralytics import YOLO
from src.core.config import settings
from src.services.legend_postprocessing import (
    Box,
    apply_nms,
    cluster_rows,
    infer_macro_and_subcolumns,
)


class VisionPipelineService:
    """Service for processing pages with drawings and legends through multi-stage vision pipeline"""

    def __init__(self):
        self.vlm_client = None  # VLM client for hosted vision models
        self.yolo_model = None  # Fine-tuned YOLO model for local inference

    async def process_drawing_page(
        self,
        page_image: Image.Image,
        page_number: int,
        document_id: str,
        metadata: Dict[str, Any],
    ) -> Document:
        """
        Process a single page with drawings/legends through the 3-stage pipeline:
        1. Legend detection with VLM
        2. Legend item detection with YOLO
        3. Item query (placeholder)

        Args:
            page_image: PIL Image of the page
            page_number: Page number in the document
            document_id: Document ID
            metadata: Base metadata for the page

        Returns:
            Document with processed vision data
        """
        # Stage 1: Legend detection with VLM
        legend_boxes = await self._detect_legends_with_vlm(page_image, page_number)

        # Stage 2: Legend item detection with YOLO (returns Box objects)
        legend_items: List[Box] = await self._detect_legend_items_with_yolo(
            page_image, legend_boxes, page_number
        )

        # Stage 3: Item query (placeholder - not implemented yet)
        item_queries = await self._query_items(legend_items, page_number)

        # Convert Box objects to dicts for metadata storage
        legend_items_dicts = [box.to_dict() for box in legend_items]

        # Create document with vision processing results
        vision_metadata = {
            **metadata,
            "page_type": "drawing",
            "page_number": page_number,
            "legend_boxes": legend_boxes,
            "legend_items": legend_items_dicts,  # Store as dicts for JSON serialization
            "item_queries": item_queries,
            "vision_processed": True,
        }

        # Create text content from vision results
        text_content = self._create_text_from_vision_results(
            legend_boxes, legend_items, item_queries
        )

        return Document(
            page_content=text_content,
            metadata=vision_metadata,
        )

    async def _detect_legends_with_vlm(
        self, page_image: Image.Image, page_number: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Stage 1: Detect legend regions using Vision Language Model (VLM)
        via hosted API.

        Args:
            page_image: PIL Image of the page
            page_number: Page number for logging

        Returns:
            List of legend bounding boxes as tuples (x1, y1, x2, y2)
        """

        if self.vlm_client is None:
            self.vlm_client = init_chat_model(
                settings.VLM_MODEL_NAME,
                model_provider="openai",
                base_url=settings.VLM_BASE_URL,
                api_key=settings.DASHSCOPE_API_KEY,
            )

        try:
            img_bytes = self._image_to_bytes(page_image)

            vlm_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Perform detection of legend boxes in the document and provide coordinates of bounding box, if it exists. A legend box usually contains a list of symbols and their meanings. There may be multiple legend boxes in one drawing. Report bbox coordinates in JSON format.",
                    },
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": base64.b64encode(img_bytes).decode("utf-8"),
                        "mime_type": "image/png",
                    },
                ],
            }

            response = self.vlm_client.invoke([vlm_message])

            # Extract bboxes from content
            bounding_boxes = self._parse_json(response.content)
            try:
                json_output = ast.literal_eval(bounding_boxes)["legend_boxes"]
            except Exception as e:
                end_idx = bounding_boxes.rfind('"}') + len('"}')
                truncated_text = bounding_boxes[:end_idx] + "]"
                json_output = ast.literal_eval(truncated_text)

            if not isinstance(json_output, list):
                json_output = [json_output]

            # Crop legend boxes from page image
            width, height = page_image.size
            cropped_images = []
            abs_bounding_boxes = []
            for i, bounding_box in enumerate(json_output):
                # Convert normalized coordinates to absolute coordinates
                abs_y1 = int(bounding_box["bbox_2d"][1] / 1000 * height)
                abs_x1 = int(bounding_box["bbox_2d"][0] / 1000 * width)
                abs_y2 = int(bounding_box["bbox_2d"][3] / 1000 * height)
                abs_x2 = int(bounding_box["bbox_2d"][2] / 1000 * width)

                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1

                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1
                abs_bounding_boxes.append((abs_x1, abs_y1, abs_x2, abs_y2))

                # Crop the bounding box region
                cropped_image = page_image.crop((abs_x1, abs_y1, abs_x2, abs_y2))
                cropped_images.append(cropped_image)

            return abs_bounding_boxes
        except Exception as e:
            # Log error and return empty list
            print(f"Error in legend detection for page {page_number}: {e}")
            return []

    async def _detect_legend_items_with_yolo(
        self,
        page_image: Image.Image,
        legend_boxes: List[Tuple[int, int, int, int]],
        page_number: int,
    ) -> List[Box]:
        """
        Stage 2: Detect legend items within legend regions using YOLO model.
        Applies post-processing: NMS, row grouping, and column detection.

        Args:
            page_image: PIL Image of the page
            legend_boxes: List of legend bounding boxes from Stage 1 as tuples (x1, y1, x2, y2)
            page_number: Page number for logging

        Returns:
            List of Box objects with post-processing metadata (row_num, macro_col_num, sub_col_num)
        """
        if self.yolo_model is None:
            self.yolo_model = YOLO(settings.YOLO_MODEL_PATH)

        try:
            all_boxes: List[Box] = []

            # Run YOLO on each legend region
            for legend_idx, bbox in enumerate(legend_boxes):
                if len(bbox) != 4:
                    continue

                # Crop legend region from page
                x1, y1, x2, y2 = bbox
                legend_region = page_image.crop((x1, y1, x2, y2))
                legend_width = x2 - x1
                legend_height = y2 - y1
                legend_array = np.array(legend_region)

                # Run YOLO inference on legend region
                results = self.yolo_model(legend_array)

                # Convert YOLO results to Box objects and adjust coordinates to page-level
                for result in results:
                    # YOLO result format: result.boxes contains detections
                    if hasattr(result, "boxes") and result.boxes is not None:
                        boxes_data = result.boxes
                        for i in range(len(boxes_data)):
                            # Extract box coordinates (relative to legend region)
                            box_xyxy = (
                                boxes_data.xyxy[i].cpu().numpy()
                            )  # [x1, y1, x2, y2]
                            conf = (
                                float(boxes_data.conf[i].cpu().numpy())
                                if boxes_data.conf is not None
                                else None
                            )
                            cls = (
                                int(boxes_data.cls[i].cpu().numpy())
                                if boxes_data.cls is not None
                                else None
                            )
                            cls_name = (
                                result.names[cls]
                                if cls is not None and hasattr(result, "names")
                                else None
                            )

                            # Adjust coordinates to page-level
                            page_x1 = float(box_xyxy[0]) + x1
                            page_y1 = float(box_xyxy[1]) + y1
                            page_x2 = float(box_xyxy[2]) + x1
                            page_y2 = float(box_xyxy[3]) + y1

                            box = Box(
                                x1=page_x1,
                                y1=page_y1,
                                x2=page_x2,
                                y2=page_y2,
                                confidence=conf,
                                class_id=cls,
                                class_name=cls_name,
                                legend_id=legend_idx,
                            )
                            all_boxes.append(box)

            if not all_boxes:
                return []

            # Step 1: Apply NMS
            filtered_boxes = apply_nms(
                all_boxes,
                iou_threshold=0.4,  # Configurable via settings if needed
                score_threshold=0.3,  # Configurable via settings if needed
                class_agnostic=False,
            )

            if not filtered_boxes:
                return []

            # Step 2: Group into rows (within each legend region)
            # Process each legend region separately
            legend_regions: Dict[int, List[Box]] = {}
            for box in filtered_boxes:
                legend_id = box.legend_id if box.legend_id is not None else 0
                if legend_id not in legend_regions:
                    legend_regions[legend_id] = []
                legend_regions[legend_id].append(box)

            all_rows: List[List[Box]] = []
            for legend_id, legend_boxes_list in legend_regions.items():
                rows = cluster_rows(legend_boxes_list, tau=0.45)
                all_rows.extend(rows)

            # Flatten rows back to list
            processed_boxes = [box for row in all_rows for box in row]

            # Step 3: Infer macro and sub-columns (within each legend region)
            for legend_id, legend_boxes_list in legend_regions.items():
                # Get the legend region bounding box for roi_width
                if legend_id < len(legend_boxes):
                    legend_bbox = legend_boxes[legend_id]
                    legend_width_pixels = legend_bbox[2] - legend_bbox[0]
                else:
                    # Fallback: calculate from boxes
                    if legend_boxes_list:
                        x_coords = [b.x1 for b in legend_boxes_list] + [
                            b.x2 for b in legend_boxes_list
                        ]
                        legend_width_pixels = (
                            max(x_coords) - min(x_coords) if x_coords else None
                        )
                    else:
                        legend_width_pixels = None

                # Infer columns
                infer_macro_and_subcolumns(
                    legend_boxes_list,
                    is_text_mask=None,  # Could be enhanced with class-based filtering
                    k_sub_max=8,
                    roi_width=legend_width_pixels,
                    min_support_per_sub=2,
                    min_center_gap_frac=0.05,
                )

            return processed_boxes

        except Exception as e:
            # Log error and return empty list
            print(f"Error in legend item detection for page {page_number}: {e}")
            import traceback

            traceback.print_exc()
            return []

    async def _query_items(
        self, legend_items: List[Box], page_number: int
    ) -> List[Dict[str, Any]]:
        """
        Stage 3: Query items (placeholder - not implemented yet).

        Args:
            legend_items: List of Box objects from Stage 2 with post-processing metadata
            page_number: Page number for logging

        Returns:
            List of item queries/results
            Format: [{"item_id": int, "query": str, "result": Any}, ...]
        """
        # Placeholder function - not implemented yet

        return []

    def _image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes"""
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=format)
        return img_bytes.getvalue()

    def _create_text_from_vision_results(
        self,
        legend_boxes: List[Tuple[int, int, int, int]],
        legend_items: List[Box],
        item_queries: List[Dict[str, Any]],
    ) -> str:
        """
        Create text content from vision processing results for embedding/search.

        Args:
            legend_boxes: Legend boxes from Stage 1
            legend_items: Legend items from Stage 2
            item_queries: Item queries from Stage 3

        Returns:
            Text representation of the vision processing results
        """
        text_parts = []

        if legend_boxes:
            text_parts.append("Legends detected:")
            for idx, box in enumerate(legend_boxes):
                text_parts.append(f"- Legend {idx + 1}")

        if legend_items:
            text_parts.append("\nLegend items detected:")
            for item in legend_items:
                label = (
                    item.class_name or f"Class_{item.class_id}"
                    if item.class_id is not None
                    else "Unknown"
                )
                confidence = item.confidence if item.confidence is not None else 0.0
                row_info = f", row={item.row_num}" if item.row_num is not None else ""
                col_info = (
                    f", macro_col={item.macro_col_num}, sub_col={item.sub_col_num}"
                    if item.macro_col_num is not None
                    else ""
                )
                text_parts.append(
                    f"- {label} (confidence: {confidence:.2f}{row_info}{col_info})"
                )

        if item_queries:
            text_parts.append("\nItem queries:")
            for query in item_queries:
                query_text = query.get("query", "")
                if query_text:
                    text_parts.append(f"- {query_text}")

        return (
            "\n".join(text_parts)
            if text_parts
            else "Drawing page with legend processing completed."
        )

    def _parse_json(self, json_output: Any) -> str:
        # Normalize to string first
        try:
            if isinstance(json_output, bytes):
                json_output = json_output.decode("utf-8", errors="ignore")
            elif not isinstance(json_output, str):
                json_output = str(json_output)
        except Exception:
            json_output = str(json_output)

        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(
                    lines[i + 1 :]
                )  # Remove everything before "```json"
                json_output = json_output.split("```")[
                    0
                ]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        return json_output


# Global service instance
vision_pipeline_service = VisionPipelineService()
