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
        self.vlm_reasoning_client = None  # VLM client for reasoning vision models
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

        # Stage 1.1: Legend summary with VLM
        legend_summaries = await self._summarise_legend(
            page_image, legend_boxes, page_number
        )

        # Stage 2: Legend item detection with YOLO (returns Box objects)
        # Disabled for now
        # legend_items: List[Box] = await self._detect_legend_items_with_yolo(
        #     page_image, legend_boxes, page_number
        # )

        # Stage 3: Item query (temp implementation - text description from VLM
        drawing_desc = await self._query_items(
            page_image, legend_boxes, legend_summaries, page_number
        )

        # Summarise entire page
        drawing_summary = await self._summarise_page(page_image, page_number)

        page_content = drawing_summary + "\n\n" + drawing_desc

        # Convert Box objects to dicts for metadata storage
        # legend_items_dicts = [box.to_dict() for box in legend_items]

        # Create document with vision processing results
        vision_metadata = {
            **metadata,
            "page_type": "drawing",
            "page_number": page_number,
            "legend_boxes": json.dumps(legend_boxes),
            # "legend_items": json.dumps(legend_items_dicts),
            "drawing_summary": drawing_summary,  # pass this on for possible context for chunking downstream
            "vision_processed": True,
        }

        return Document(
            page_content=page_content,
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

    async def _summarise_legend(
        self,
        page_image: Image.Image,
        legend_boxes: List[Tuple[int, int, int, int]],
        page_number: int,
    ) -> List[str]:
        """
        Stage 1.1: Summarise the legend using VLM.
        """
        try:
            legend_summary = []
            for legend_box in legend_boxes:
                legend_summary.append(
                    await self._summarise_legend_region(
                        page_image, legend_box, page_number
                    )
                )
            return legend_summary
        except Exception as e:
            # Log error and return empty list
            print(f"Error in legend summarisation for page {page_number}: {e}")
            import traceback

            traceback.print_exc()
            return []

    async def _summarise_legend_region(
        self,
        page_image: Image.Image,
        legend_box: Tuple[int, int, int, int],
        page_number: int,
    ) -> str:
        """
        Summarise a legend region using VLM.
        """
        if self.vlm_client is None:
            self.vlm_client = init_chat_model(
                settings.VLM_MODEL_NAME,
                model_provider="openai",
                base_url=settings.VLM_BASE_URL,
                api_key=settings.DASHSCOPE_API_KEY,
            )
        try:
            # Crop legend region from page
            x1, y1, x2, y2 = legend_box
            legend_region = page_image.crop((x1, y1, x2, y2))

            legend_bytes = self._image_to_bytes(legend_region)
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """This is a legend for a technical drawing. Please extract each item in the legend and its description into a ordered list. 
- Describe the symbol if not clear.
- Use the exact description defined in the legend.""",
                    },
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": base64.b64encode(legend_bytes).decode("utf-8"),
                        "mime_type": "image/png",
                    },
                ],
            }

            response = self.vlm_client.invoke([message])
            return response.content

        except Exception as e:
            # Log error and return default description
            print(f"Error in legend summarisation for page {page_number}: {e}")
            import traceback

            traceback.print_exc()
            return ""

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

            # Flatten rows back to list (only boxes from valid rows, orphaned boxes excluded)
            processed_boxes = [box for row in all_rows for box in row]

            # Rebuild legend_regions from processed_boxes (excludes orphaned boxes)
            # This ensures infer_macro_and_subcolumns only processes boxes with valid rows
            legend_regions = {}
            for box in processed_boxes:
                legend_id = box.legend_id if box.legend_id is not None else 0
                if legend_id not in legend_regions:
                    legend_regions[legend_id] = []
                legend_regions[legend_id].append(box)

            # Step 3: Infer macro and sub-columns (within each legend region)
            # Only boxes with row_num is not None (non-orphaned) are included
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

            # Step 4: Within each macro column, mark singleton rows as orphaned
            # If a row within a macro column contains only one bbox, set its row_num to None
            for legend_id, legend_boxes_list in legend_regions.items():
                # Build mapping: macro_col -> row_num -> list[Box]
                macro_to_rows: Dict[int, Dict[int, List[Box]]] = {}
                for b in legend_boxes_list:
                    if b.macro_col_num is None or b.row_num is None:
                        continue
                    if b.macro_col_num not in macro_to_rows:
                        macro_to_rows[b.macro_col_num] = {}
                    if b.row_num not in macro_to_rows[b.macro_col_num]:
                        macro_to_rows[b.macro_col_num][b.row_num] = []
                    macro_to_rows[b.macro_col_num][b.row_num].append(b)

                # Mark rows with only one box as orphaned
                for macro_col, rows_map in macro_to_rows.items():
                    for row_idx, boxes_in_row in rows_map.items():
                        if len(boxes_in_row) == 1:
                            boxes_in_row[0].row_num = None

            # Return boxes with metadata updated by Steps 3 and 4
            updated_boxes: List[Box] = []
            for legend_id, legend_boxes_list in legend_regions.items():
                updated_boxes.extend(legend_boxes_list)

            return updated_boxes

        except Exception as e:
            # Log error and return empty list
            print(f"Error in legend item detection for page {page_number}: {e}")
            import traceback

            traceback.print_exc()
            return []

    async def _query_items(
        self,
        page_image: Image.Image,
        legend_boxes: List[Tuple[int, int, int, int]],
        legend_summaries: List[str],
        page_number: int,
    ) -> str:
        """
        Stage 3: Query items using VLM to analyze symbols in the drawing based on legend regions.

        Args:
            page_image: PIL Image of the page
            legend_boxes: List of legend bounding boxes from Stage 1 as tuples (x1, y1, x2, y2)
            legend_summaries: List of legend summaries from Stage 1.1
            page_number: Page number for logging

        Returns:
            String description of the drawing with symbol analysis
        """
        if self.vlm_reasoning_client is None:
            self.vlm_reasoning_client = init_chat_model(
                "qwen3-vl-235b-a22b-thinking",
                model_provider="openai",
                base_url=settings.VLM_BASE_URL,
                api_key=settings.DASHSCOPE_API_KEY,
            )

        try:
            # Crop legend regions from page image
            legend_crops = []
            for bbox in legend_boxes:
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                legend_crop = page_image.crop((x1, y1, x2, y2))
                legend_crops.append(legend_crop)

            # Generate legend summary text
            legend_summary_str = ""
            for legend_idx, legend_summary in enumerate(legend_summaries):
                legend_summary_str += (
                    f"Legend summary {legend_idx + 1}: \n\n{legend_summary}\n"
                )

            # Build message content: text prompt + legend crops + full page image
            content = [
                {
                    "type": "text",
                    "text": """The following images are legend regions for a technical drawing, followed by the full drawing page. For each symbol in the legend regions, analyse the symbol's occurrence in the drawing and return the following: 
- symbol name 
- description (use exact description from legend)
- where in the drawing is it defined - include relevant details such as which room, area for each occurrence of the symbol
- total number of occurrences of the symbol in the drawing

Use the following list of legend items as a guide:
{legend_summary_str}""",
                }
            ]

            # Add all legend crop images first
            for legend_crop in legend_crops:
                legend_bytes = self._image_to_bytes(legend_crop)
                content.append(
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": base64.b64encode(legend_bytes).decode("utf-8"),
                        "mime_type": "image/png",
                    }
                )

            # Add full page image last
            page_bytes = self._image_to_bytes(page_image)
            content.append(
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": base64.b64encode(page_bytes).decode("utf-8"),
                    "mime_type": "image/png",
                }
            )

            # Create message and invoke VLM
            message = {
                "role": "user",
                "content": content,
            }

            response = self.vlm_reasoning_client.invoke([message])
            return response.content

        except Exception as e:
            # Log error and return default description
            print(f"Error in item query for page {page_number}: {e}")
            import traceback

            traceback.print_exc()
            return "Drawing description"

    async def _summarise_page(self, page_image: Image.Image, page_number: int) -> str:
        """
        Summarise the entire page using VLM.
        """
        if self.vlm_client is None:
            self.vlm_client = init_chat_model(
                settings.VLM_MODEL_NAME,
                model_provider="openai",
                base_url=settings.VLM_BASE_URL,
                api_key=settings.DASHSCOPE_API_KEY,
            )

        try:
            page_bytes = self._image_to_bytes(page_image)
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""The image provided is a technical drawing. Please describe and summarise the drawing. The summary will be used as part of an index to semantically search for this page. 

Include the following information and nothing else:
- Summary and description of the drawing
- Type/Title of drawing (eg. Electrical Plan, First Floor Plan)
- Area of building covered by the drawing (eg. First floor, Ground level)""",
                    },
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": base64.b64encode(page_bytes).decode("utf-8"),
                        "mime_type": "image/png",
                    },
                ],
            }

            response = self.vlm_client.invoke([message])
            return response.content

        except Exception as e:
            # Log error and return default description
            print(f"Error in page summarisation for page {page_number}: {e}")
            import traceback

            traceback.print_exc()
            return ""

    def _image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes"""
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=format)
        return img_bytes.getvalue()

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
