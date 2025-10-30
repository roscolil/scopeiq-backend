"""Post-processing utilities for legend item detection results"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterable
import numpy as np

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class Box:
    """Bounding box with metadata for legend items"""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    row_num: Optional[int] = None
    macro_col_num: Optional[int] = None
    sub_col_num: Optional[int] = None
    legend_id: Optional[int] = None
    meta: Optional[Dict] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def to_dict(self) -> Dict:
        """Convert Box to dictionary for serialization"""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "row_num": self.row_num,
            "macro_col_num": self.macro_col_num,
            "sub_col_num": self.sub_col_num,
            "legend_id": self.legend_id,
            "meta": self.meta,
        }


def calculate_iou(box1: Box, box2: Box) -> float:
    """Calculate Intersection over Union (IoU) between two boxes."""
    # Calculate intersection coordinates
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    # Check if there's an intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Calculate intersection area
    intersection = (x2 - x1) * (y2 - y1)

    # Calculate union area
    union = box1.area + box2.area - intersection

    return intersection / union if union > 0 else 0.0


def apply_nms(
    boxes: List[Box],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.2,
    class_agnostic: bool = False,
) -> List[Box]:
    """
    Apply Non-Maximum Suppression to boxes.

    Args:
        boxes: List of Box objects
        iou_threshold: IoU threshold for overlap detection
        score_threshold: Minimum confidence to keep
        class_agnostic: If True, apply NMS across all classes; if False, apply per class

    Returns:
        List of filtered Box objects
    """
    if not boxes:
        return []

    # Filter by confidence threshold
    valid_boxes = [
        box
        for box in boxes
        if box.confidence is not None and box.confidence >= score_threshold
    ]

    if not valid_boxes:
        return []

    if class_agnostic:
        # Apply NMS across all classes
        return _apply_nms_single_class(valid_boxes, iou_threshold)
    else:
        # Apply NMS per class
        filtered_boxes = []
        class_groups: Dict[Optional[int], List[Box]] = {}

        # Group boxes by class
        for box in valid_boxes:
            class_id = box.class_id
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(box)

        # Apply NMS to each class
        for class_id, class_boxes in class_groups.items():
            filtered_class = _apply_nms_single_class(class_boxes, iou_threshold)
            filtered_boxes.extend(filtered_class)

        return filtered_boxes


def _apply_nms_single_class(boxes: List[Box], iou_threshold: float) -> List[Box]:
    """Apply NMS to boxes of a single class."""
    if len(boxes) <= 1:
        return boxes

    # Sort by confidence (descending)
    sorted_boxes = sorted(
        boxes,
        key=lambda b: b.confidence if b.confidence is not None else 0.0,
        reverse=True,
    )

    keep = []
    while sorted_boxes:
        # Take the box with highest confidence
        current = sorted_boxes.pop(0)
        keep.append(current)

        # Remove boxes with high IoU overlap
        remaining = []
        for box in sorted_boxes:
            iou = calculate_iou(current, box)
            if iou < iou_threshold:
                remaining.append(box)

        sorted_boxes = remaining

    return keep


def _y_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Pixel overlap along y between intervals a and b: (y1, y2)."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def cluster_rows(
    boxes: List[Box],
    tau: float = 0.45,  # overlap threshold as a fraction of median height
    sort_within_row: bool = True,  # left->right ordering inside each row
) -> List[List[Box]]:
    """
    Cluster boxes into rows based on vertical overlap.

    Args:
        boxes: List of Box objects
        tau: Overlap threshold as a fraction of median height
        sort_within_row: If True, sort boxes within each row left->right

    Returns:
        List of rows, where each row is a list of Box objects
    """
    if not boxes:
        return []

    # Calculate median height
    heights = [box.height for box in boxes]
    median_height = float(np.median(heights)) if heights else 1.0
    overlap_threshold = tau * median_height

    # Sort boxes by y1
    sorted_boxes = sorted(boxes, key=lambda b: b.y1)

    rows: List[List[Box]] = []
    for box in sorted_boxes:
        # Find which row this box belongs to
        box_y_interval = (box.y1, box.y2)
        placed = False

        for row in rows:
            # Check overlap with first box in row (representative)
            row_representative = row[0]
            row_y_interval = (row_representative.y1, row_representative.y2)

            overlap = _y_overlap(box_y_interval, row_y_interval)
            if overlap >= overlap_threshold:
                row.append(box)
                placed = True
                break

        if not placed:
            # Create new row
            rows.append([box])

    # Sort boxes within each row (left->right)
    if sort_within_row:
        for row in rows:
            row.sort(key=lambda b: b.x1)

    # Assign row_num to boxes
    for row_idx, row in enumerate(rows):
        for box in row:
            box.row_num = row_idx

    return rows


def _xcenters(boxes: List[Box]) -> np.ndarray:
    """Extract x-centers from boxes as numpy array."""
    return np.array([box.center_x for box in boxes], dtype=float).reshape(-1, 1)


def infer_macro_and_subcolumns(
    boxes: List[Box],
    is_text_mask: Optional[List[bool]] = None,
    k_sub_max: int = 8,  # maximum number of sub-columns to try
    roi_width: Optional[float] = None,
    min_support_per_sub: int = 2,  # at least this many boxes per sub-column
    min_center_gap_frac: float = 0.05,  # adjacent sub-centers must differ by >= 5% ROI width
) -> Tuple[int, np.ndarray, Dict[int, Tuple[int, int]]]:
    """
    Infer macro and sub-columns from boxes.

    Args:
        boxes: List of Box objects
        is_text_mask: Optional list[bool] same length as boxes (True for text)
        k_sub_max: Maximum number of sub-columns to try
        roi_width: Width of the ROI in pixels (for spacing sanity checks)
        min_support_per_sub: Minimum boxes per sub-column
        min_center_gap_frac: Minimum gap between sub-column centers as fraction of ROI width

    Returns:
        Tuple of (n_macro, sub_centers, assign_dict)
        where assign_dict maps box index to (macro_id, sub_id)
    """
    # Only include boxes where row_num is not None
    filtered = [(i, b) for i, b in enumerate(boxes) if b.row_num is not None]
    if not filtered:
        return 0, np.array([]), {}

    filtered_indices, filtered_boxes = zip(*filtered)
    idx_all = np.arange(len(filtered_boxes))

    # Optionally filter is_text_mask if it was provided
    filtered_is_text_mask = None
    if is_text_mask is not None:
        filtered_is_text_mask = [is_text_mask[i] for i in filtered_indices]
    if filtered_is_text_mask is not None and any(filtered_is_text_mask):
        base_idx = np.array(
            [i for i, m in enumerate(filtered_is_text_mask) if m], dtype=int
        )
    else:
        base_idx = idx_all

    if len(base_idx) < 2:
        # trivial: 1 macro, 2 sub-columns at median split
        xc_all = _xcenters(filtered_boxes).flatten()
        thr = float(np.median(xc_all)) if len(xc_all) else 0.0
        sub_centers = (
            np.array([np.min(xc_all), np.max(xc_all)])
            if len(xc_all)
            else np.array([0.25, 0.75])
        )
        assign = {}
        for idx, bxc in enumerate(xc_all):
            macro = 0
            sub = 0 if bxc <= thr else 1
            assign[filtered_indices[idx]] = (macro, sub)
            # Set macro_col_num and sub_col_num on the Box object
            boxes[filtered_indices[idx]].macro_col_num = macro
            boxes[filtered_indices[idx]].sub_col_num = sub
        return 1, np.sort(sub_centers), assign

    X = _xcenters([filtered_boxes[i] for i in base_idx])

    # Discover an even number of sub-columns via BIC (2,4,6,...)
    best = {"bic": np.inf, "k_sub": 2, "labels": None, "centers": None}
    tried_any = False

    if SKLEARN_AVAILABLE:
        try:
            for k_sub in range(2, min(k_sub_max, len(X)) + 1, 2):
                gmm = GaussianMixture(
                    n_components=k_sub, covariance_type="full", random_state=42
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                centers = np.sort(gmm.means_.flatten())
                # sanity checks: center spacing & support
                ok = True
                if roi_width is not None and k_sub > 1:
                    gaps = np.diff(centers) / max(roi_width, 1.0)
                    if np.any(gaps < min_center_gap_frac):
                        ok = False
                labels = gmm.predict(X)
                counts = np.array([np.sum(labels == j) for j in range(k_sub)])
                if np.any(counts < min_support_per_sub):
                    ok = False
                if ok and bic < best["bic"]:
                    best = {
                        "bic": bic,
                        "k_sub": k_sub,
                        "labels": labels,
                        "centers": centers,
                    }
                    tried_any = True
        except Exception:
            tried_any = False

    if not tried_any:
        # histogram fallback to 2 sub-columns
        xc = X.flatten()
        thr = np.median(xc)
        centers = np.array([np.median(xc[xc <= thr]), np.median(xc[xc > thr])])
        best = {"k_sub": 2, "centers": np.sort(centers)}

    k_sub = best["k_sub"]
    sub_centers = best["centers"]  # sorted left->right

    # Pair adjacent sub-columns into macro-columns using GAP clustering
    gaps = np.diff(sub_centers)
    if len(gaps) == 0:
        gap_thr = np.inf
    else:
        # Otsu-like split on gaps (or kmeans on 1D gaps)
        if SKLEARN_AVAILABLE:
            try:
                km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(
                    gaps.reshape(-1, 1)
                )
                centers_gap = km.cluster_centers_.flatten()
                gap_thr = float(np.mean(centers_gap))
            except Exception:
                gap_thr = float(np.percentile(gaps, 75))
        else:
            gap_thr = float(np.percentile(gaps, 75))

    # Walk through centers and form pairs
    pairs = []
    i = 0
    while i < len(sub_centers) - 1:
        pairs.append((i, i + 1))
        i += 2

    n_macro = len(pairs)

    # Assign every box to nearest sub-center, then map to (macro_id, sub_id)
    all_xc = _xcenters(boxes).flatten()
    nearest_sub = np.argmin(np.abs(all_xc[:, None] - sub_centers[None, :]), axis=1)

    # Build sub -> (macro_id, sub_id) map
    sub_to_pair: Dict[int, Tuple[int, int]] = {}
    for mid, (li, ri) in enumerate(pairs):
        sub_to_pair[li] = (mid, 0)  # left sub
        sub_to_pair[ri] = (mid, 1)  # right sub

    assign: Dict[int, Tuple[int, int]] = {}
    # Only iterate over filtered boxes
    for idx_in_filtered, filtered_idx in enumerate(filtered_indices):
        s = nearest_sub[filtered_idx]
        # clamp in case k_sub < 2
        s = int(np.clip(s, 0, len(sub_centers) - 1))
        # if odd leftover, map last sub to the closest existing pair
        if s not in sub_to_pair:
            # attach to the nearest existing sub (rare)
            alt = int(np.argmin(np.abs(sub_centers - sub_centers[s])))
            assign[filtered_idx] = sub_to_pair.get(alt, (0, 0))
            macro, sub = assign[filtered_idx]
            boxes[filtered_idx].macro_col_num = macro
            boxes[filtered_idx].sub_col_num = sub
        else:
            assign[filtered_idx] = sub_to_pair[s]
            macro, sub = assign[filtered_idx]
            boxes[filtered_idx].macro_col_num = macro
            boxes[filtered_idx].sub_col_num = sub

    return n_macro, sub_centers, assign
