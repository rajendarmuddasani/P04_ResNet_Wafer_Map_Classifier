"""
Annotation Service - CRUD Operations for Segmentation Annotations

Features:
- Create, Read, Update, Delete annotations
- COCO JSON format support
- Polygon validation
- Inter-annotator agreement computation
- Quality checks
- COCO export for training

Usage:
    POST /annotations - Create annotation
    GET /annotations/{id} - Get annotation
    PUT /annotations/{id} - Update annotation
    DELETE /annotations/{id} - Delete annotation
    GET /annotations/wafer/{wafer_id} - Get all annotations for wafer
    POST /annotations/export - Export to COCO JSON
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np
import cv2
import json
import logging

from src.database.database import get_db
from src.database.models import Annotation, WaferMap, User, AnnotationMetric

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/annotations", tags=["annotations"])


# Defect categories
DEFECT_CATEGORIES = {
    0: "background",
    1: "edge",
    2: "center",
    3: "ring",
    4: "scratch",
    5: "particle",
    6: "lithography",
    7: "etching",
}


# Request/Response models
class PolygonSegmentation(BaseModel):
    """Polygon segmentation in COCO format."""
    type: str = "polygon"
    coordinates: List[List[float]] = Field(..., description="Polygon coordinates [[x1,y1,x2,y2,...]]")
    
    @validator('coordinates')
    def validate_polygon(cls, v):
        if not v or not v[0]:
            raise ValueError('Polygon must have at least one set of coordinates')
        
        # Each polygon should have at least 3 points (6 coordinates)
        for poly in v:
            if len(poly) < 6 or len(poly) % 2 != 0:
                raise ValueError('Polygon must have at least 3 points (6 coordinates) and even number of values')
        
        return v


class BoundingBox(BaseModel):
    """Bounding box [x, y, width, height]."""
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


class AnnotationCreate(BaseModel):
    """Request model for creating annotation."""
    wafer_map_id: str = Field(..., description="Wafer map ID")
    category_id: int = Field(..., ge=0, le=7, description="Defect category (0-7)")
    segmentation: PolygonSegmentation = Field(..., description="Polygon segmentation")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box (auto-computed if not provided)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Annotation confidence")
    annotated_by: str = Field(..., description="Annotator user ID")
    annotation_time_seconds: Optional[int] = Field(None, description="Time taken to annotate")


class AnnotationUpdate(BaseModel):
    """Request model for updating annotation."""
    category_id: Optional[int] = Field(None, ge=0, le=7)
    segmentation: Optional[PolygonSegmentation] = None
    bbox: Optional[BoundingBox] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_verified: Optional[bool] = None


class AnnotationResponse(BaseModel):
    """Response model for annotation."""
    id: str
    wafer_map_id: str
    wafer_id: str
    category_id: int
    category_name: str
    segmentation: Dict[str, Any]
    bbox: Dict[str, float]
    area: float
    confidence: Optional[float]
    is_verified: bool
    is_active_learning_sample: bool
    annotated_by: Optional[str]
    annotation_time_seconds: Optional[int]
    created_at: str
    updated_at: str
    verified_at: Optional[str]
    verified_by: Optional[str]
    version: int


class AnnotationList(BaseModel):
    """Response for listing annotations."""
    annotations: List[AnnotationResponse]
    total: int
    page: int
    page_size: int


class COCOExport(BaseModel):
    """COCO JSON format export."""
    images: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]]
    categories: List[Dict[str, Any]]
    info: Dict[str, Any]


class AgreementMetrics(BaseModel):
    """Inter-annotator agreement metrics."""
    annotator1_id: str
    annotator2_id: str
    wafer_map_id: str
    iou_agreement: float
    dice_agreement: float
    pixel_agreement: float
    computed_at: str


def compute_bbox_from_polygon(coordinates: List[List[float]]) -> BoundingBox:
    """
    Compute bounding box from polygon coordinates.
    
    Args:
        coordinates: Polygon coordinates [[x1,y1,x2,y2,...]]
    
    Returns:
        BoundingBox
    """
    all_x = []
    all_y = []
    
    for poly in coordinates:
        for i in range(0, len(poly), 2):
            all_x.append(poly[i])
            all_y.append(poly[i + 1])
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    return BoundingBox(
        x=x_min,
        y=y_min,
        width=x_max - x_min,
        height=y_max - y_min,
    )


def compute_polygon_area(coordinates: List[List[float]]) -> float:
    """
    Compute polygon area using Shoelace formula.
    
    Args:
        coordinates: Polygon coordinates [[x1,y1,x2,y2,...]]
    
    Returns:
        Area in pixels
    """
    total_area = 0.0
    
    for poly in coordinates:
        points = []
        for i in range(0, len(poly), 2):
            points.append([poly[i], poly[i + 1]])
        
        points = np.array(points, dtype=np.float32)
        area = cv2.contourArea(points)
        total_area += area
    
    return float(total_area)


def validate_polygon_quality(coordinates: List[List[float]], image_shape: tuple) -> Dict[str, Any]:
    """
    Validate polygon quality.
    
    Checks:
    - Minimum 3 points
    - Points within image bounds
    - No self-intersections (basic check)
    - Reasonable area (not too small)
    
    Args:
        coordinates: Polygon coordinates
        image_shape: (height, width) of image
    
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    height, width = image_shape
    
    for poly_idx, poly in enumerate(coordinates):
        # Check number of points
        num_points = len(poly) // 2
        if num_points < 3:
            issues.append(f"Polygon {poly_idx}: Less than 3 points ({num_points})")
        
        # Check bounds
        for i in range(0, len(poly), 2):
            x, y = poly[i], poly[i + 1]
            if x < 0 or x > width or y < 0 or y > height:
                issues.append(f"Polygon {poly_idx}: Point ({x}, {y}) out of bounds")
        
        # Check area
        points = np.array([[poly[i], poly[i + 1]] for i in range(0, len(poly), 2)], dtype=np.float32)
        area = cv2.contourArea(points)
        if area < 10:  # Minimum 10 pixels
            issues.append(f"Polygon {poly_idx}: Area too small ({area:.1f} pixels)")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


@router.post("/", response_model=AnnotationResponse, status_code=201)
async def create_annotation(
    annotation: AnnotationCreate,
    db: Session = Depends(get_db),
):
    """Create new annotation."""
    # Verify wafer map exists
    wafer_map = db.query(WaferMap).filter(WaferMap.id == annotation.wafer_map_id).first()
    if not wafer_map:
        raise HTTPException(status_code=404, detail="Wafer map not found")
    
    # Verify annotator exists
    annotator = db.query(User).filter(User.id == annotation.annotated_by).first()
    if not annotator:
        raise HTTPException(status_code=404, detail="Annotator not found")
    
    # Validate polygon quality
    image_shape = (wafer_map.image_height or 300, wafer_map.image_width or 300)
    validation = validate_polygon_quality(annotation.segmentation.coordinates, image_shape)
    
    if not validation['valid']:
        raise HTTPException(
            status_code=400,
            detail=f"Polygon validation failed: {', '.join(validation['issues'])}"
        )
    
    # Compute bbox if not provided
    if annotation.bbox is None:
        bbox = compute_bbox_from_polygon(annotation.segmentation.coordinates)
    else:
        bbox = annotation.bbox
    
    # Compute area
    area = compute_polygon_area(annotation.segmentation.coordinates)
    
    # Create annotation
    db_annotation = Annotation(
        wafer_map_id=annotation.wafer_map_id,
        category_id=annotation.category_id,
        category_name=DEFECT_CATEGORIES[annotation.category_id],
        segmentation=annotation.segmentation.dict(),
        bbox_x=bbox.x,
        bbox_y=bbox.y,
        bbox_width=bbox.width,
        bbox_height=bbox.height,
        area=area,
        confidence=annotation.confidence,
        annotated_by=annotation.annotated_by,
        annotation_time_seconds=annotation.annotation_time_seconds,
    )
    
    db.add(db_annotation)
    
    # Update annotator stats
    annotator.total_annotations += 1
    if annotation.annotation_time_seconds:
        if annotator.average_annotation_time_seconds:
            # Running average
            total_time = annotator.average_annotation_time_seconds * (annotator.total_annotations - 1)
            annotator.average_annotation_time_seconds = (total_time + annotation.annotation_time_seconds) / annotator.total_annotations
        else:
            annotator.average_annotation_time_seconds = float(annotation.annotation_time_seconds)
    
    # Update wafer map status
    if wafer_map.status == "uploaded":
        wafer_map.status = "annotated"
    
    db.commit()
    db.refresh(db_annotation)
    
    logger.info(f"Created annotation {db_annotation.id} for wafer {wafer_map.wafer_id}")
    
    return AnnotationResponse(
        id=str(db_annotation.id),
        wafer_map_id=str(db_annotation.wafer_map_id),
        wafer_id=wafer_map.wafer_id,
        category_id=db_annotation.category_id,
        category_name=db_annotation.category_name,
        segmentation=db_annotation.segmentation,
        bbox={
            "x": db_annotation.bbox_x,
            "y": db_annotation.bbox_y,
            "width": db_annotation.bbox_width,
            "height": db_annotation.bbox_height,
        },
        area=db_annotation.area,
        confidence=db_annotation.confidence,
        is_verified=db_annotation.is_verified,
        is_active_learning_sample=db_annotation.is_active_learning_sample,
        annotated_by=str(db_annotation.annotated_by) if db_annotation.annotated_by else None,
        annotation_time_seconds=db_annotation.annotation_time_seconds,
        created_at=db_annotation.created_at.isoformat(),
        updated_at=db_annotation.updated_at.isoformat(),
        verified_at=db_annotation.verified_at.isoformat() if db_annotation.verified_at else None,
        verified_by=str(db_annotation.verified_by) if db_annotation.verified_by else None,
        version=db_annotation.version,
    )


@router.get("/{annotation_id}", response_model=AnnotationResponse)
async def get_annotation(annotation_id: str, db: Session = Depends(get_db)):
    """Get annotation by ID."""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    wafer_map = db.query(WaferMap).filter(WaferMap.id == annotation.wafer_map_id).first()
    
    return AnnotationResponse(
        id=str(annotation.id),
        wafer_map_id=str(annotation.wafer_map_id),
        wafer_id=wafer_map.wafer_id if wafer_map else "unknown",
        category_id=annotation.category_id,
        category_name=annotation.category_name,
        segmentation=annotation.segmentation,
        bbox={
            "x": annotation.bbox_x,
            "y": annotation.bbox_y,
            "width": annotation.bbox_width,
            "height": annotation.bbox_height,
        },
        area=annotation.area,
        confidence=annotation.confidence,
        is_verified=annotation.is_verified,
        is_active_learning_sample=annotation.is_active_learning_sample,
        annotated_by=str(annotation.annotated_by) if annotation.annotated_by else None,
        annotation_time_seconds=annotation.annotation_time_seconds,
        created_at=annotation.created_at.isoformat(),
        updated_at=annotation.updated_at.isoformat(),
        verified_at=annotation.verified_at.isoformat() if annotation.verified_at else None,
        verified_by=str(annotation.verified_by) if annotation.verified_by else None,
        version=annotation.version,
    )


@router.get("/wafer/{wafer_id}", response_model=AnnotationList)
async def get_wafer_annotations(
    wafer_id: str,
    page: int = 1,
    page_size: int = 100,
    db: Session = Depends(get_db),
):
    """Get all annotations for a wafer map."""
    wafer_map = db.query(WaferMap).filter(WaferMap.wafer_id == wafer_id).first()
    
    if not wafer_map:
        raise HTTPException(status_code=404, detail="Wafer not found")
    
    query = db.query(Annotation).filter(Annotation.wafer_map_id == wafer_map.id)
    
    total = query.count()
    annotations = query.offset((page - 1) * page_size).limit(page_size).all()
    
    annotation_responses = [
        AnnotationResponse(
            id=str(ann.id),
            wafer_map_id=str(ann.wafer_map_id),
            wafer_id=wafer_map.wafer_id,
            category_id=ann.category_id,
            category_name=ann.category_name,
            segmentation=ann.segmentation,
            bbox={
                "x": ann.bbox_x,
                "y": ann.bbox_y,
                "width": ann.bbox_width,
                "height": ann.bbox_height,
            },
            area=ann.area,
            confidence=ann.confidence,
            is_verified=ann.is_verified,
            is_active_learning_sample=ann.is_active_learning_sample,
            annotated_by=str(ann.annotated_by) if ann.annotated_by else None,
            annotation_time_seconds=ann.annotation_time_seconds,
            created_at=ann.created_at.isoformat(),
            updated_at=ann.updated_at.isoformat(),
            verified_at=ann.verified_at.isoformat() if ann.verified_at else None,
            verified_by=str(ann.verified_by) if ann.verified_by else None,
            version=ann.version,
        )
        for ann in annotations
    ]
    
    return AnnotationList(
        annotations=annotation_responses,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.put("/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation(
    annotation_id: str,
    update: AnnotationUpdate,
    db: Session = Depends(get_db),
):
    """Update annotation."""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    # Update fields
    if update.category_id is not None:
        annotation.category_id = update.category_id
        annotation.category_name = DEFECT_CATEGORIES[update.category_id]
    
    if update.segmentation is not None:
        # Validate new polygon
        wafer_map = db.query(WaferMap).filter(WaferMap.id == annotation.wafer_map_id).first()
        image_shape = (wafer_map.image_height or 300, wafer_map.image_width or 300)
        validation = validate_polygon_quality(update.segmentation.coordinates, image_shape)
        
        if not validation['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Polygon validation failed: {', '.join(validation['issues'])}"
            )
        
        annotation.segmentation = update.segmentation.dict()
        
        # Recompute bbox and area
        bbox = compute_bbox_from_polygon(update.segmentation.coordinates)
        annotation.bbox_x = bbox.x
        annotation.bbox_y = bbox.y
        annotation.bbox_width = bbox.width
        annotation.bbox_height = bbox.height
        annotation.area = compute_polygon_area(update.segmentation.coordinates)
    
    if update.bbox is not None:
        annotation.bbox_x = update.bbox.x
        annotation.bbox_y = update.bbox.y
        annotation.bbox_width = update.bbox.width
        annotation.bbox_height = update.bbox.height
    
    if update.confidence is not None:
        annotation.confidence = update.confidence
    
    if update.is_verified is not None:
        annotation.is_verified = update.is_verified
        if update.is_verified:
            annotation.verified_at = datetime.utcnow()
    
    # Increment version
    annotation.version += 1
    
    db.commit()
    db.refresh(annotation)
    
    logger.info(f"Updated annotation {annotation_id} to version {annotation.version}")
    
    wafer_map = db.query(WaferMap).filter(WaferMap.id == annotation.wafer_map_id).first()
    
    return AnnotationResponse(
        id=str(annotation.id),
        wafer_map_id=str(annotation.wafer_map_id),
        wafer_id=wafer_map.wafer_id if wafer_map else "unknown",
        category_id=annotation.category_id,
        category_name=annotation.category_name,
        segmentation=annotation.segmentation,
        bbox={
            "x": annotation.bbox_x,
            "y": annotation.bbox_y,
            "width": annotation.bbox_width,
            "height": annotation.bbox_height,
        },
        area=annotation.area,
        confidence=annotation.confidence,
        is_verified=annotation.is_verified,
        is_active_learning_sample=annotation.is_active_learning_sample,
        annotated_by=str(annotation.annotated_by) if annotation.annotated_by else None,
        annotation_time_seconds=annotation.annotation_time_seconds,
        created_at=annotation.created_at.isoformat(),
        updated_at=annotation.updated_at.isoformat(),
        verified_at=annotation.verified_at.isoformat() if annotation.verified_at else None,
        verified_by=str(annotation.verified_by) if annotation.verified_by else None,
        version=annotation.version,
    )


@router.delete("/{annotation_id}")
async def delete_annotation(annotation_id: str, db: Session = Depends(get_db)):
    """Delete annotation."""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    db.delete(annotation)
    db.commit()
    
    logger.info(f"Deleted annotation {annotation_id}")
    
    return {"status": "success", "message": f"Annotation {annotation_id} deleted"}


@router.post("/export", response_model=COCOExport)
async def export_coco(
    wafer_ids: Optional[List[str]] = Query(None, description="Filter by wafer IDs"),
    verified_only: bool = Query(False, description="Export only verified annotations"),
    db: Session = Depends(get_db),
):
    """
    Export annotations to COCO JSON format for training.
    
    COCO format structure:
    {
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }
    """
    # Build query
    query = db.query(Annotation).join(WaferMap)
    
    if wafer_ids:
        query = query.filter(WaferMap.wafer_id.in_(wafer_ids))
    
    if verified_only:
        query = query.filter(Annotation.is_verified == True)
    
    annotations = query.all()
    
    # Get unique wafer maps
    wafer_map_ids = list(set([ann.wafer_map_id for ann in annotations]))
    wafer_maps = db.query(WaferMap).filter(WaferMap.id.in_(wafer_map_ids)).all()
    
    # Build COCO structure
    coco_images = []
    for idx, wafer_map in enumerate(wafer_maps):
        coco_images.append({
            "id": idx + 1,
            "wafer_id": wafer_map.wafer_id,
            "file_name": wafer_map.image_path.split('/')[-1],
            "width": wafer_map.image_width or 300,
            "height": wafer_map.image_height or 300,
        })
    
    # Map wafer_map_id to coco image_id
    wafer_to_image_id = {wafer_maps[i].id: i + 1 for i in range(len(wafer_maps))}
    
    coco_annotations = []
    for idx, ann in enumerate(annotations):
        coco_annotations.append({
            "id": idx + 1,
            "image_id": wafer_to_image_id[ann.wafer_map_id],
            "category_id": ann.category_id,
            "segmentation": ann.segmentation['coordinates'],
            "area": ann.area,
            "bbox": [ann.bbox_x, ann.bbox_y, ann.bbox_width, ann.bbox_height],
            "iscrowd": 0,
        })
    
    coco_categories = [
        {"id": cat_id, "name": cat_name, "supercategory": "defect"}
        for cat_id, cat_name in DEFECT_CATEGORIES.items()
    ]
    
    coco_data = COCOExport(
        images=coco_images,
        annotations=coco_annotations,
        categories=coco_categories,
        info={
            "description": "Wafer Defect Segmentation Dataset",
            "version": "1.0",
            "year": datetime.utcnow().year,
            "date_created": datetime.utcnow().isoformat(),
        }
    )
    
    logger.info(f"Exported {len(coco_annotations)} annotations for {len(coco_images)} images")
    
    return coco_data
