
import logging
import os
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.services.image_processor import analyze_image_bytes
from app.services.object_detector import detect_objects
from app.services.report_generator import generate_inspection_report


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FastAPI application
app = FastAPI(
    title="SiteSight AI",
    description="An AI-assisted visual inspection API.",
    version="0.3.0",
)


# CORS configuration
# Explicitly allow the deployed Vercel frontend and local development.
allowed_origins = [
    "https://site-sight-ai.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Allow additional frontend origins through environment configuration.
configured_origins = os.getenv("FRONTEND_ORIGINS", "")

for origin in configured_origins.split(","):
    origin = origin.strip().rstrip("/")

    if origin and origin not in allowed_origins:
        allowed_origins.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Upload configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


# Request models
class ReportRequest(BaseModel):
    analysis: dict[str, Any]


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to SiteSight AI",
        "status": "running",
        "docs": "/docs",
    }


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "sitesight-ai-api",
    }


# Image analysis endpoint
@app.post("/api/v1/analyze-image")
async def analyze_image(
    file: Annotated[
        UploadFile,
        File(description="Image to analyze"),
    ],
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        await file.close()
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    try:
        contents = await file.read(MAX_FILE_SIZE + 1)
    finally:
        await file.close()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        image_statistics = analyze_image_bytes(contents)

        object_detection = await run_in_threadpool(
            detect_objects,
            contents,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Image analysis failed")

        raise HTTPException(
            status_code=500,
            detail="Image analysis failed. Check the backend logs.",
        ) from exc

    return {
        "success": True,
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size_bytes": len(contents),
        "image": image_statistics,
        "object_detection": object_detection,
        "message": "Image analyzed successfully.",
    }


# AI report generation endpoint
@app.post("/api/v1/generate-report")
async def generate_report(payload: ReportRequest):
    try:
        report = await run_in_threadpool(
            generate_inspection_report,
            payload.analysis,
        )

        return {
            "success": True,
            "model": os.getenv(
                "GEMINI_MODEL",
                "gemini-3.8-flash",
            ),
            "report": report,
        }

    except ValueError as exc:
        # For example, the Gemini API key is missing.
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Gemini report generation failed")

        raise HTTPException(
            status_code=502,
            detail=(
                "Report generation failed. Check the backend logs, "
                "API key, model availability, and quota."
            ),
        ) from exc
