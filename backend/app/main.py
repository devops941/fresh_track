import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.api import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("freshtrack.api")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "FreshTrack REST API: Multi-Parameter Machine Learning Platform "
        "for Fruit Freshness & Quality Intelligence (Good vs. Bad classification). "
        "Built with FastAPI, Scikit-learn, and Pandas."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# -----------------------------------------------------------------------------
# CORS CONFIGURATION
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prevent stale browser caching during development/presentation
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/client"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# -----------------------------------------------------------------------------
# SAFE ERROR HANDLERS (DO NOT EXPOSE PYTHON TRACEBACKS)
# -----------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats validation errors cleanly without internal tracebacks."""
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Validation error")
        errors.append(f"{field}: {msg}")
        
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "ValidationError",
            "message": "Input validation failed. Please check the supplied telemetry fields.",
            "details": errors
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Formats HTTP exceptions consistently."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": "HTTPException",
            "message": str(exc.detail)
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catches unhandled exceptions safely without leaking stack traces."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred while processing the request. Please try again later."
        }
    )

# -----------------------------------------------------------------------------
# ROUTER MOUNTING
# -----------------------------------------------------------------------------
app.include_router(api_router)

# Root & client entry redirects
@app.get("/client", include_in_schema=False)
@app.get("/client/", include_in_schema=False)
def client_root_redirect():
    """Redirects client root requests to the modern glassmorphic home page."""
    return RedirectResponse(url="/client/home.html", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

# Mount static frontend for convenience
from fastapi.staticfiles import StaticFiles
frontend_dir = settings.BACKEND_DIR.parent / 'frontend'
if frontend_dir.exists():
    app.mount('/client', StaticFiles(directory=str(frontend_dir), html=True), name='client')


@app.get("/", tags=["Root"], summary="API Root & Discovery")
def root_discovery(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/client/home.html")
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/openapi.json"
        },
        "endpoints": {
            "health": "/api/health",
            "summary": "/api/analysis/summary",
            "fruit_analysis": "/api/analysis/fruits",
            "environmental_analysis": "/api/analysis/environment",
            "predict": "/api/predictions/predict",
            "supported_fruits": "/api/predictions/fruits"
        }
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    icon_path = frontend_dir / "assets" / "images" / "tomato.svg"
    if icon_path.exists():
        return FileResponse(str(icon_path), media_type="image/svg+xml")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
