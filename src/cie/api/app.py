"""FastAPI application factory and middleware configuration."""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cie.api.dependencies import get_settings
from cie.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the Customer Intelligence Engine FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Customer Intelligence Engine API",
        version="1.0.0",
        debug=settings.debug,
        description=(
            "Production decision platform combining unsupervised RFM segmentation "
            "and supervised 90-day gross Customer Lifetime Value (CLV) regression."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Enable CORS for frontend applications
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom exception handler for consistent error formatting
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        from fastapi.encoders import jsonable_encoder

        errors = []
        for err in exc.errors():
            loc = " -> ".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Validation error")
            errors.append(f"{loc}: {msg}")

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": "; ".join(errors),
                "errors": jsonable_encoder(exc.errors()),
            },
        )

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    def root_redirect():
        return {
            "message": "Customer Intelligence Engine API is online. Visit /docs for interactive documentation.",
            "health": "/v1/health",
            "model_card": "/v1/model-card",
        }

    app.include_router(router)
    return app


app = create_app()
