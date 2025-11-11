from fastapi import APIRouter
from fastapi.responses import FileResponse
from config.settings import STATIC_DIR


router = APIRouter()


@router.get("/", include_in_schema=False)
async def root():
    if STATIC_DIR.exists():
        return FileResponse(str(STATIC_DIR / "index.html"))
    return {"message": "Recipe RAG API is running. Visit /docs for API documentation."}

