from fastapi import APIRouter
from config.schemas import AskRequest
from services.pipeline import run_pipeline


router = APIRouter()


@router.post("/ask")
def ask(req: AskRequest):
    return run_pipeline(req)


@router.post("/query")
def query(req: AskRequest):
    # frontend helper route mapping to the same pipeline
    return run_pipeline(req)

