from fastapi import APIRouter
from config.schemas import AskRequest
from nodes.router_node import router_node
from nodes.rewrite_node import rewrite_node
from nodes.retrieve_node import retrieve_node
from nodes.context_builder_node import build_context_node
from utils.auto_ask_runner import start_background_job, load_questions
from nodes.ood_guard_node import get_moderation_report


router = APIRouter()


@router.get("/debug/search/{query}")
def debug_search(query: str, k: int = 5):
    result = retrieve_node(query, k=k)

    return {
        "query": query,
        "branch": result["branch"],
        "found_docs": len(result["retrieved_docs"]),
        "scores": result["retrieved_scores"],
        "docs_preview": [doc[:200] for doc in result["retrieved_docs"]],
        "full_docs": result["retrieved_docs"],  # Full content for inspection
    }


@router.get("/debug/judge/{query}")
def debug_judge(query: str, k: int = 3):
    """Debug endpoint to see what the judge evaluates.

    If no documents are retrieved, return a clarification prompt instead of an
    ungrounded recipe answer to reflect production behavior more closely.
    """
    from nodes.generate_node_v2 import generate_node, generate_with_history
    from nodes.relevance_check_node import relevance_check_node
    from nodes.context_builder_node import build_context_with_images

    # Retrieve
    result = retrieve_node(query, k=k)
    docs = result["retrieved_docs"]
    images = result.get("retrieved_images", [])

    # Build context
    context, _, _ = build_context_with_images(docs, images)

    # Router intent (fallback to recipe)
    route = router_node(query)
    intent = route.get("intent") or "recipe"

    if not docs:
        answer = generate_with_history(
            query=query,
            intent="clarify",
            context="clarify_mode",
            conversation_history=[],
            model=None,
        )
        judge_branch = None
    else:
        answer = generate_node(query, intent, context)
        judge_result = relevance_check_node(answer, docs)
        judge_branch = judge_result.get("branch")

    return {
        "query": query,
        "retrieved_docs_count": len(docs),
        "retrieved_docs": docs,
        "context_length": len(context),
        "context": context,
        "moderation": get_moderation_report(query),
        "intent": intent,
        "answer": answer,
        "judge_verdict": judge_branch,
        "scores": result["retrieved_scores"],
    }


@router.post("/debug/pipeline")
def debug_pipeline(req: AskRequest):
    route = router_node(req.query)
    rewritten = rewrite_node(req.query) if req.enable_rewrite else req.query
    retrieve_result = retrieve_node(rewritten, req.k)
    context = build_context_node(retrieve_result["retrieved_docs"])

    return {
        "original_query": req.query,
        "step1_router": route,
        "step2_rewritten": rewritten,
        "step3_retrieve": {
            "branch": retrieve_result["branch"],
            "doc_count": len(retrieve_result["retrieved_docs"]),
            "scores": retrieve_result["retrieved_scores"][:3],
        },
        "step4_context_len": len(context),
        "step5_ready_for_generate": True,
    }


@router.get("/debug/auto_ask/preview")
def debug_auto_ask_preview():
    qs = load_questions()
    return {
        "questions": len(qs),
        "sample": qs[:5],
    }


@router.post("/debug/auto_ask/run")
def debug_auto_ask_run():
    meta = start_background_job()
    return meta


@router.get("/debug/moderation/{query}")
def debug_moderation(query: str):
    rep = get_moderation_report(query)
    return rep or {"flagged": False, "categories": {}, "category_scores": {}}
