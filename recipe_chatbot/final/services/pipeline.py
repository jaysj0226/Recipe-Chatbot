"""Pipeline service for the main RAG flow."""
from __future__ import annotations

import re
from typing import Optional
from config.settings import (
    SCORE_THRESHOLD,
    ALLOW_NO_CONTEXT_ANSWER,
    ENABLE_CRAG,
    RERANK_MMR,
    MMR_FETCH,
    MMR_LAMBDA,
    SIMILARITY_THRESHOLD,
    DOMAIN_CAP,
)
from config.schemas import AskRequest
from utils.conversation_memory import memory_manager

from nodes.router_node import router_node
from nodes.rewrite_node import rewrite_node
from nodes.retrieve_node import retrieve_node
from nodes.context_builder_node import build_context_with_images
from nodes.generate_node_v2 import generate_with_history, extract_target_dish
from nodes.relevance_check_node import relevance_check_node
from nodes.ood_guard_node import ood_guard
from config.settings import USE_CE_RERANK, CE_MODEL, CE_TOPN, DEBUG_RAW, LOWCONF_MODE, MIN_CONF_DOCS
from utils.reranker import rerank_pairs


def _sanitize_answer_links(answer: str, sources: list[dict]) -> tuple[str, list[str]]:
    """
    Remove or mask any URLs in the answer that are not present in provided sources.

    Returns (sanitized_answer, removed_urls).
    """
    if not answer:
        return answer, []

    # Extract URLs from the answer body
    urls = set(re.findall(r"https?://\S+", answer or ""))
    if not urls:
        return answer, []

    allowed = set()
    for s in sources or []:
        try:
            u = (s.get("url") or "").strip()
            if u:
                allowed.add(u)
        except Exception as e:
            if DEBUG_RAW:
                print(f"where hint: {e}")
            continue

    removed = []
    for u in urls:
        if u not in allowed:
            # Replace unknown URL with a neutral note
            answer = answer.replace(u, "[출처: 아래 목록 참조]")
            removed.append(u)

    return answer, removed


def _remove_links_in_body(answer: str) -> tuple[str, list[str]]:
    """Remove all links from the answer body.

    - Markdown links: [text](http://...) -> text
    - Raw URLs: http(s)://... -> removed
    Returns (sanitized_answer, removed_urls)
    """
    if not answer:
        return answer, []

    removed: list[str] = []

    # 1) Markdown links -> keep text only
    def _md_repl(m: re.Match) -> str:
        url = m.group(2)
        removed.append(url)
        return m.group(1)

    answer = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", _md_repl, answer)

    # 2) Raw URLs
    urls = re.findall(r"https?://\S+", answer)
    if urls:
        removed.extend(urls)
        answer = re.sub(r"https?://\S+", "", answer)

    # Collapse extra spaces left by removals
    answer = re.sub(r"\s+\)\s*", ") ", answer)
    answer = re.sub(r"\s+\]", "]", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

    return answer, list(dict.fromkeys(removed))


def _strip_sources_section(answer: str) -> tuple[str, bool]:
    """Strip a trailing '출처' section from the answer body if present.

    Recognizes lines starting with '출처' or numbered '7) 출처'. Returns
    (text_without_sources, stripped: bool).
    """
    if not answer:
        return answer, False

    lines = answer.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        l = line.strip()
        if re.match(r"^(\d+\)\s*)?출처\s*:?.*$", l):
            start_idx = i
            break

    if start_idx is None:
        return answer, False

    kept = lines[:start_idx]
    text = "\n".join(kept).rstrip()
    return text, True


def _should_skip_ood_short_followup(query: str, session_id: str) -> bool:
    """Return True when the query is a very short follow-up and we have recent context.

    This helps avoid blocking with OOD when the user is just replying to our
    previous question (e.g., "네", "처음이에요").
    """
    try:
        t = (query or "").strip()
        if (len(t) <= 4) or (len(t.split()) <= 2):
            summary = memory_manager.get_context_summary(session_id)
            return bool(summary)
        return False
    except Exception:
        return False


def run_pipeline(req: AskRequest) -> dict:
    """Execute the end-to-end RAG pipeline and return the API response payload."""
    original_query = req.query
    pipeline_steps: list[str] = []
    # Image controls
    include_images: bool = getattr(req, "include_images", True)
    image_policy: str = getattr(req, "image_policy", "strict")  # strict | lenient | always
    max_images: int = max(0, int(getattr(req, "max_images", 5)))

    # Session handling
    session_id = req.session_id
    if not session_id or not memory_manager.get_session(session_id):
        session_id = memory_manager.create_session()
        is_new_session = True
    else:
        is_new_session = False

    conversation_history = memory_manager.get_history(session_id, as_langchain=True)

    # Decision-state helpers
    def _get_pending_decision():
        sess = memory_manager.get_session(session_id)
        if not sess:
            return None
        return (sess.get("metadata") or {}).get("pending_decision")

    def _set_pending_decision(info: dict):
        memory_manager.update_metadata(session_id, "pending_decision", info)

    def _clear_pending_decision():
        sess = memory_manager.get_session(session_id)
        if sess and "pending_decision" in sess.get("metadata", {}):
            sess["metadata"].pop("pending_decision", None)

    def _parse_decision(decision_text: Optional[str], fallback_query: str) -> Optional[str]:
        t = (decision_text or "").strip().lower()
        if t in {"proceed", "1", "진행", "그대로 진행", "계속", "예"}:
            return "proceed"
        if t in {"clarify", "2", "질문 다듬기", "다듬기", "수정", "아니오"}:
            return "clarify"
        # Parse from the actual query text if decision param missing
        q = (fallback_query or "").strip().lower()
        if q in {"1", "진행", "그대로 진행", "계속", "proceed"}:
            return "proceed"
        if q in {"2", "clarify", "질문 다듬기", "다듬기", "수정"}:
            return "clarify"
        return None

    # Handle pending low-confidence decision
    pending = _get_pending_decision()
    allow_low_override = False
    if pending:
        user_decision = _parse_decision(getattr(req, "decision", None), req.query)
        if user_decision == "proceed":
            allow_low_override = True
            _clear_pending_decision()
        elif user_decision == "clarify":
            clarify_text = generate_with_history(
                query=req.query,
                intent="clarify",
                context="clarify_mode",
                conversation_history=conversation_history,
                model=req.model,
            )
            _clear_pending_decision()
            return {
                "answer": clarify_text,
                "router": {"intent": "clarify", "needs_retrieval": False, "notes": "pending_decision"},
                "intent": "clarify",
                "original_query": req.query,
                "rewritten_query": None,
                "context_len": 0,
                "used_docs": 0,
                "context_found": False,
                "retrieved_count": 0,
                "retrieved_scores": [],
                "k": req.k,
                "mode": "clarify",
                "branch": "decision_clarify",
                "pipeline": ["decision_clarify"],
                "session_id": session_id,
                "is_new_session": is_new_session,
                "history_used": len(conversation_history) > 0,
                "conversation_turns": len(conversation_history) // 2,
                "sources": [],
                "low_confidence": True,
                "warning": "사용자 선택에 따라 질문 다듬기 제안 제공",
                "decision_required": False,
                "suggested_actions": [],
            }
        else:
            # Re-prompt decision without clearing the state
            warning = (
                "저신뢰 상태입니다. 1) 그대로 진행(정확도 낮음 허용), 2) 질문 다듬기 중 선택해 주세요."
            )
            return {
                "answer": warning,
                "router": {"intent": "clarify", "needs_retrieval": False, "notes": "pending_decision"},
                "intent": "clarify",
                "original_query": req.query,
                "rewritten_query": None,
                "context_len": 0,
                "used_docs": 0,
                "context_found": False,
                "retrieved_count": 0,
                "retrieved_scores": [],
                "k": req.k,
                "mode": "clarify",
                "branch": "decision_pending",
                "pipeline": ["decision_pending"],
                "session_id": session_id,
                "is_new_session": is_new_session,
                "history_used": len(conversation_history) > 0,
                "conversation_turns": len(conversation_history) // 2,
                "sources": [],
                "low_confidence": True,
                "warning": warning,
                "decision_required": True,
                "suggested_actions": ["proceed_with_low_confidence", "clarify"],
            }

    # 0) Pre-router OOD guard (fast keyword + small LLM)
    try:
        ood = ood_guard(original_query)
        pipeline_steps.append("ood_guard")
        if (not _should_skip_ood_short_followup(original_query, session_id)) and ood.get("branch") == "out":
            answer = ood.get(
                "answer",
                "죄송해요. 요리·레시피·조리·재료·보관·영양 관련 질문만 도와드릴 수 있어요.",
            )
            response = {
                "answer": answer,
                "router": {"intent": "out_of_domain", "needs_retrieval": False, "notes": "pre_ood_guard"},
                "intent": "out_of_domain",
                "original_query": original_query,
                "rewritten_query": None,
                "context_len": 0,
                "used_docs": 0,
                "context_found": False,
                "retrieved_count": 0,
                "retrieved_scores": [],
                "image_urls": [],
                "k": req.k,
                "mode": "ood_block",
                "branch": "out_of_domain",
                "pipeline": pipeline_steps,
                "session_id": session_id,
                "is_new_session": is_new_session,
                "history_used": len(conversation_history) > 0,
                "conversation_turns": len(conversation_history) // 2,
                "sources": [],
            }
            memory_manager.add_message(session_id, "user", original_query)
            memory_manager.add_message(session_id, "assistant", answer, {"intent": "out_of_domain"})
            return response
    except Exception:
        # best-effort; continue to router on any error
        pass

    # 1) Router
    route = router_node(original_query)
    intent = route["intent"]
    needs_retrieval = route["needs_retrieval"]
    pipeline_steps.append("router")

    # 1.5) Clarify-First branch for short/ambiguous queries
    def _needs_clarify_first(q: str, intent_label: str) -> bool:
        if not q:
            return True
        qn = q.strip()
        # very short or 1-2 tokens
        if len(qn) <= 4 or len(qn.split()) <= 1:
            # allow short but specific dish like "된장찌개" by checking target extraction
            dish = extract_target_dish(qn)
            return not bool(dish)
        # generic how-to without a clear target
        '''
        if re.search(r"(어떻게|방법|레시피)", qn) and not extract_target_dish(qn):
            return True
        '''
        if qn in {"어떻게","방법","뭐","뭘","어디","언제"}:
            return True
        # storage/substitution/nutrition intents are often fine without clarify
        if intent_label in {"storage", "substitution", "nutrition"}:
            return False
        return False

    if _needs_clarify_first(original_query, intent):
        clarify_text = generate_with_history(
            query=original_query,
            intent="clarify",
            context="clarify_mode",
            conversation_history=conversation_history,
            model=req.model,
        )
        pipeline_steps.append("clarify_first")
        response = {
            "answer": clarify_text,
            "router": route,
            "intent": "clarify",
            "original_query": original_query,
            "rewritten_query": None,
            "context_len": 0,
            "used_docs": 0,
            "context_found": False,
            "retrieved_count": 0,
            "retrieved_scores": [],
            "image_urls": [],
            "k": req.k,
            "mode": "clarify",
            "branch": "clarify_first",
            "pipeline": pipeline_steps,
            "session_id": session_id,
            "is_new_session": is_new_session,
            "history_used": len(conversation_history) > 0,
            "conversation_turns": len(conversation_history) // 2,
            "sources": [],
        }
        memory_manager.add_message(session_id, "user", original_query)
        memory_manager.add_message(
            session_id,
            "assistant",
            clarify_text,
            {"intent": "clarify", "context_found": False, "used_docs": 0, "clarify_stage": "first"},
        )
        return response

    # OOD
    if intent == "out_of_domain":
        response = {
            "answer": "죄송해요. 요리·레시피·조리·재료·보관·영양 관련 질문에만 답할 수 있어요.",
            "retrieved_count": 0,
            "k": req.k,
            "sources": [],
            "router": route,
            "pipeline": pipeline_steps,
            "session_id": session_id,
            "is_new_session": is_new_session,
            "history_used": False,
            "conversation_turns": len(conversation_history) // 2,
        }
        memory_manager.add_message(session_id, "user", original_query)
        memory_manager.add_message(
            session_id,
            "assistant",
            response["answer"],
            {"intent": intent, "out_of_domain": True},
        )
        return response

    # 2) (optional) Rewrite
    query_for_search = original_query
    if req.enable_rewrite and needs_retrieval:
        query_for_search = rewrite_node(original_query)
        pipeline_steps.append("rewrite")

    # 3) Retrieve
    retrieve_result = {
        "retrieved_docs": [],
        "retrieved_scores": [],
        "retrieved_images": [],
        "branch": "no_docs",
    }
    if needs_retrieval:
        retrieve_result = retrieve_node(query_for_search, req.k)
        pipeline_steps.append("retrieve")

    docs = retrieve_result["retrieved_docs"]
    scores = retrieve_result["retrieved_scores"]
    images = retrieve_result.get("retrieved_images", [])
    branch = retrieve_result["branch"]
    metas = retrieve_result.get("retrieved_meta", [])
    score_mode = retrieve_result.get("score_mode", "distance")

    # Optional cross-encoder rerank on topN candidates
    if USE_CE_RERANK and docs:
        try:
            topn = min(len(docs), max(1, int(CE_TOPN)))
            order = rerank_pairs(original_query, docs[:topn], topn=topn, model_name=CE_MODEL)

            # Reorder docs/images/scores/metas for topn chunk, then append the rest
            def _reorder(lst, default_val=None):
                prefix = [lst[i] for i in order if i < len(lst)]
                suffix = [lst[i] for i in range(topn, len(lst))]
                return prefix + suffix

            docs = _reorder(docs)
            images = _reorder(images)
            metas = _reorder(metas)
            # scores may contain None; slice length-safe
            scores = _reorder(scores)
            pipeline_steps.append("rerank_ce")
        except Exception as e:
            if DEBUG_RAW: 
                print(f"where hint: {e}")

    # 3.5) Clarify branch when no documents found
    if needs_retrieval and not docs:
        pipeline_steps.append("clarify")

        # Generate concise clarification questions
        answer = generate_with_history(
            query=original_query,
            intent="clarify",
            context="clarify_mode",  # non-empty to avoid no-context refusal
            conversation_history=conversation_history,
            model=req.model,
        )

        response = {
            "answer": answer,
            "router": route,
            "intent": "clarify",
            "original_query": original_query,
            "rewritten_query": query_for_search if req.enable_rewrite else None,
            "context_len": 0,
            "used_docs": 0,
            "context_found": False,
            "retrieved_count": 0,
            "retrieved_scores": [],
            "image_urls": [],
            "k": req.k,
            "mode": "clarify",
            "branch": branch,
            "pipeline": pipeline_steps,
            "session_id": session_id,
            "is_new_session": is_new_session,
            "history_used": len(conversation_history) > 0,
            "conversation_turns": len(conversation_history) // 2,
            "judge_verdict_1": None,
            "judge_verdict_2": None,
            "corrected": False,
            "final_pass": 1,
            "sources": [],
        }

        memory_manager.add_message(session_id, "user", original_query)
        memory_manager.add_message(
            session_id,
            "assistant",
            answer,
            {"intent": "clarify", "context_found": False, "used_docs": 0},
        )
        return response

    # 4) Context build (+aligned images)
    context_text = ""
    sources: list[dict] = []
    if docs:
        try:
            if (
                scores
                and len(scores) == len(docs)
                and SCORE_THRESHOLD
                and SCORE_THRESHOLD > 0
            ):
                paired = [
                    (d, i, s)
                    for d, i, s in zip(docs, images, scores)
                    if (s is not None and s >= SCORE_THRESHOLD)
                ]
                if paired:
                    docs, images, scores = [list(x) for x in zip(*paired)]
        except Exception as e:
            if DEBUG_RAW:
                print(f"where hint: {e}")

        context_text, selected_images, selected_docs_texts = build_context_with_images(
            docs, images
        )
        images = selected_images
        pipeline_steps.append("context_builder")

        # Build sources aligned to selected docs (up to 3)
        try:
            index_map = {}
            for idx, d in enumerate(docs):
                if isinstance(d, str) and d not in index_map:
                    index_map[d] = idx
            for d in selected_docs_texts:
                idx = index_map.get(d)
                if idx is None:
                    continue
                meta = metas[idx] if idx < len(metas) else {}
                t = (meta.get("title") or "").strip() if isinstance(meta, dict) else ""
                u = (meta.get("url") or "").strip() if isinstance(meta, dict) else ""
                if t or u:
                    sources.append({"title": t, "url": u})
                if len(sources) >= 3:
                    break
        except Exception as e:
            if DEBUG_RAW:
                print(f"source_align_error: {e}")
            

    # 5) Generate with history
    answer = generate_with_history(
        query=original_query,
        intent=intent,
        context=context_text,
        conversation_history=conversation_history,
        model=req.model,
    )
    pipeline_steps.append("generate")

    # Heuristic image gating pre-judge
    try:
        from nodes.generate_node_v2 import extract_target_dish as _extract

        # Early exit if images disabled
        if not include_images:
            images = []
        else:
            # Allowed intents gating (applies to strict and lenient)
            allowed_image_intents = {"recipe", "dish_overview", "substitution", "storage"}
            if image_policy in ("strict", "lenient") and intent not in allowed_image_intents:
                images = []
            else:
                # dish-based and CRAG gating only for strict
                if image_policy == "strict":
                    dish = _extract(answer) or _extract(original_query)
                    if dish and 'selected_docs_texts' in locals() and images:
                        filtered = [
                            url
                            for doc_text, url in zip(selected_docs_texts, images)
                            if url and isinstance(doc_text, str) and dish in doc_text
                        ]
                        if filtered:
                            images = filtered

                    final_verdict = None
                    if 'judge_verdict_2' in locals() and judge_verdict_2:
                        final_verdict = judge_verdict_2
                    elif 'judge_verdict_1' in locals() and judge_verdict_1:
                        final_verdict = judge_verdict_1

                    if ENABLE_CRAG and final_verdict and final_verdict != "grounded":
                        images = []
                    elif not dish:
                        images = []
    except Exception as e:
        if DEBUG_RAW:
            print(f"where hint: {e}")

    # CRAG verifier + corrective loop
    judge_verdict_1 = None
    judge_verdict_2 = None
    corrected = False
    final_pass = 1
    # Ensure verifier metrics always defined for response payload
    verifier_metrics_1 = {}
    verifier_metrics_2 = {}

    if ENABLE_CRAG and docs:
        # Prefer the exact texts used to build context for judging if available
        # CRAG 판정 후 처리
        judge_inputs = 'selected_docs_texts' in locals() and selected_docs_texts or docs
        judge_result_1 = relevance_check_node(answer, judge_inputs)
        judge_verdict_1 = judge_result_1.get("branch")
        verifier_metrics_1 = judge_result_1.get("metrics", {})
        # 문서 조회 속도
        support_rate = verifier_metrics_1.get("support_rate", 0)
        # 신뢰도 수준
        confidence_level = verifier_metrics_1.get("confidence_level", "unknown")
        pipeline_steps.append("judge1")

        # Not Sure 확장 (세분화)
        should_correct = (
            judge_verdict_1 == "notGrounded" 
            or (judge_verdict_1 == "notSure" and confidence_level in ("very_weak", "weak")) 
            or (judge_verdict_1 == "notSure" and support_rate < 0.30))
        correction_reason = ""

        if judge_verdict_1 == "notGrounded":
            rewritten2 = rewrite_node(original_query, context_text)
            pipeline_steps.append("rewrite2")

            retrieve_result2 = retrieve_node(rewritten2, req.k)
            pipeline_steps.append("retrieve2")

            docs2 = retrieve_result2.get("retrieved_docs", [])
            scores2 = retrieve_result2.get("retrieved_scores", [])
            images2 = retrieve_result2.get("retrieved_images", [])
            branch2 = retrieve_result2.get("branch", "no_docs")

            context_text2 = ""
            if docs2:
                try:
                    if (
                        scores2
                        and len(scores2) == len(docs2)
                        and SCORE_THRESHOLD
                        and SCORE_THRESHOLD > 0
                    ):
                        paired2 = [
                            (d, i, s)
                            for d, i, s in zip(docs2, images2, scores2)
                            if (s is not None and s >= SCORE_THRESHOLD)
                        ]
                        if paired2:
                            docs2, images2, scores2 = [list(x) for x in zip(*paired2)]
                except Exception as e:
                    if DEBUG_RAW:
                        print(f"where hint: {e}")

                context_text2, selected_images2, selected_docs_texts2 = build_context_with_images(
                    docs2, images2
                )
                images2 = selected_images2
                pipeline_steps.append("context_builder2")

            answer2 = generate_with_history(
                query=original_query,
                intent=intent,
                context=context_text2,
                conversation_history=conversation_history,
                model=req.model,
            )
            pipeline_steps.append("generate2")

            try:
                from nodes.generate_node_v2 import extract_target_dish as _extract
                # Apply policy again for pass2
                if not include_images:
                    images2 = []
                else:
                    allowed_image_intents = {"recipe", "dish_overview", "substitution", "storage"}
                    if image_policy in ("strict", "lenient") and intent not in allowed_image_intents:
                        images2 = []
                    else:
                        if image_policy == "strict":
                            dish2 = _extract(answer2) or _extract(original_query)
                            if dish2 and 'selected_docs_texts2' in locals() and images2:
                                filtered2 = [
                                    url
                                    for doc_text, url in zip(selected_docs_texts2, images2)
                                    if url and isinstance(doc_text, str) and dish2 in doc_text
                                ]
                                if filtered2:
                                    images2 = filtered2

                            final_verdict2 = None
                            if 'judge_verdict_2' in locals() and judge_verdict_2:
                                final_verdict2 = judge_verdict_2
                            elif 'judge_verdict_1' in locals() and judge_verdict_1:
                                final_verdict2 = judge_verdict_1

                            if ENABLE_CRAG and final_verdict2 and final_verdict2 != "grounded":
                                images2 = []
                            elif not dish2:
                                images2 = []
            except Exception as e:
                if DEBUG_RAW:
                    print(f"where hint: {e}")

            if docs2:
                judge_inputs2 = 'selected_docs_texts2' in locals() and selected_docs_texts2 or docs2
                judge_result_2 = relevance_check_node(answer2, judge_inputs2)
                judge_verdict_2 = judge_result_2.get("branch")
                verifier_metrics_2 = judge_result_2.get("metrics", {})
                pipeline_steps.append("judge2")

            # adopt pass 2
            answer = answer2
            docs = docs2
            scores = scores2
            images = images2
            branch = branch2
            context_text = context_text2
            corrected = True
            final_pass = 2

   # Low-confidence detection (after judge and retrieval)
    low_confidence = False
    try:
        numeric_scores = [s for s in (scores or []) if isinstance(s, (int, float))]
        max_sim = max(numeric_scores) if numeric_scores else None
        final_verdict = None

        # 세분화
        final_support_rate = 0
        final_confidence_level="unknown"

        if 'judge_verdict_2' in locals() and judge_verdict_2:
            final_verdict = judge_verdict_2
            final_support_rate = verifier_metrics_2.get("support_rate",0)
            final_confidence_level = verifier_metrics_2.get("confidence_level","unknown")
        elif 'judge_verdict_1' in locals() and judge_verdict_1:
            final_verdict = judge_verdict_1
            final_support_rate = verifier_metrics_1.get("support_rate",0)
            final_confidence_level = verifier_metrics_1.get("confidence_level","unknown")

        doc_count = len(docs or []) 
        low_sim = bool(SIMILARITY_THRESHOLD and (max_sim is None or max_sim < SIMILARITY_THRESHOLD))
        low_sim_slight = bool(SIMILARITY_THRESHOLD and (max_sim is None or max_sim < (SIMILARITY_THRESHOLD + 0.05)))
        is_not_grounded = (final_verdict == "notGrounded")
        is_not_sure = (final_verdict == "notSure")  # ✅ NEW: track notSure separately

        mode = (LOWCONF_MODE or "balanced").lower()
        if mode == "strict":
            crag_fail = (ENABLE_CRAG and final_verdict and final_verdict != "grounded")
            low_confidence = bool(low_sim or crag_fail)
        elif mode == "lenient":
            low_confidence = bool(doc_count < 1 or (low_sim and doc_count < 1))
        else:  # balanced (권장)
            crag_fail_bal = bool(ENABLE_CRAG and is_not_grounded and low_sim_slight)
            # Not Sure 세밀화
            notsure_weak =bool(
                ENABLE_CRAG and is_not_sure and (
                    final_support_rate < 0.30 
                    or final_confidence_level in ("very_weak", "weak")
                )
            )

            # ✅ NEW: notSure + borderline similarity도 저신뢰로 플래그
            notsure_and_borderline = bool(
                ENABLE_CRAG 
                and is_not_sure 
                and low_sim_slight 
                and doc_count < max(2, int(MIN_CONF_DOCS))
            )
            low_confidence = bool(
                (low_sim and doc_count < max(1, int(MIN_CONF_DOCS))) 
                or crag_fail_bal 
                or notsure_weak
                or notsure_and_borderline  # ✅ NEW
            )
    except Exception:
        low_confidence = False

    # If low confidence and user did not allow, switch to clarify guidance
    # Respect explicit decision override from pending state
    if low_confidence and not (getattr(req, 'allow_low_confidence', False) or allow_low_override):
        try:
            clarify_text = generate_with_history(
                query=original_query,
                intent="clarify",
                context="clarify_mode",
                conversation_history=conversation_history,
                model=req.model,
            )
            pipeline_steps.append("low_confidence_clarify")
            warning = (
                "검증 가능한 출처가 부족하거나 유사도가 낮아 부정확할 수 있어요.\n"
                "아래 중 하나를 선택해 주세요: 1) 그대로 진행(정확도 낮음 허용), 2) 질문 다듬기.\n"
                "질문 다듬기 제안:\n" + (clarify_text or "")
            )
            # Persist decision state in session metadata
            _set_pending_decision({
                "type": "low_confidence",
                "original_query": original_query,
            })
            # Replace answer with warning-only guidance; keep context for transparency
            answer = warning
        except Exception:
            warning = (
                "검증 가능한 출처가 부족하거나 유사도가 낮아 부정확할 수 있어요. "
                "그대로 진행하거나 질문을 더 구체화해 주세요."
            )
            _set_pending_decision({"type": "low_confidence", "original_query": original_query})

    # Sanitize links in answer body using collected sources (unknown URLs)
    sanitized_removed_urls: list[str] = []
    try:
        answer, sanitized_removed_urls = _sanitize_answer_links(answer, sources)
    except Exception:
        pass

    # Remove all links from answer body (UI renders sources separately)
    removed_body_urls: list[str] = []
    try:
        answer, removed_body_urls = _remove_links_in_body(answer)
    except Exception:
        pass

    # Remove any inline '출처' section from body
    stripped_sources = False
    try:
        answer, stripped_sources = _strip_sources_section(answer)
    except Exception:
        pass

    # Log conversation
    memory_manager.add_message(session_id, "user", original_query)
    memory_manager.add_message(
        session_id,
        "assistant",
        answer,
        {"intent": intent, "context_found": bool(docs), "used_docs": len(docs)},
    )

    # Build response
    response = {
        "answer": answer,
        "router": route,
        "intent": intent,
        "original_query": original_query,
        "rewritten_query": query_for_search if req.enable_rewrite else None,
        "context_text": context_text,  # ✅ NEW: Include actual context used for generation
        "context_len": len(context_text),
        "used_docs": len(docs),
        "context_found": bool(docs),
        "retrieved_count": len(docs),
        "retrieved_scores": scores[:5] if scores else [],
        "image_urls": [u for u in images if u][:max_images],
        "k": req.k,
        "mode": (
            "context_based"
            if docs
            else ("general_knowledge" if ALLOW_NO_CONTEXT_ANSWER else "no_context_refusal")
        ),
        "branch": branch,
        "pipeline": pipeline_steps,
        "session_id": session_id,
        "is_new_session": is_new_session,
        "history_used": len(conversation_history) > 0,
        "conversation_turns": len(conversation_history) // 2,
        "judge_verdict_1": judge_verdict_1 if 'judge_verdict_1' in locals() else None,
        "judge_verdict_2": judge_verdict_2 if 'judge_verdict_2' in locals() else None,
        "corrected": corrected if 'corrected' in locals() else False,
        "final_pass": final_pass if 'final_pass' in locals() else 1,
        "sources": sources,
        # link hygiene flags for observability
        "link_sanitized": bool(sanitized_removed_urls),
        "removed_urls": sanitized_removed_urls,
        "links_in_body_removed": bool(removed_body_urls or stripped_sources),
        "removed_body_urls": removed_body_urls,
        "stripped_sources_section": stripped_sources,
        # low-confidence handling
        "low_confidence": low_confidence,
        "warning": (
            "검증 가능한 출처가 부족하거나 유사도가 낮아 부정확할 수 있어요."
            if low_confidence else ""
        ),
        "decision_required": bool(low_confidence and not getattr(req, 'allow_low_confidence', False)),
        "suggested_actions": (
            ["proceed_with_low_confidence", "clarify"] if low_confidence else []
        ),
        "retrieval_metrics": {
            "score_mode": score_mode,
            "k": req.k,
            "mmr_enabled": RERANK_MMR,
            "mmr_fetch": MMR_FETCH,
            "mmr_lambda": MMR_LAMBDA,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "domain_cap": DOMAIN_CAP,
            "have_scores": any(s is not None for s in scores) if scores else False,
            "scores_summary": (lambda vals: {
                "count": len(vals),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "avg": (sum(vals)/len(vals)) if vals else None,
                "p50": (sorted(vals)[len(vals)//2] if vals else None),
                "p90": (sorted(vals)[int(0.9*(len(vals)-1))] if vals else None),
            })([s for s in scores if isinstance(s, (int, float))]),
            "unique_domains": len({(m.get("url") or "").split('/')[2] if (m.get("url") or "").startswith('http') else "" for m in metas}) if metas else 0,
            "verifier_metrics_1": verifier_metrics_1,
            "verifier_metrics_2": verifier_metrics_2,
        },
    }

    # Debug fields
    # local import to avoid unused in prod
    if DEBUG_RAW:
        response["context_preview"] = context_text[:300] if context_text else ""
        response["conversation_context"] = memory_manager.get_context_summary(session_id)

    return response
