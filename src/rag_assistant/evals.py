from __future__ import annotations

from typing import Any, Dict, Iterable, List

from rag_assistant.knowledge_base import KnowledgeBase


def run_retrieval_eval(kb: KnowledgeBase, cases: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    total_hits = 0
    total_reciprocal_rank = 0.0

    for case in cases:
        expected_source_ids = [str(item).strip() for item in case.get("expected_source_ids", []) if str(item).strip()]
        chunks, trace = kb.query_with_trace(
            question=case["question"],
            top_k=int(case.get("top_k", 5)),
            min_score=float(case.get("min_score", 0.0)),
            retrieval=str(case.get("retrieval", "hybrid")),
            embedding_model=case.get("embedding_model"),
            embedding_provider=str(case.get("embedding_provider", "local")),
            local_dimensions=int(case.get("local_dimensions", 256)),
            reranker=str(case.get("reranker", "term_overlap")),
            candidate_pool_size=int(case.get("candidate_pool_size", 10)),
        )

        returned_source_ids: List[str] = []
        seen_sources: set[str] = set()
        for chunk in chunks:
            if chunk.source_id in seen_sources:
                continue
            returned_source_ids.append(chunk.source_id)
            seen_sources.add(chunk.source_id)

        match_rank = None
        expected = set(expected_source_ids)
        for index, source_id in enumerate(returned_source_ids, start=1):
            if source_id in expected:
                match_rank = index
                break

        matched = match_rank is not None
        reciprocal_rank = round(1.0 / match_rank, 4) if match_rank is not None else 0.0
        total_hits += int(matched)
        total_reciprocal_rank += reciprocal_rank

        results.append(
            {
                "question": case["question"],
                "expected_source_ids": expected_source_ids,
                "returned_source_ids": returned_source_ids,
                "matched": matched,
                "reciprocal_rank": reciprocal_rank,
                "top_hit_source_id": returned_source_ids[0] if returned_source_ids else None,
                "trace": trace,
            }
        )

    case_count = len(results)
    if case_count == 0:
        return {"cases": 0, "hits": 0, "hit_rate": 0.0, "mrr": 0.0, "results": []}

    return {
        "cases": case_count,
        "hits": total_hits,
        "hit_rate": round(total_hits / case_count, 4),
        "mrr": round(total_reciprocal_rank / case_count, 4),
        "results": results,
    }
