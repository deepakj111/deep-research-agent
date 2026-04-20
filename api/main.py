import json
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from agent.graph import graph

app = FastAPI(title="DeepResearch Agent API")


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.post("/research/stream")
async def stream_research(query: str, profile: str = "fast"):
    run_id = str(uuid.uuid4())

    async def event_generator():
        async for event in graph.astream_events(
            {
                "query": query,
                "profile": profile,
                "run_id": run_id,
                "findings": [],
                "error_log": [],
                "thought_log": [],
            },
            version="v2",
            config={"configurable": {"thread_id": run_id}},
        ):
            kind = event["event"]
            if kind == "on_chain_start":
                yield sse({"type": "node_start", "node": event["name"]})
            elif kind == "on_tool_start":
                yield sse(
                    {
                        "type": "tool_call",
                        "tool": event["name"],
                        "input": str(event.get("data", {}).get("input", ""))[:200],
                    }
                )
            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output", [])
                yield sse(
                    {
                        "type": "tool_result",
                        "tool": event["name"],
                        "count": len(output) if isinstance(output, list) else 1,
                    }
                )
            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield sse({"type": "token", "content": chunk.content})
            elif kind == "on_chain_end" and event["name"] == "writer":
                yield sse({"type": "complete", "run_id": run_id})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "deep-research-agent-api"}
