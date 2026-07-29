from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from services.app_logger import get_app_logger
from agent_loop import AgentLoop


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="The user's message for the agent")

class QueryResponse(BaseModel):
    response: str
    statuses: list[str]
    image_links: list[str]

app = FastAPI(title="Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_loop = AgentLoop()
logger = get_app_logger()


async def send_status(websocket: WebSocket, status: str):
    """Forward a graph status update to the connected browser."""
    await websocket.send_json({"type": "status", "status": status})


def create_initial_state(query: str, convo_memory: str = "", history: list | None = None):
    messages = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if item.get("role") in ("user", "Human"):
            messages.append(HumanMessage(content=content))
        elif item.get("role") in ("assistant", "AI"):
            messages.append(AIMessage(content=content))

    if not messages or messages[-1].type != "human" or messages[-1].content != query:
        messages.append(HumanMessage(content=query))

    return {
        "messages": messages,
        "search_required": [False],
        "search_results": None,
        "search_queries": [],
        "image_links": [],
        "status_messages": ["generating"],
        "convo_memory": convo_memory,
        "final_response": [],
        "conversation_title": "",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    state = agent_loop.run(create_initial_state(request.query))
    print("after running query", state)
    return QueryResponse(
        response=state["final_response"][-1],
        statuses=state["status_messages"],
        image_links=state["image_links"],
    )

@app.websocket("/ws/query")
async def query_agent_stream(websocket: WebSocket):
    """Receive {"query": "...", "thread_id": "..."} and send JSON status/response messages.

    thread_id is required — it is generated on the frontend (crypto.randomUUID())
    and used to maintain per-thread conversation state via MemorySaver.
    """
    await websocket.accept()
    logger.info("WebSocket connected")
    await websocket.send_json({"type": "connection", "status": "connected"})
    disconnected = False
    try:
        data = await websocket.receive_json()
        query = data.get("query", "").strip()
        thread_id = data.get("thread_id")
        convo_memory = data.get("summary", "")
        history = data.get("history", [])
        if not isinstance(convo_memory, str):
            convo_memory = ""
        if not isinstance(history, list):
            history = []

        if not query:
            await websocket.send_json({"type": "error", "detail": "Query is required."})
            return

        restore_history = not agent_loop.has_memory(thread_id)
        state_history = history if restore_history else []
        logger.info(
            "WebSocket query received: %s (thread=%s, history_messages=%d, restoring=%s)",
            query,
            thread_id,
            len(history),
            restore_history,
        )

        if restore_history and history:
            await send_status(websocket, "restoring chat")

        await send_status(websocket, "generating")

        title_suggestion = None
        for update in agent_loop.stream(
            create_initial_state(query, convo_memory, state_history), thread_id=thread_id
        ):
            logger.info("Graph update received: %s", list(update))
            node_update = next(iter(update.values()))

            for status in node_update.get("status_messages", []):
                await send_status(websocket, status)

            if image_links := node_update.get("image_links"):
                await websocket.send_json({"type": "images", "image_links": image_links})

            for response in node_update.get("final_response", []):
                await websocket.send_json({"type": "response", "response": response})

            if summary := node_update.get("convo_memory"):
                logger.info("Sending generated summary for thread=%s", thread_id)
                await websocket.send_json({"type": "summary", "summary": summary})

            # Capture title suggestion from the title gen node
            if node_update.get("conversation_title"):
                title_suggestion = node_update["conversation_title"]

        # Send the title suggestion as a final message
        if title_suggestion:
            await websocket.send_json({
                "type": "title_suggestion",
                "title": title_suggestion
            })

    except ValidationError as error:
        await websocket.send_json({"type": "error", "detail": error.errors()})
    except WebSocketDisconnect:
        disconnected = True
    except Exception:
        logger.exception("Agent execution failed")
        await websocket.send_json(
            {"type": "error", "detail": "The agent could not complete the request."}
        )
    finally:
        if not disconnected:
            await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
