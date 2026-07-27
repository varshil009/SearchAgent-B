from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
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


def create_initial_state(query: str):
    return {
        "messages": [HumanMessage(content=query)],
        "search_required": [False],
        "search_results": None,
        "search_queries": [],
        "image_links": [],
        "status_messages": ["generating"],
        "convo_memory": "",
        "final_response": [],
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
    """Receive {"query": "..."} and send JSON status/response messages.

    This endpoint is stateless: it does not authenticate users or persist
    messages and threads.
    """
    await websocket.accept()
    logger.info("WebSocket connected")
    disconnected = False
    try:
        request = QueryRequest(**await websocket.receive_json())
        query = request.query
        logger.info("WebSocket query received: %s", query)

        await websocket.send_json({"type": "status", "status": "generating"})

        for update in agent_loop.stream(create_initial_state(query)):
            logger.info("Graph update received: %s", list(update))
            node_update = next(iter(update.values()))

            for status in node_update.get("status_messages", []):
                await websocket.send_json({"type": "status", "status": status})

            if image_links := node_update.get("image_links"):
                await websocket.send_json({"type": "images", "image_links": image_links})

            for response in node_update.get("final_response", []):
                await websocket.send_json({"type": "response", "response": response})

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

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
