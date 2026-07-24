from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError

from agent_loop import AgentLoop


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="The user's message for the agent")


class QueryResponse(BaseModel):
    response: str
    statuses: list[str]


app = FastAPI(title="Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_loop = AgentLoop()


def create_initial_state(query: str):
    return {
        "messages": [HumanMessage(content=query)],
        "search_required": [False],
        "search_results": None,
        "search_queries": [],
        "status_messages": ["generating"],
        "final_response": [],
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    state = agent_loop.run(create_initial_state(request.query))
    return QueryResponse(
        response=state["final_response"][-1],
        statuses=state["status_messages"],
    )


@app.websocket("/ws/query")
async def query_agent_stream(websocket: WebSocket):
    """Receive {"query": "..."} and send JSON status/response messages."""
    await websocket.accept()
    disconnected = False

    try:
        request = QueryRequest(**await websocket.receive_json())
        await websocket.send_json({"type": "status", "status": "generating"})

        for update in agent_loop.stream(create_initial_state(request.query)):
            node_update = next(iter(update.values()))

            for status in node_update.get("status_messages", []):
                await websocket.send_json({"type": "status", "status": status})

            for response in node_update.get("final_response", []):
                await websocket.send_json({"type": "response", "response": response})

    except ValidationError as error:
        await websocket.send_json({"type": "error", "detail": error.errors()})
    except WebSocketDisconnect:
        disconnected = True
    finally:
        if not disconnected:
            await websocket.close()
