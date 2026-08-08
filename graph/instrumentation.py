import phoenix as px
from phoenix.otel import register

px.launch_app()
tracer_provider = register(
    project_name="local-search-chatbot",
    auto_instrument=True
)