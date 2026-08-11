"""
try:
    import phoenix as px
    import os
    from phoenix.otel import register
except ImportError:
    tracer_provider = None
else:
    if os.getenv("USE_PHOENIX") is None:
        px.launch_app()
    tracer_provider = register(
        project_name="local-search-chatbot",
        auto_instrument=True
    )
"""
