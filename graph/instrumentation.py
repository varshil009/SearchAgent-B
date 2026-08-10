try:
    import phoenix as px
    from phoenix.otel import register
except ImportError:
    tracer_provider = None
else:
    px.launch_app()
    tracer_provider = register(
        project_name="local-search-chatbot",
        auto_instrument=True
    )
