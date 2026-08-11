# The project-wide UV environment targets Python 3.14.
FROM python:3.14-slim

WORKDIR /app

# Install the application dependencies used by SearchAgent-B.  Versions are
# aligned with the root UV project's declared dependency set.
RUN pip install --no-cache-dir \
    "fastapi[standard]>=0.139.0" \
    "langchain>=1.3.14" \
    "langgraph>=1.2.7" \
    "groq>=1.5.0" \
    "exa-py>=2.16.0" \
    "pydantic>=2.13.4" \
    "python-dotenv>=1.1.0" \
    "supabase>=2.0.0" \
    "uvicorn[standard]>=0.50.0" \
    "requests>=2.31.0"

COPY . .

# The defaults in main.py are appropriate for local development, not Docker.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    RELOAD=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Run from /app, which is the SearchAgent-B project root.
CMD ["python", "main.py"]
