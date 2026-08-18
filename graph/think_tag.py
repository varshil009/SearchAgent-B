import re
def _strip_think_tags(text: str) -> str:
    """Remove everything between think tags (including the tags)."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()