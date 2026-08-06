"""
Generate a PNG visualization of the current agent graph.
Usage: python -m graph.generate_graph_viz
Output: agent_graph.png
"""

from graph.main import Graph

def main():
    # Build the graph (no checkpointer needed for visualization)
    app = Graph().compileX()

    # Generate PNG bytes and write to file
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("agent_graph.png", "wb") as f:
        f.write(png_bytes)

    print("Graph visualization saved to agent_graph.png")

if __name__ == "__main__":
    main()