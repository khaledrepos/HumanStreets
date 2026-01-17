
import base64

diagram = """graph TD
    User[User] -->|Interacts| Frontend[React Frontend]

    subgraph FrontEnd ["Frontend Logic"]
        App[App.jsx] -->|Manages State| ChatSidebar
        App -->|Renders| MapContainer
        ChatSidebar[ChatSidebar Component] -->|Displays| ChatUI
        MapContainer[MapContainer Component] -->|Renders| DeckGL/MapLibre
    end

    Frontend -->|HTTP POST /chat| Backend[FastAPI Backend]
    Frontend -->|HTTP GET /health| Backend

    subgraph BackEnd ["Backend Logic"]
        Backend -->|Loads| Model[LiquidAI/LFM2-1.2B]
        Backend -->|Tokenizes| Tokenizer
        Model -->|Streams Tokens| Streamer[TextIteratorStreamer]
        Streamer -->|Yields Chunks| Backend
    end

    Backend -.->|SSE-like Stream| Frontend
"""

encoded = base64.b64encode(diagram.encode('utf-8')).decode('ascii')
url = f"https://mermaid.ink/img/{encoded}"

with open("mermaid_link.txt", "w", encoding="utf-8") as f:
    f.write(url)
