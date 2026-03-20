# rag-vision-telegram-bot

Base code for the assignment: a Telegram bot that supports RAG question answering and image captioning.

Features
- /ask for RAG queries over local docs
- /image for image captioning
- /help for usage
- Optional history, caching, and source snippets

Project layout
- app.py: Telegram bot entrypoint
- config.py: environment-driven settings
- rag.py: document indexing and retrieval
- vision.py: image captioning
- llm.py: OpenAI API client
- data/docs: sample documents

Setup
1. Create a Telegram bot token and set TELEGRAM_BOT_TOKEN.
2. Create an OpenAI API key and set OPENAI_API_KEY.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a .env file using .env.example as a template.
5. (Optional) Set OPENAI_MODEL if you want to use a different model.
6. Run the bot:

```bash
python app.py
```

RAG notes
- Add .md or .txt files to data/docs.
- The index is stored at data/index.sqlite and auto-built on first run.

Vision notes
- The first image request will download the BLIP model if not cached.
- Change VISION_MODEL in .env to use a different caption model.

Commands
- /ask <query>
- /image
- /help
