from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Deque

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from config import load_settings
from llm import LLMError
from rag import RAGPipeline
from utils import ensure_dir
from vision import VisionCaptioner, VisionError


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rag-vision-bot")

settings = load_settings()

rag_pipeline: RAGPipeline | None = None
vision_captioner: VisionCaptioner | None = None

USER_HISTORY: dict[int, Deque[tuple[str, str]]] = {}


def _mode_allows(target: str) -> bool:
    return settings.mode in {"hybrid", target}


def _get_history(user_id: int) -> list[tuple[str, str]]:
    if not settings.enable_history:
        return []
    return list(USER_HISTORY.get(user_id, []))


def _add_history(user_id: int, question: str, answer: str) -> None:
    if not settings.enable_history:
        return
    history = USER_HISTORY.setdefault(user_id, deque(maxlen=settings.history_len))
    history.append((question, answer))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Commands:\n"
        "/ask <query> - ask a question (RAG)\n"
        "/image - send an image for captioning\n"
        "/help - show this message"
    )
    if update.message:
        await update.message.reply_text(text)


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _mode_allows("rag"):
        if update.message:
            await update.message.reply_text("RAG mode is disabled. Set MODE=rag or MODE=hybrid.")
        return

    if rag_pipeline is None:
        if update.message:
            await update.message.reply_text("RAG pipeline is not available.")
        return

    query = " ".join(context.args).strip()
    if not query and update.message and update.message.text:
        query = update.message.text.partition(" ")[2].strip()

    if not query:
        if update.message:
            await update.message.reply_text("Usage: /ask <your question>")
        return

    user_id = update.effective_user.id if update.effective_user else 0
    history = _get_history(user_id)

    try:
        answer, sources = rag_pipeline.answer(query, history)
    except (LLMError, RuntimeError) as exc:
        logger.exception("RAG error")
        if update.message:
            await update.message.reply_text(f"RAG error: {exc}")
        return

    if update.message:
        await update.message.reply_text(answer)
        if sources:
            await update.message.reply_text("Sources:\n" + "\n".join(sources))

    _add_history(user_id, query, answer)


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.photo:
        await _handle_photo(update, context)
        return

    if update.message:
        await update.message.reply_text("Send an image after /image, or just send an image directly.")


async def photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_photo(update, context)


async def _handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _mode_allows("vision"):
        if update.message:
            await update.message.reply_text("Vision mode is disabled. Set MODE=vision or MODE=hybrid.")
        return

    if vision_captioner is None:
        if update.message:
            await update.message.reply_text("Vision model is not available.")
        return

    message = update.message
    if message is None:
        return

    file = None
    file_suffix = "jpg"

    if message.photo:
        photo = message.photo[-1]
        file = await photo.get_file()
    elif message.document and message.document.mime_type:
        if message.document.mime_type.startswith("image/"):
            file = await message.document.get_file()
            file_suffix = message.document.mime_type.split("/")[-1]

    if file is None:
        await message.reply_text("No image found. Please send a photo.")
        return

    ensure_dir(settings.image_dir)
    image_path = settings.image_dir / f"{file.file_unique_id}.{file_suffix}"
    await file.download_to_drive(custom_path=str(image_path))

    try:
        caption, tags = vision_captioner.caption(image_path)
    except VisionError as exc:
        logger.exception("Vision error")
        await message.reply_text(f"Vision error: {exc}")
        return

    tags_text = ", ".join(tags) if tags else "(none)"
    await message.reply_text(f"Caption: {caption}\nTags: {tags_text}")


def main() -> None:
    if not settings.telegram_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    if settings.mode not in {"rag", "vision", "hybrid"}:
        raise RuntimeError("MODE must be rag, vision, or hybrid")

    global rag_pipeline, vision_captioner
    if _mode_allows("rag"):
        rag_pipeline = RAGPipeline(settings)

    if _mode_allows("vision"):
        vision_captioner = VisionCaptioner(settings.vision_model)

    application = ApplicationBuilder().token(settings.telegram_token).build()

    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("image", image_command))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, photo_message))

    logger.info("Bot starting in %s mode", settings.mode)
    application.run_polling()


if __name__ == "__main__":
    main()
