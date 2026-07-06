"""Telegram bot handlers for video processing."""
import asyncio
import logging
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, TypeVar
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ChatType
from telegram.ext import ContextTypes, CallbackQueryHandler, MessageHandler, filters
from telegram.error import NetworkError, TimedOut

from bot.temp_manager import TempManager
from bot.video_processor import VideoProcessor
from bot.video_merger import VideoAudioMerger
from bot.format_processor import FormatConverter, AudioExtractor
from bot.split_processor import VideoSplitter
from bot.join_processor import VideoJoiner
from bot.error_handler import (
    DownloadError,
    FFmpegError,
    ProcessingTimeoutError,
    FormatConversionError,
    AudioExtractionError,
    VideoSplitError,
    VideoJoinError,
    VoiceConversionError,
    VoiceToMp3Error,
    AudioSplitError,
    AudioJoinError,
    AudioFormatConversionError,
    AudioEnhancementError,
    AudioEffectsError,
    VideoMergeError,
    ImageProcessingError,
    ImageCompressionError,
    ImageConversionError,
    ImageResizeError,
    ImageEnhancementError,
    ImageNoiseError,
    handle_processing_error,
    get_user_error_message,
    DEFAULT_ERROR_MESSAGE,
)
from bot.config import config
from bot.validators import (
    validate_file_size,
    validate_video_file,
    validate_audio_file,
    check_disk_space,
    estimate_required_space,
    ValidationError,
)
from bot.audio_processor import VoiceNoteConverter, VoiceToMp3Converter, get_audio_duration
from bot.audio_splitter import AudioSplitter
from bot.audio_joiner import AudioJoiner
from bot.audio_format_converter import AudioFormatConverter, detect_audio_format, get_supported_audio_formats
from bot.audio_enhancer import AudioEnhancer
from bot.audio_effects import AudioEffects
from bot.screenshot_processor import ScreenshotProcessor
from bot.image_processor import (
    ImageProcessor,
    SUPPORTED_IMAGE_FORMATS,
    ENHANCEMENT_PROFILES,
    NOISE_STRENGTH_LEVELS,
)

# Import downloaders for URL handling
from bot.downloaders import (
    DownloadFacade,
    download_url,
    URLDetector,
    URLType,
    is_youtube_url,
)
from bot.downloaders.exceptions import (
    DownloadError,
    FileTooLargeError,
    URLValidationError,
    UnsupportedURLError,
)

# Internal IG inter-download delay helpers (used by the three _start_* paths;
# hoisted to avoid repeated local imports on hot paths; _apply/_mark are private).
from bot.downloaders.platforms.instagram import (
    _apply_instagram_delay,
    _mark_instagram_download_complete,
    is_instagram_url,
)

logger = logging.getLogger(__name__)

# URL detector instance for detecting URLs in messages
url_detector = URLDetector()

# Audio file extensions accepted when sent as Telegram documents
AUDIO_DOCUMENT_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}


def _is_audio_document(document) -> bool:
    """Return True if a Telegram document attachment is an audio file."""
    if not document:
        return False
    mime_type = (document.mime_type or "").lower()
    if mime_type.startswith("audio/"):
        return True
    filename = (document.file_name or "").lower()
    return Path(filename).suffix in AUDIO_DOCUMENT_EXTENSIONS


def _get_message_audio_source(message) -> tuple[str | None, int | None, str | None]:
    """Extract audio file_id, file_size, and unique_id from a message.

    Supports native Telegram audio messages and audio sent as documents.
    """
    if message.audio:
        audio = message.audio
        return audio.file_id, audio.file_size, audio.file_unique_id
    document = message.document
    if document and _is_audio_document(document):
        return document.file_id, document.file_size, document.file_unique_id
    return None, None, None


async def _download_with_retry(file, destination_path: str, max_retries: int = 3, correlation_id: str = None) -> bool:
    """Download file with retry logic for transient failures.

    With local Bot API, copies from a shared filesystem when available and
    otherwise downloads via the configured local file endpoint.

    Args:
        file: Telegram file object to download
        destination_path: Path to save the file
        max_retries: Maximum number of retry attempts
        correlation_id: Optional correlation ID for request tracing

    Returns:
        True if download succeeded

    Raises:
        NetworkError, TimedOut: If all retries exhausted
    """
    import shutil

    cid = correlation_id or "no-cid"

    if file.file_path and os.path.isfile(file.file_path):
        shutil.copy2(file.file_path, destination_path)
        logger.info(f"[{cid}] File copied from shared path to {destination_path}")
        return True

    for attempt in range(max_retries):
        try:
            await file.download_to_drive(destination_path)
            logger.info(f"[{cid}] File downloaded to {destination_path}")
            return True
        except (NetworkError, TimedOut) as e:
            if attempt < max_retries - 1:
                logger.warning(f"[{cid}] Download attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"[{cid}] Download failed after {max_retries} attempts: {e}")
                raise
    return False


_T = TypeVar("_T")


async def _send_with_retry(
    send_callable: Callable[[], Awaitable[_T]],
    *,
    max_retries: int = 3,
    correlation_id: str | None = None,
    label: str = "send",
) -> _T:
    """Run a Telegram send operation with retry on TimedOut.

    The callable must be safe to call multiple times (re-open file handles
    on each invocation).
    """
    cid = correlation_id or "no-cid"

    for attempt in range(max_retries):
        try:
            return await send_callable()
        except TimedOut as e:
            if attempt < max_retries - 1:
                delay = 2 * (attempt + 1)
                logger.warning(
                    f"[{cid}] {label} timed out (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"[{cid}] {label} timed out after {max_retries} attempts: {e}")
                raise

    raise RuntimeError("unreachable")


async def _process_video_with_timeout(
    update: Update,
    temp_mgr: TempManager,
    user_id: int,
    correlation_id: str = None
) -> None:
    """Process video with timeout handling.

    Internal function that handles the actual video processing
    with timeout and proper error handling.

    Args:
        update: Telegram update object
        temp_mgr: TempManager instance for file handling
        user_id: ID of the user sending the video
        correlation_id: Optional correlation ID for request tracing

    Raises:
        DownloadError: If video download fails
        FFmpegError: If video processing fails
        ProcessingTimeoutError: If processing times out
    """
    cid = correlation_id or "no-cid"
    # Get video from message
    video = update.message.video

    # Generate safe filenames
    input_filename = f"input_{user_id}_{video.file_unique_id}.mp4"
    output_filename = f"output_{user_id}_{video.file_unique_id}.mp4"

    input_path = temp_mgr.get_temp_path(input_filename)
    output_path = temp_mgr.get_temp_path(output_filename)

    # Download video to temp file
    logger.info(f"[{cid}] Downloading video from user {user_id}")
    try:
        file = await video.get_file()
        await _download_with_retry(file, input_path, correlation_id=cid)
    except Exception as e:
        logger.error(f"[{cid}] Failed to download video for user {user_id}: {e}")
        raise DownloadError("No pude descargar el video") from e

    # Validate video integrity after download
    is_valid, error_msg = validate_video_file(str(input_path))
    if not is_valid:
        logger.warning(f"[{cid}] Video validation failed for user {user_id}: {error_msg}")
        raise ValidationError(error_msg)

    # Check disk space before processing
    video_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
    required_space = estimate_required_space(int(video_size_mb))
    has_space, space_error = check_disk_space(required_space)
    if not has_space:
        logger.warning(f"[{cid}] Disk space check failed for user {user_id}: {space_error}")
        raise ValidationError(space_error)

    # Process video with timeout
    logger.info(f"[{cid}] Processing video for user {user_id}")
    logger.debug(f"[{cid}] Processing with timeout: {config.PROCESSING_TIMEOUT}s")
    try:
        # Use asyncio.wait_for to enforce timeout
        loop = asyncio.get_event_loop()
        success = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                VideoProcessor.process_video,
                str(input_path),
                str(output_path)
            ),
            timeout=config.PROCESSING_TIMEOUT
        )

        if not success:
            logger.error(f"[{cid}] Video processing failed for user {user_id}")
            raise FFmpegError("El procesamiento de video falló")

    except asyncio.TimeoutError as e:
        logger.error(f"[{cid}] Video processing timed out for user {user_id}")
        raise ProcessingTimeoutError("El video tardó demasiado en procesarse") from e

    # Send as video note
    logger.info(f"[{cid}] Sending video note to user {user_id}")
    try:
        with open(output_path, "rb") as video_file:
            await update.message.reply_video_note(video_note=video_file)
        logger.info(f"[{cid}] Video note sent successfully to user {user_id}")
    except Exception as e:
        logger.error(f"[{cid}] Failed to send video note to user {user_id}: {e}")
        raise


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle video messages by showing an inline menu with available actions.

    When a user sends a video, displays an inline keyboard with options:
    - Nota de Video: Convert to circular video note
    - Extraer Audio: Extract audio from video
    - Convertir Formato: Convert video to different format
    - Dividir Video: Split video into segments

    If there's an active video join session, routes to handle_join_video instead.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Video received from user {user_id}")

    # Check if there's an active video join session
    if context.user_data.get("join_session"):
        logger.info(f"[{correlation_id}] Routing video to join session handler for user {user_id}")
        await handle_join_video(update, context)
        return

    # Validate file size before showing menu
    video = update.message.video
    if video.file_size:
        logger.debug(f"[{correlation_id}] Video file size: {video.file_size} bytes")
        is_valid, error_msg = validate_file_size(video.file_size, config.max_incoming_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Store file info for callback handler
    context.user_data["video_menu_file_id"] = video.file_id
    context.user_data["video_menu_correlation_id"] = correlation_id

    # Show inline menu
    reply_markup = _get_video_menu_keyboard()
    await update.message.reply_text(
        "Video recibido. Selecciona una acción:",
        reply_markup=reply_markup
    )
    logger.info(f"[{correlation_id}] Video menu displayed to user {user_id}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    await update.message.reply_text(
        "¡Hola! Envíame un video, audio, o enlace de video y te mostraré opciones de procesamiento.\n\n"
        "📥 Descargas desde plataformas:\n"
        "/download <url> - Descargar video/audio de YouTube, Instagram, TikTok, Twitter/X, Facebook\n"
        "/downloads - Ver descargas activas y recientes\n"
        "También puedes enviarme directamente un enlace de video\n\n"
        "🎬 Procesamiento de video:\n"
        "/convert <formato> - Convierte un video a otro formato (mp4, avi, mov, mkv, webm)\n"
        "/extract_audio <formato> - Extrae el audio de un video (mp3, aac, wav, ogg)\n"
        "/split [duration|parts] <valor> - Divide un video en segmentos\n"
        "/join - Une múltiples videos en uno solo\n\n"
        "🎵 Procesamiento de audio:\n"
        "/split_audio [duration|parts] <valor> - Divide un audio en segmentos\n"
        "/join_audio - Une múltiples archivos de audio\n"
        "/convert_audio - Convierte un audio a otro formato (MP3, WAV, OGG, AAC, FLAC)\n"
        "/bass_boost - Aumenta los bajos del audio (intensidad ajustable)\n"
        "/treble_boost - Aumenta los agudos del audio (intensidad ajustable)\n"
        "/equalize - Ecualizador de 3 bandas (bass, mid, treble)\n"
        "/denoise - Reduce el ruido de fondo del audio (intensidad ajustable)\n"
        "/compress - Comprime el rango dinámico del audio (nivel ajustable)\n"
        "/normalize - Normaliza el volumen del audio (EBU R128)\n"
        "/effects - Aplica múltiples efectos en cadena (pipeline)\n\n"
        "💡 También puedes usar los menús inline:\n"
        "- Envía un video → Menú con opciones (Nota de Video, Extraer Audio, Merge con Audio, etc.)\n"
        "- Envía un audio → Menú con opciones (Nota de Voz, Dividir Audio, Unir Audios, etc.)\n"
        "- Envía una foto o imagen → Menú con opciones (Comprimir, Convertir, Redimensionar, Naturalizar, Info)\n"
        "- Envía un enlace de video → Menú de descarga con opciones combinadas"
    )


async def _get_video_from_message(update: Update) -> tuple:
    """Extract video object and file info from message.

    Args:
        update: Telegram update object

    Returns:
        Tuple of (video_object, is_reply) or (None, False) if no video found
    """
    # Check if the message itself has a video
    if update.message.video:
        return update.message.video, False

    # Check if it's a reply to a video
    if update.message.reply_to_message and update.message.reply_to_message.video:
        return update.message.reply_to_message.video, True

    return None, False


async def _get_audio_from_message(update: Update) -> tuple:
    """Extract audio object and file info from message.

    Args:
        update: Telegram update object

    Returns:
        Tuple of (audio_object, is_reply) or (None, False) if no audio found
    """
    # Check if the message itself has audio
    if update.message.audio:
        return update.message.audio, False

    # Check if it's a reply to an audio message
    if update.message.reply_to_message and update.message.reply_to_message.audio:
        return update.message.reply_to_message.audio, True

    return None, False


async def handle_convert_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /convert command to convert video to different format.

    Usage: /convert <formato> (when replying to a video or with video attached)

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    logger.info(f"Convert command received from user {user_id}")

    # Get video from message or reply
    video, is_reply = await _get_video_from_message(update)

    if not video:
        await update.message.reply_text(
            "Por favor envía un video o responde a un video con /convert <formato>\n"
            "Formatos soportados: mp4, avi, mov, mkv, webm"
        )
        return

    # Validate file size before downloading
    if video.file_size:
        is_valid, error_msg = validate_file_size(video.file_size, config.max_incoming_file_size_mb)
        if not is_valid:
            logger.warning(f"File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Parse format from command arguments
    args = context.args
    if not args:
        await update.message.reply_text(
            "Por favor especifica el formato de salida.\n"
            "Ejemplo: /convert mov\n"
            "Formatos soportados: mp4, avi, mov, mkv, webm"
        )
        return

    output_format = args[0].lower().lstrip(".")
    supported_formats = FormatConverter.get_supported_formats()

    if output_format not in supported_formats:
        await update.message.reply_text(
            f"Formato no soportado: {output_format}\n"
            f"Formatos soportados: {', '.join(supported_formats)}"
        )
        return

    # Send processing message
    processing_message = None
    try:
        processing_message = await update.message.reply_text(
            f"Convirtiendo video a {output_format.upper()}..."
        )
    except Exception as e:
        logger.warning(f"Could not send processing message to user {user_id}: {e}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_{user_id}_{video.file_unique_id}.mp4"
            output_filename = f"output_{user_id}_{video.file_unique_id}.{output_format}"

            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download video
            logger.info(f"Downloading video from user {user_id} for format conversion")
            try:
                file = await video.get_file()
                await _download_with_retry(file, input_path)
                logger.info(f"Video downloaded to {input_path}")
            except Exception as e:
                logger.error(f"Failed to download video for user {user_id}: {e}")
                raise DownloadError("No pude descargar el video") from e

            # Validate video integrity after download
            is_valid, error_msg = validate_video_file(str(input_path))
            if not is_valid:
                logger.warning(f"Video validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space before processing
            video_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(video_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Convert video with timeout
            logger.info(f"Converting video to {output_format} for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                converter = FormatConverter(str(input_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, converter.convert, output_format),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not success:
                    logger.error(f"Format conversion failed for user {user_id}")
                    raise FormatConversionError(f"No pude convertir el video a {output_format.upper()}")

            except asyncio.TimeoutError as e:
                logger.error(f"Format conversion timed out for user {user_id}")
                raise ProcessingTimeoutError("La conversión tardó demasiado") from e

            # Send converted video
            logger.info(f"Sending converted video to user {user_id}")
            try:
                with open(output_path, "rb") as video_file:
                    await update.message.reply_video(video=video_file)
                logger.info(f"Converted video sent successfully to user {user_id}")
            except Exception as e:
                logger.error(f"Failed to send converted video to user {user_id}: {e}")
                raise

            # Delete processing message on success
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")

        except (DownloadError, FormatConversionError, ProcessingTimeoutError, ValidationError) as e:
            await handle_processing_error(update, e, user_id)
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")

        except Exception as e:
            logger.exception(f"Unexpected error converting video for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")


async def handle_extract_audio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /extract_audio command to extract audio from video.

    Usage: /extract_audio <formato> (when replying to a video or with video attached)

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    logger.info(f"Extract audio command received from user {user_id}")

    # Get video from message or reply
    video, is_reply = await _get_video_from_message(update)

    if not video:
        await update.message.reply_text(
            "Por favor envía un video o responde a un video con /extract_audio <formato>\n"
            "Formatos soportados: mp3, aac, wav, ogg"
        )
        return

    # Validate file size before downloading
    if video.file_size:
        is_valid, error_msg = validate_file_size(video.file_size, config.max_incoming_file_size_mb)
        if not is_valid:
            logger.warning(f"File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Parse format from command arguments (default to mp3)
    args = context.args
    output_format = args[0].lower().lstrip(".") if args else "mp3"
    supported_formats = AudioExtractor.get_supported_formats()

    if output_format not in supported_formats:
        await update.message.reply_text(
            f"Formato no soportado: {output_format}\n"
            f"Formatos soportados: {', '.join(supported_formats)}"
        )
        return

    # Send processing message
    processing_message = None
    try:
        processing_message = await update.message.reply_text(
            f"Extrayendo audio como {output_format.upper()}..."
        )
    except Exception as e:
        logger.warning(f"Could not send processing message to user {user_id}: {e}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_{user_id}_{video.file_unique_id}.mp4"
            output_filename = f"audio_{user_id}_{video.file_unique_id}.{output_format}"

            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download video
            logger.info(f"Downloading video from user {user_id} for audio extraction")
            try:
                file = await video.get_file()
                await _download_with_retry(file, input_path)
                logger.info(f"Video downloaded to {input_path}")
            except Exception as e:
                logger.error(f"Failed to download video for user {user_id}: {e}")
                raise DownloadError("No pude descargar el video") from e

            # Validate video integrity after download
            is_valid, error_msg = validate_video_file(str(input_path))
            if not is_valid:
                logger.warning(f"Video validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space before processing
            video_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(video_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Extract audio with timeout
            logger.info(f"Extracting audio as {output_format} for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                extractor = AudioExtractor(str(input_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, extractor.extract, output_format),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not success:
                    logger.error(f"Audio extraction failed for user {user_id}")
                    raise AudioExtractionError(f"No pude extraer el audio en formato {output_format.upper()}")

            except asyncio.TimeoutError as e:
                logger.error(f"Audio extraction timed out for user {user_id}")
                raise ProcessingTimeoutError("La extracción de audio tardó demasiado") from e

            # Send extracted audio
            logger.info(f"Sending extracted audio to user {user_id}")
            try:
                with open(output_path, "rb") as audio_file:
                    await update.message.reply_audio(audio=audio_file)
                logger.info(f"Audio sent successfully to user {user_id}")
            except Exception as e:
                logger.error(f"Failed to send audio to user {user_id}: {e}")
                raise

            # Delete processing message on success
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")

        except (DownloadError, AudioExtractionError, ProcessingTimeoutError, ValidationError) as e:
            await handle_processing_error(update, e, user_id)
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")

        except Exception as e:
            logger.exception(f"Unexpected error extracting audio for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")


# Default segment duration for split command
DEFAULT_SEGMENT_DURATION = 60
DEFAULT_AUDIO_SEGMENT_DURATION = 60

# Split session states
SPLIT_WAITING_START_TIME = "waiting_start_time"
SPLIT_WAITING_END_TIME = "waiting_end_time"
SPLIT_CONFIRMING = "confirming"


async def handle_split_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /split command to split video into segments.

    Usage:
        /split duration <segundos> - Divide en segmentos de N segundos
        /split parts <cantidad> - Divide en N partes iguales
        /split (solo) - Divide en segmentos de 60 segundos

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    logger.info(f"Split command received from user {user_id}")

    # Get video from message or reply
    video, is_reply = await _get_video_from_message(update)

    if not video:
        await update.message.reply_text(
            "Responde a un video con este comando para dividirlo.\n\n"
            "Uso:\n"
            "/split duration 30 - Divide en segmentos de 30 segundos\n"
            "/split parts 5 - Divide en 5 partes iguales\n"
            "/split - Divide en segmentos de 60 segundos (default)"
        )
        return

    # Validate file size before downloading
    if video.file_size:
        is_valid, error_msg = validate_file_size(video.file_size, config.max_incoming_file_size_mb)
        if not is_valid:
            logger.warning(f"File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Parse command arguments
    args = context.args if context.args else []

    split_mode = "duration"  # Default mode
    split_value = DEFAULT_SEGMENT_DURATION

    if len(args) >= 1:
        if args[0].lower() in ["duration", "duracion", "tiempo", "time"]:
            split_mode = "duration"
            if len(args) >= 2:
                try:
                    split_value = int(args[1])
                    if split_value < config.MIN_SEGMENT_SECONDS:
                        await update.message.reply_text(
                            f"La duración mínima es {config.MIN_SEGMENT_SECONDS} segundos."
                        )
                        return
                except ValueError:
                    await update.message.reply_text(
                        "Por favor especifica un número válido de segundos.\n"
                        "Ejemplo: /split duration 30"
                    )
                    return
        elif args[0].lower() in ["parts", "partes", "cantidad", "number"]:
            split_mode = "parts"
            if len(args) >= 2:
                try:
                    split_value = int(args[1])
                    if split_value < 1:
                        await update.message.reply_text(
                            "El número de partes debe ser al menos 1."
                        )
                        return
                    if split_value > config.MAX_SEGMENTS:
                        await update.message.reply_text(
                            f"El máximo de partes es {config.MAX_SEGMENTS}."
                        )
                        return
                except ValueError:
                    await update.message.reply_text(
                        "Por favor especifica un número válido de partes.\n"
                        "Ejemplo: /split parts 5"
                    )
                    return
            else:
                await update.message.reply_text(
                    "Por favor especifica cuántas partes quieres.\n"
                    "Ejemplo: /split parts 5"
                )
                return
        else:
            # Try to parse as a number (assume duration mode)
            try:
                split_value = int(args[0])
                if split_value < config.MIN_SEGMENT_SECONDS:
                    await update.message.reply_text(
                        f"La duración mínima es {config.MIN_SEGMENT_SECONDS} segundos."
                    )
                    return
            except ValueError:
                await update.message.reply_text(
                    "Argumento no reconocido. Usa 'duration' o 'parts'.\n"
                    "Ejemplo: /split duration 30 o /split parts 5"
                )
                return

    # Send processing message
    processing_message = None
    try:
        if split_mode == "duration":
            processing_message = await update.message.reply_text(
                f"Dividiendo video en segmentos de {split_value} segundos..."
            )
        else:
            processing_message = await update.message.reply_text(
                f"Dividiendo video en {split_value} partes iguales..."
            )
    except Exception as e:
        logger.warning(f"Could not send processing message to user {user_id}: {e}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_{user_id}_{video.file_unique_id}.mp4"
            output_dir = temp_mgr.get_temp_path(f"split_{user_id}_{video.file_unique_id}")

            input_path = temp_mgr.get_temp_path(input_filename)

            # Download video
            logger.info(f"Downloading video from user {user_id} for splitting")
            try:
                file = await video.get_file()
                await _download_with_retry(file, input_path)
                logger.info(f"Video downloaded to {input_path}")
            except Exception as e:
                logger.error(f"Failed to download video for user {user_id}: {e}")
                raise DownloadError("No pude descargar el video") from e

            # Validate video integrity after download
            is_valid, error_msg = validate_video_file(str(input_path))
            if not is_valid:
                logger.warning(f"Video validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space before processing
            video_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(video_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Create output directory for segments
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Split video with timeout
            logger.info(f"Splitting video for user {user_id} (mode={split_mode}, value={split_value})")
            try:
                loop = asyncio.get_event_loop()
                splitter = VideoSplitter(str(input_path), str(output_dir))

                if split_mode == "duration":
                    # Check how many segments would be created
                    duration = await loop.run_in_executor(None, splitter.get_video_duration)
                    expected_segments = int(duration // split_value) + (1 if duration % split_value > 0 else 0)

                    if expected_segments > config.MAX_SEGMENTS:
                        await update.message.reply_text(
                            f"El video generaría demasiadas partes ({expected_segments}). "
                            f"Intenta con una duración mayor (máximo {config.MAX_SEGMENTS} partes)."
                        )
                        if processing_message:
                            try:
                                await processing_message.delete()
                            except Exception:
                                pass
                        return

                    segments = await asyncio.wait_for(
                        loop.run_in_executor(None, splitter.split_by_duration, split_value),
                        timeout=config.PROCESSING_TIMEOUT
                    )
                else:  # split_mode == "parts"
                    segments = await asyncio.wait_for(
                        loop.run_in_executor(None, splitter.split_by_parts, split_value),
                        timeout=config.PROCESSING_TIMEOUT
                    )

                    # Check if we got too many segments (shouldn't happen due to validation in split_by_parts)
                    if len(segments) > config.MAX_SEGMENTS:
                        await update.message.reply_text(
                            f"El video generaría demasiadas partes ({len(segments)}). "
                            f"Intenta con menos partes (máximo {config.MAX_SEGMENTS})."
                        )
                        if processing_message:
                            try:
                                await processing_message.delete()
                            except Exception:
                                pass
                        return

                if not segments:
                    logger.error(f"Video splitting produced no segments for user {user_id}")
                    raise VideoSplitError("No se generaron segmentos del video")

            except asyncio.TimeoutError as e:
                logger.error(f"Video splitting timed out for user {user_id}")
                raise ProcessingTimeoutError("La división del video tardó demasiado") from e

            # Send segments to user
            logger.info(f"Sending {len(segments)} segments to user {user_id}")
            total_segments = len(segments)

            for i, segment_path in enumerate(segments, 1):
                try:
                    # Update progress message
                    if processing_message:
                        try:
                            await processing_message.edit_text(
                                f"Enviando parte {i} de {total_segments}..."
                            )
                        except Exception as e:
                            logger.warning(f"Could not update progress message: {e}")

                    # Send segment
                    with open(segment_path, "rb") as video_file:
                        await update.message.reply_video(
                            video=video_file,
                            caption=f"Parte {i} de {total_segments}"
                        )
                    logger.info(f"Sent segment {i}/{total_segments} to user {user_id}")

                except Exception as e:
                    logger.error(f"Failed to send segment {i} to user {user_id}: {e}")
                    await update.message.reply_text(
                        f"Error enviando la parte {i} de {total_segments}."
                    )

            # Send completion message
            await update.message.reply_text(
                f"¡Listo! El video se dividió en {total_segments} partes."
            )
            logger.info(f"All segments sent successfully to user {user_id}")

            # Delete processing message on success
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")

        except (DownloadError, VideoSplitError, ProcessingTimeoutError, ValidationError) as e:
            await handle_processing_error(update, e, user_id)
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")

        except Exception as e:
            logger.exception(f"Unexpected error splitting video for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")


# Default audio segment duration for split_audio command
DEFAULT_AUDIO_SEGMENT_DURATION = 60


async def handle_split_audio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /split_audio command to split audio files into segments.

    Usage:
        /split_audio duration <segundos> - Divide en segmentos de N segundos
        /split_audio parts <cantidad> - Divide en N partes iguales
        /split_audio (solo) - Divide en segmentos de 60 segundos

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    logger.info(f"Split audio command received from user {user_id}")

    # Get audio from message or reply
    audio, is_reply = await _get_audio_from_message(update)

    if not audio:
        await update.message.reply_text(
            "Responde a un audio con este comando para dividirlo.\n\n"
            "Uso:\n"
            "/split_audio duration 30 - Divide en segmentos de 30 segundos\n"
            "/split_audio parts 5 - Divide en 5 partes iguales\n"
            "/split_audio - Divide en segmentos de 60 segundos (default)"
        )
        return

    # Validate file size before downloading
    if audio.file_size:
        is_valid, error_msg = validate_file_size(audio.file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Parse command arguments
    args = context.args if context.args else []

    split_mode = "duration"  # Default mode
    split_value = DEFAULT_AUDIO_SEGMENT_DURATION

    if len(args) >= 1:
        if args[0].lower() in ["duration", "duracion", "tiempo", "time"]:
            split_mode = "duration"
            if len(args) >= 2:
                try:
                    split_value = int(args[1])
                    if split_value < config.MIN_AUDIO_SEGMENT_SECONDS:
                        await update.message.reply_text(
                            f"La duración mínima es {config.MIN_AUDIO_SEGMENT_SECONDS} segundos."
                        )
                        return
                except ValueError:
                    await update.message.reply_text(
                        "Por favor especifica un número válido de segundos.\n"
                        "Ejemplo: /split_audio duration 30"
                    )
                    return
        elif args[0].lower() in ["parts", "partes", "cantidad", "number"]:
            split_mode = "parts"
            if len(args) >= 2:
                try:
                    split_value = int(args[1])
                    if split_value < 1:
                        await update.message.reply_text(
                            "El número de partes debe ser al menos 1."
                        )
                        return
                    if split_value > config.MAX_AUDIO_SEGMENTS:
                        await update.message.reply_text(
                            f"El máximo de partes es {config.MAX_AUDIO_SEGMENTS}."
                        )
                        return
                except ValueError:
                    await update.message.reply_text(
                        "Por favor especifica un número válido de partes.\n"
                        "Ejemplo: /split_audio parts 5"
                    )
                    return
            else:
                await update.message.reply_text(
                    "Por favor especifica cuántas partes quieres.\n"
                    "Ejemplo: /split_audio parts 5"
                )
                return
        else:
            # Try to parse as a number (assume duration mode)
            try:
                split_value = int(args[0])
                if split_value < config.MIN_AUDIO_SEGMENT_SECONDS:
                    await update.message.reply_text(
                        f"La duración mínima es {config.MIN_AUDIO_SEGMENT_SECONDS} segundos."
                    )
                    return
            except ValueError:
                await update.message.reply_text(
                    "Argumento no reconocido. Usa 'duration' o 'parts'.\n"
                    "Ejemplo: /split_audio duration 30 o /split_audio parts 5"
                )
                return

    # Send processing message
    processing_message = None
    try:
        if split_mode == "duration":
            processing_message = await update.message.reply_text(
                f"Dividiendo audio en segmentos de {split_value} segundos..."
            )
        else:
            processing_message = await update.message.reply_text(
                f"Dividiendo audio en {split_value} partes iguales..."
            )
    except Exception as e:
        logger.warning(f"Could not send processing message to user {user_id}: {e}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_audio_{user_id}_{audio.file_unique_id}.mp3"
            output_dir = temp_mgr.get_temp_path(f"split_audio_{user_id}_{audio.file_unique_id}")

            input_path = temp_mgr.get_temp_path(input_filename)

            # Download audio
            logger.info(f"Downloading audio from user {user_id} for splitting")
            try:
                file = await audio.get_file()
                await _download_with_retry(file, input_path)
                logger.info(f"Audio downloaded to {input_path}")
            except Exception as e:
                logger.error(f"Failed to download audio for user {user_id}: {e}")
                raise DownloadError("No pude descargar el audio") from e

            # Validate audio integrity after download
            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                logger.warning(f"Audio validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space before processing
            audio_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(audio_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Create output directory for segments
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Split audio with timeout
            logger.info(f"Splitting audio for user {user_id} (mode={split_mode}, value={split_value})")
            try:
                loop = asyncio.get_event_loop()
                splitter = AudioSplitter(str(input_path), str(output_dir))

                if split_mode == "duration":
                    # Check how many segments would be created
                    duration = await loop.run_in_executor(None, splitter.get_audio_duration)
                    expected_segments = int(duration // split_value) + (1 if duration % split_value > 0 else 0)

                    if expected_segments > config.MAX_AUDIO_SEGMENTS:
                        await update.message.reply_text(
                            f"El audio generaría demasiadas partes ({expected_segments}). "
                            f"Intenta con una duración mayor (máximo {config.MAX_AUDIO_SEGMENTS} partes)."
                        )
                        if processing_message:
                            try:
                                await processing_message.delete()
                            except Exception:
                                pass
                        return

                    segments = await asyncio.wait_for(
                        loop.run_in_executor(None, splitter.split_by_duration, split_value),
                        timeout=config.PROCESSING_TIMEOUT
                    )
                else:  # split_mode == "parts"
                    segments = await asyncio.wait_for(
                        loop.run_in_executor(None, splitter.split_by_parts, split_value),
                        timeout=config.PROCESSING_TIMEOUT
                    )

                    # Check if we got too many segments
                    if len(segments) > config.MAX_AUDIO_SEGMENTS:
                        await update.message.reply_text(
                            f"El audio generaría demasiadas partes ({len(segments)}). "
                            f"Intenta con menos partes (máximo {config.MAX_AUDIO_SEGMENTS})."
                        )
                        if processing_message:
                            try:
                                await processing_message.delete()
                            except Exception:
                                pass
                        return

                if not segments:
                    logger.error(f"Audio splitting produced no segments for user {user_id}")
                    raise AudioSplitError("No se generaron segmentos del audio")

            except asyncio.TimeoutError as e:
                logger.error(f"Audio splitting timed out for user {user_id}")
                raise ProcessingTimeoutError("La división del audio tardó demasiado") from e

            # Send segments to user
            logger.info(f"Sending {len(segments)} audio segments to user {user_id}")
            total_segments = len(segments)

            for i, segment_path in enumerate(segments, 1):
                try:
                    # Update progress message
                    if processing_message:
                        try:
                            await processing_message.edit_text(
                                f"Enviando parte {i} de {total_segments}..."
                            )
                        except Exception as e:
                            logger.warning(f"Could not update progress message: {e}")

                    # Send segment
                    with open(segment_path, "rb") as audio_file:
                        await update.message.reply_audio(
                            audio=audio_file,
                            caption=f"Parte {i} de {total_segments}"
                        )
                    logger.info(f"Sent audio segment {i}/{total_segments} to user {user_id}")

                except Exception as e:
                    logger.error(f"Failed to send audio segment {i} to user {user_id}: {e}")
                    await update.message.reply_text(
                        f"Error enviando la parte {i} de {total_segments}."
                    )

            # Send completion message
            await update.message.reply_text(
                f"¡Listo! El audio se dividió en {total_segments} partes."
            )
            logger.info(f"All audio segments sent successfully to user {user_id}")

            # Delete processing message on success
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")

        except (DownloadError, AudioSplitError, ProcessingTimeoutError, ValidationError) as e:
            await handle_processing_error(update, e, user_id)
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")

        except Exception as e:
            logger.exception(f"Unexpected error splitting audio for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"Could not delete processing message: {e}")


# Note: Join command configuration now uses bot.config values


async def handle_join_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /join command to start a video join session.

    Usage: /join - Start a session to collect videos for joining

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    logger.info(f"Join command received from user {user_id}")

    # Check if there's already an active session
    if context.user_data.get("join_session"):
        await update.message.reply_text(
            "Ya tienes una sesión de unión activa. "
            f"Tienes {len(context.user_data['join_session']['videos'])} video(s) agregados.\n\n"
            "Envía más videos o usa /done para unir, /cancel para cancelar."
        )
        return

    # Initialize join session
    context.user_data["join_session"] = {
        "videos": [],
        "temp_mgr": TempManager(),
        "last_activity": asyncio.get_event_loop().time(),
    }

    await update.message.reply_text(
        "🎬 *Modo unión de videos activado*\n\n"
        "Envíame los videos que quieres unir (máximo 10).\n"
        "Los videos se unirán en el orden en que los envíes.\n\n"
        "Actualmente tienes: *0 videos*\n\n"
        "Envía el primer video:",
        parse_mode="Markdown",
        reply_markup=_get_join_video_keyboard(0)
    )


async def handle_join_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle video messages during an active join session.

    Downloads each video and tracks it in the user's join session.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id

    # Check if there's an active join session
    session = context.user_data.get("join_session")
    if not session:
        # No active session, let the default video handler process it
        await handle_video(update, context)
        return

    # Check session timeout
    current_time = asyncio.get_event_loop().time()
    if current_time - session["last_activity"] > config.JOIN_SESSION_TIMEOUT:
        logger.info(f"Join session expired for user {user_id}")
        # Clean up expired session
        session["temp_mgr"].cleanup()
        context.user_data.pop("join_session", None)
        await update.message.reply_text(
            "La sesión expiró. Usa /join para comenzar de nuevo."
        )
        return

    # Update last activity
    session["last_activity"] = current_time

    # Check if we've reached the maximum
    if len(session["videos"]) >= config.JOIN_MAX_VIDEOS:
        await update.message.reply_text(
            f"Máximo {config.JOIN_MAX_VIDEOS} videos permitidos.\n"
            "Usa /done para unir o /cancel para cancelar."
        )
        return

    # Get video from message
    video = update.message.video
    if not video:
        await update.message.reply_text(
            "Por favor envía un video válido."
        )
        return

    # Validate file size before downloading
    if video.file_size:
        is_valid, error_msg = validate_file_size(video.file_size, config.max_incoming_file_size_mb)
        if not is_valid:
            logger.warning(f"File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Send processing message
    processing_message = None
    try:
        processing_message = await update.message.reply_text(
            f"Descargando video {len(session['videos']) + 1}..."
        )
    except Exception as e:
        logger.warning(f"Could not send processing message to user {user_id}: {e}")

    try:
        temp_mgr = session["temp_mgr"]

        # Generate safe filename
        video_index = len(session["videos"]) + 1
        input_filename = f"join_{user_id}_video{video_index:02d}_{video.file_unique_id}.mp4"
        input_path = temp_mgr.get_temp_path(input_filename)

        # Download video
        logger.info(f"Downloading video {video_index} for join session, user {user_id}")
        try:
            file = await video.get_file()
            await _download_with_retry(file, input_path)
            logger.info(f"Video downloaded to {input_path}")
        except Exception as e:
            logger.error(f"Failed to download video for user {user_id}: {e}")
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception:
                    pass
            await update.message.reply_text(
                "No pude descargar el video. Intenta con otro archivo."
            )
            return

        # Validate video integrity after download
        is_valid, error_msg = validate_video_file(str(input_path))
        if not is_valid:
            logger.warning(f"Video validation failed for user {user_id}: {error_msg}")
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception:
                    pass
            await update.message.reply_text(error_msg)
            return

        # Track the video
        session["videos"].append(str(input_path))
        temp_mgr.track_file(str(input_path))

        video_count = len(session["videos"])

        # Delete processing message
        if processing_message:
            try:
                await processing_message.delete()
            except Exception:
                pass

        # Send confirmation with keyboard
        if video_count == 1:
            await update.message.reply_text(
                f"✓ Video {video_count} agregado.\n\n"
                f"Actualmente tienes: *{video_count} video*\n"
                f"Envía más videos o presiona el botón para unir:",
                parse_mode="Markdown",
                reply_markup=_get_join_video_keyboard(video_count)
            )
        elif video_count < config.JOIN_MIN_VIDEOS:
            remaining = config.JOIN_MIN_VIDEOS - video_count
            await update.message.reply_text(
                f"✓ Video {video_count} agregado.\n\n"
                f"Necesitas *{remaining}* video(s) más para poder unir.\n"
                f"Actualmente tienes: *{video_count} videos*",
                parse_mode="Markdown",
                reply_markup=_get_join_video_keyboard(video_count)
            )
        else:
            await update.message.reply_text(
                f"✓ Video {video_count} agregado.\n\n"
                f"Actualmente tienes: *{video_count} videos*\n"
                f"Máximo: {config.JOIN_MAX_VIDEOS}\n\n"
                f"Envía más videos o presiona el botón para unir:",
                parse_mode="Markdown",
                reply_markup=_get_join_video_keyboard(video_count)
            )

    except Exception as e:
        logger.exception(f"Unexpected error handling join video for user {user_id}: {e}")
        if processing_message:
            try:
                await processing_message.delete()
            except Exception:
                pass
        await update.message.reply_text(
            "Ocurrió un error procesando el video. Intenta de nuevo."
        )


async def handle_join_done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /done command or button to complete video or audio joining.

    Joins all collected videos or audios and sends the result.
    Checks for video join session first, then audio join session.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    logger.info(f"Join done command received from user {user_id}")

    # Determine effective message for replies (handles both command and callback)
    effective_message = update.message
    if not effective_message and update.callback_query:
        effective_message = update.callback_query.message

    # Check if there's an active video join session
    session = context.user_data.get("join_session")
    if not session:
        # No video session - check for audio join session
        if context.user_data.get("join_audio_session"):
            # Delegate to audio join handler
            await handle_join_audio_done(update, context)
            return
        if effective_message:
            await effective_message.reply_text(
                "No hay una sesión de unión activa. Usa /join o /join_audio para comenzar."
            )
        return

    # Check session timeout
    current_time = asyncio.get_event_loop().time()
    if current_time - session["last_activity"] > config.JOIN_SESSION_TIMEOUT:
        logger.info(f"Join session expired for user {user_id}")
        session["temp_mgr"].cleanup()
        context.user_data.pop("join_session", None)
        if effective_message:
            await effective_message.reply_text(
                "La sesión expiró. Usa /join para comenzar de nuevo."
            )
        return

    # Check minimum videos
    video_count = len(session["videos"])
    if video_count < config.JOIN_MIN_VIDEOS:
        if effective_message:
            await effective_message.reply_text(
                f"Necesitas al menos {config.JOIN_MIN_VIDEOS} videos para unir. "
                f"Actualmente tienes {video_count}."
            )
        return

    # Check disk space before joining
    total_size_mb = 0
    for video_path in session["videos"]:
        total_size_mb += Path(video_path).stat().st_size / (1024 * 1024)
    required_space = estimate_required_space(int(total_size_mb))
    has_space, space_error = check_disk_space(required_space)
    if not has_space:
        logger.warning(f"Disk space check failed for user {user_id}: {space_error}")
        if effective_message:
            await effective_message.reply_text(space_error)
        return

    # Delete the callback query message (the one with buttons) if it exists
    if update.callback_query and update.callback_query.message:
        try:
            await update.callback_query.message.delete()
        except Exception as e:
            logger.debug(f"Could not delete callback message: {e}")

    # Send processing message
    processing_message = None
    try:
        if effective_message:
            processing_message = await effective_message.reply_text(
                f"Uniendo {video_count} videos... Esto puede tomar un momento."
            )
    except Exception as e:
        logger.warning(f"Could not send processing message to user {user_id}: {e}")

    temp_mgr = session["temp_mgr"]

    try:
        # Generate output path
        output_filename = f"joined_{user_id}_{int(asyncio.get_event_loop().time())}.mp4"
        output_path = temp_mgr.get_temp_path(output_filename)

        # Create VideoJoiner and add all videos
        logger.info(f"Starting video join for user {user_id} with {video_count} videos")
        joiner = VideoJoiner(str(output_path))

        for video_path in session["videos"]:
            joiner.add_video(video_path)

        # Join videos with timeout
        try:
            loop = asyncio.get_event_loop()
            success = await asyncio.wait_for(
                loop.run_in_executor(None, joiner.join_videos),
                timeout=config.JOIN_TIMEOUT  # Dedicated join timeout (120s default)
            )

            if not success:
                logger.error(f"Video joining failed for user {user_id}")
                raise VideoJoinError("No pude unir los videos")

        except asyncio.TimeoutError as e:
            logger.error(f"Video joining timed out for user {user_id}")
            raise ProcessingTimeoutError("La unión de videos tardó demasiado") from e

        # Send joined video
        logger.info(f"Sending joined video to user {user_id}")
        try:
            if effective_message:
                with open(output_path, "rb") as video_file:
                    await effective_message.reply_video(
                        video=video_file,
                        caption=f"Video unido ({video_count} partes)"
                    )
                logger.info(f"Joined video sent successfully to user {user_id}")
        except Exception as e:
            logger.error(f"Failed to send joined video to user {user_id}: {e}")
            raise

        # Delete processing message on success
        if processing_message:
            try:
                await processing_message.delete()
            except Exception as e:
                logger.warning(f"Could not delete processing message: {e}")

        # Clean up session
        temp_mgr.cleanup()
        context.user_data.pop("join_session", None)

    except (VideoJoinError, ProcessingTimeoutError) as e:
        await handle_processing_error(update, e, user_id)
        if processing_message:
            try:
                await processing_message.delete()
            except Exception:
                pass
        # Clean up session on error
        temp_mgr.cleanup()
        context.user_data.pop("join_session", None)

    except Exception as e:
        logger.exception(f"Unexpected error joining videos for user {user_id}: {e}")
        await handle_processing_error(update, e, user_id)
        if processing_message:
            try:
                await processing_message.delete()
            except Exception:
                pass
        # Clean up session on error
        temp_mgr.cleanup()
        context.user_data.pop("join_session", None)


async def handle_join_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cancel command or button to cancel a join session.

    Clears session data and cleans up temporary files.
    Checks for video join session first, then audio join session.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    logger.info(f"Join cancel command received from user {user_id}")

    # Determine effective message for replies (handles both command and callback)
    effective_message = update.message
    if not effective_message and update.callback_query:
        effective_message = update.callback_query.message

    # Check if there's an active video join session
    session = context.user_data.get("join_session")
    if not session:
        # No video session - check for audio join session
        if context.user_data.get("join_audio_session"):
            # Delegate to audio join handler
            await handle_join_audio_cancel(update, context)
            return
        if effective_message:
            await effective_message.reply_text(
                "No hay una sesión de unión activa."
            )
        return

    # Delete the callback query message (the one with buttons) if it exists
    if update.callback_query and update.callback_query.message:
        try:
            await update.callback_query.message.delete()
        except Exception as e:
            logger.debug(f"Could not delete callback message: {e}")

    # Clean up temp files
    video_count = len(session["videos"])
    session["temp_mgr"].cleanup()
    context.user_data.pop("join_session", None)

    if effective_message:
        await effective_message.reply_text(
            f"Sesión cancelada. {video_count} video(s) descartados."
        )


async def handle_join_video_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle callback queries from video join session buttons.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    logger.info(f"Join video callback received: {callback_data} from user {user_id}")

    # Parse action from callback data (format: join_video_action:<action>)
    if not callback_data.startswith("join_video_action:"):
        logger.warning(f"Unexpected callback data: {callback_data}")
        return

    action = callback_data.split(":")[1]

    if action == "done":
        # Delegate to existing done handler
        await handle_join_done(update, context)
    elif action == "cancel":
        # Delegate to existing cancel handler
        await handle_join_cancel(update, context)
    else:
        logger.warning(f"Unknown join video action: {action}")


# =============================================================================
# Audio Join Handlers
# =============================================================================

async def handle_join_audio_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /join_audio command to start an audio join session.

    Usage: /join_audio - Start a session to collect audio files for joining

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    logger.info(f"Join audio command received from user {user_id}")

    # Check if there's already an active audio join session
    if context.user_data.get("join_audio_session"):
        await update.message.reply_text(
            "Ya tienes una sesión de unión de audio activa. "
            f"Tienes {len(context.user_data['join_audio_session']['audios'])} audio(s) agregados.\n\n"
            "Envía más audios o usa /done para unir, /cancel para cancelar."
        )
        return

    # Check if there's an active video join session (can't have both)
    if context.user_data.get("join_session"):
        await update.message.reply_text(
            "Ya tienes una sesión de unión de videos activa. "
            "Usa /cancel para cancelarla primero, luego usa /join_audio."
        )
        return

    # Initialize audio join session
    context.user_data["join_audio_session"] = {
        "audios": [],
        "temp_mgr": TempManager(),
        "last_activity": asyncio.get_event_loop().time(),
    }

    await update.message.reply_text(
        "🎵 *Modo unión de audio activado*\n\n"
        "Envíame los archivos de audio que quieres unir (máximo 20).\n"
        "Los audios se unirán en el orden en que los envíes.\n\n"
        "Actualmente tienes: *0 audios*\n\n"
        "Envía el primer archivo de audio:",
        parse_mode="Markdown",
        reply_markup=_get_join_audio_keyboard(0)
    )


async def handle_join_audio_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio file messages during an active audio join session.

    Downloads each audio file and tracks it in the user's join session.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id

    # Check if there's an active audio join session
    session = context.user_data.get("join_audio_session")
    if not session:
        # No active audio join session, let the default audio handler process it
        await handle_audio_file(update, context)
        return

    # Check session timeout
    current_time = asyncio.get_event_loop().time()
    if current_time - session["last_activity"] > config.JOIN_SESSION_TIMEOUT:
        logger.info(f"Join audio session expired for user {user_id}")
        # Clean up expired session
        session["temp_mgr"].cleanup()
        context.user_data.pop("join_audio_session", None)
        await update.message.reply_text(
            "La sesión expiró. Usa /join_audio para comenzar de nuevo."
        )
        return

    # Update last activity
    session["last_activity"] = current_time

    # Check if we've reached the maximum
    if len(session["audios"]) >= config.JOIN_MAX_AUDIO_FILES:
        await update.message.reply_text(
            f"Máximo {config.JOIN_MAX_AUDIO_FILES} archivos de audio permitidos.\n"
            "Usa /done para unir o /cancel para cancelar."
        )
        return

    # Get audio from message (native audio or document attachment)
    file_id, file_size, file_unique_id = _get_message_audio_source(update.message)
    if not file_id:
        await update.message.reply_text(
            "Por favor envía un archivo de audio válido."
        )
        return

    # Validate file size before downloading
    if file_size:
        is_valid, error_msg = validate_file_size(file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Send processing message
    processing_message = None
    try:
        processing_message = await update.message.reply_text(
            f"Descargando audio {len(session['audios']) + 1}..."
        )
    except Exception as e:
        logger.warning(f"Could not send processing message to user {user_id}: {e}")

    try:
        temp_mgr = session["temp_mgr"]

        # Generate safe filename
        audio_index = len(session["audios"]) + 1
        input_filename = f"join_audio_{user_id}_{audio_index:02d}_{file_unique_id}.mp3"
        input_path = temp_mgr.get_temp_path(input_filename)

        # Download audio
        logger.info(f"Downloading audio {audio_index} for join session, user {user_id}")
        try:
            file = await context.bot.get_file(file_id)
            await _download_with_retry(file, input_path)
            logger.info(f"Audio downloaded to {input_path}")
        except Exception as e:
            logger.error(f"Failed to download audio for user {user_id}: {e}")
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception:
                    pass
            await update.message.reply_text(
                "No pude descargar el audio. Intenta con otro archivo."
            )
            return

        # Validate audio integrity after download
        is_valid, error_msg = validate_audio_file(str(input_path))
        if not is_valid:
            logger.warning(f"Audio validation failed for user {user_id}: {error_msg}")
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception:
                    pass
            await update.message.reply_text(error_msg)
            return

        # Track the audio
        session["audios"].append(str(input_path))
        temp_mgr.track_file(str(input_path))

        audio_count = len(session["audios"])

        # Delete processing message
        if processing_message:
            try:
                await processing_message.delete()
            except Exception:
                pass

        # Send confirmation with keyboard
        if audio_count == 1:
            await update.message.reply_text(
                f"✓ Audio {audio_count} agregado.\n\n"
                f"Actualmente tienes: *{audio_count} audio*\n"
                f"Envía más audios o presiona el botón para unir:",
                parse_mode="Markdown",
                reply_markup=_get_join_audio_keyboard(audio_count)
            )
        elif audio_count < config.JOIN_MIN_AUDIO_FILES:
            remaining = config.JOIN_MIN_AUDIO_FILES - audio_count
            await update.message.reply_text(
                f"✓ Audio {audio_count} agregado.\n\n"
                f"Necesitas *{remaining}* audio(s) más para poder unir.\n"
                f"Actualmente tienes: *{audio_count} audios*",
                parse_mode="Markdown",
                reply_markup=_get_join_audio_keyboard(audio_count)
            )
        else:
            await update.message.reply_text(
                f"✓ Audio {audio_count} agregado.\n\n"
                f"Actualmente tienes: *{audio_count} audios*\n"
                f"Máximo: {config.JOIN_MAX_AUDIO_FILES}\n\n"
                f"Envía más audios o presiona el botón para unir:",
                parse_mode="Markdown",
                reply_markup=_get_join_audio_keyboard(audio_count)
            )

    except Exception as e:
        logger.exception(f"Unexpected error handling join audio for user {user_id}: {e}")
        if processing_message:
            try:
                await processing_message.delete()
            except Exception:
                pass
        await update.message.reply_text(
            "Ocurrió un error procesando el audio. Intenta de nuevo."
        )


async def handle_join_audio_done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /done command or button to complete audio joining.

    Joins all collected audio files and sends the result.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    logger.info(f"Join audio done command received from user {user_id}")

    # Determine effective message for replies (handles both command and callback)
    effective_message = update.message
    if not effective_message and update.callback_query:
        effective_message = update.callback_query.message

    # Check if there's an active audio join session
    session = context.user_data.get("join_audio_session")
    if not session:
        # No active audio join session - let video join handler check
        # This will be handled by the router in main.py or the video handler
        if effective_message:
            await effective_message.reply_text(
                "No hay una sesión de unión de audio activa. Usa /join_audio para comenzar."
            )
        return

    # Check session timeout
    current_time = asyncio.get_event_loop().time()
    if current_time - session["last_activity"] > config.JOIN_SESSION_TIMEOUT:
        logger.info(f"Join audio session expired for user {user_id}")
        session["temp_mgr"].cleanup()
        context.user_data.pop("join_audio_session", None)
        if effective_message:
            await effective_message.reply_text(
                "La sesión expiró. Usa /join_audio para comenzar de nuevo."
            )
        return

    # Check minimum audios
    audio_count = len(session["audios"])
    if audio_count < config.JOIN_MIN_AUDIO_FILES:
        if effective_message:
            await effective_message.reply_text(
                f"Necesitas al menos {config.JOIN_MIN_AUDIO_FILES} audios para unir. "
                f"Actualmente tienes {audio_count}."
            )
        return

    # Check disk space before joining
    total_size_mb = 0
    for audio_path in session["audios"]:
        total_size_mb += Path(audio_path).stat().st_size / (1024 * 1024)
    required_space = estimate_required_space(int(total_size_mb))
    has_space, space_error = check_disk_space(required_space)
    if not has_space:
        logger.warning(f"Disk space check failed for user {user_id}: {space_error}")
        if effective_message:
            await effective_message.reply_text(space_error)
        return

    # Delete the callback query message (the one with buttons) if it exists
    if update.callback_query and update.callback_query.message:
        try:
            await update.callback_query.message.delete()
        except Exception as e:
            logger.debug(f"Could not delete callback message: {e}")

    # Send processing message
    processing_message = None
    try:
        if effective_message:
            processing_message = await effective_message.reply_text(
                f"Uniendo {audio_count} audios... Esto puede tomar un momento."
            )
    except Exception as e:
        logger.warning(f"Could not send processing message to user {user_id}: {e}")

    temp_mgr = session["temp_mgr"]

    try:
        # Generate output path
        output_filename = f"joined_audio_{user_id}_{int(asyncio.get_event_loop().time())}.mp3"
        output_path = temp_mgr.get_temp_path(output_filename)

        # Create AudioJoiner and add all audios
        logger.info(f"Starting audio join for user {user_id} with {audio_count} audios")
        joiner = AudioJoiner(str(output_path))

        for audio_path in session["audios"]:
            joiner.add_audio(audio_path)

        # Join audios with timeout
        try:
            loop = asyncio.get_event_loop()
            success = await asyncio.wait_for(
                loop.run_in_executor(None, joiner.join_audios),
                timeout=config.JOIN_AUDIO_TIMEOUT
            )

            if not success:
                logger.error(f"Audio joining failed for user {user_id}")
                raise AudioJoinError("No pude unir los archivos de audio")

        except asyncio.TimeoutError as e:
            logger.error(f"Audio joining timed out for user {user_id}")
            raise ProcessingTimeoutError("La unión de audios tardó demasiado") from e

        # Send joined audio
        logger.info(f"Sending joined audio to user {user_id}")
        try:
            if effective_message:
                with open(output_path, "rb") as audio_file:
                    await effective_message.reply_audio(
                        audio=audio_file,
                        caption=f"Audio unido ({audio_count} partes)"
                    )
                logger.info(f"Joined audio sent successfully to user {user_id}")
        except Exception as e:
            logger.error(f"Failed to send joined audio to user {user_id}: {e}")
            raise

        # Delete processing message on success
        if processing_message:
            try:
                await processing_message.delete()
            except Exception as e:
                logger.warning(f"Could not delete processing message: {e}")

        # Clean up session
        temp_mgr.cleanup()
        context.user_data.pop("join_audio_session", None)

    except (AudioJoinError, ProcessingTimeoutError) as e:
        await handle_processing_error(update, e, user_id)
        if processing_message:
            try:
                await processing_message.delete()
            except Exception:
                pass
        # Clean up session on error
        temp_mgr.cleanup()
        context.user_data.pop("join_audio_session", None)

    except Exception as e:
        logger.exception(f"Unexpected error joining audios for user {user_id}: {e}")
        await handle_processing_error(update, e, user_id)
        if processing_message:
            try:
                await processing_message.delete()
            except Exception:
                pass
        # Clean up session on error
        temp_mgr.cleanup()
        context.user_data.pop("join_audio_session", None)


async def handle_join_audio_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cancel command or button to cancel an audio join session.

    Clears session data and cleans up temporary files.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    logger.info(f"Join audio cancel command received from user {user_id}")

    # Determine effective message for replies (handles both command and callback)
    effective_message = update.message
    if not effective_message and update.callback_query:
        effective_message = update.callback_query.message

    # Check if there's an active audio join session
    session = context.user_data.get("join_audio_session")
    if not session:
        # No active audio join session
        if effective_message:
            await effective_message.reply_text(
                "No hay una sesión de unión de audio activa."
            )
        return

    # Delete the callback query message (the one with buttons) if it exists
    if update.callback_query and update.callback_query.message:
        try:
            await update.callback_query.message.delete()
        except Exception as e:
            logger.debug(f"Could not delete callback message: {e}")

    # Clean up temp files
    audio_count = len(session["audios"])
    session["temp_mgr"].cleanup()
    context.user_data.pop("join_audio_session", None)

    if effective_message:
        await effective_message.reply_text(
            f"Sesión cancelada. {audio_count} audio(s) descartados."
        )


async def handle_join_audio_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle callback queries from audio join session buttons.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    logger.info(f"Join audio callback received: {callback_data} from user {user_id}")

    # Parse action from callback data (format: join_audio_action:<action>)
    if not callback_data.startswith("join_audio_action:"):
        logger.warning(f"Unexpected callback data: {callback_data}")
        return

    action = callback_data.split(":")[1]

    if action == "done":
        # Delegate to existing done handler
        await handle_join_audio_done(update, context)
    elif action == "cancel":
        # Delegate to existing cancel handler
        await handle_join_audio_cancel(update, context)
    else:
        logger.warning(f"Unknown join audio action: {action}")


async def _route_incoming_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Route incoming audio to join or merge handlers when a session is active."""
    if context.user_data.get("join_audio_session"):
        await handle_join_audio_file(update, context)
        return True
    if context.user_data.get("merge_video_file_id"):
        await handle_merge_audio_received(update, context)
        return True
    return False


async def _show_audio_menu_for_file(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
    file_size: int | None,
    source_label: str = "Audio",
) -> None:
    """Validate audio size and show the standard audio processing menu."""
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] {source_label} received from user {user_id}")

    if file_size:
        logger.debug(f"[{correlation_id}] Audio file size: {file_size} bytes")
        is_valid, error_msg = validate_file_size(file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(
                f"[{correlation_id}] File size validation failed for user {user_id}: {error_msg}"
            )
            await update.message.reply_text(error_msg)
            return

    context.user_data["audio_menu_file_id"] = file_id
    context.user_data["audio_menu_correlation_id"] = correlation_id

    reply_markup = _get_audio_menu_keyboard()
    await update.message.reply_text(
        "Audio recibido. Selecciona una acción:",
        reply_markup=reply_markup,
    )
    logger.info(f"[{correlation_id}] Audio menu displayed to user {user_id}")


async def handle_audio_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio file messages by showing the audio processing menu.

    If there's an active audio join session, routes to handle_join_audio_file instead.
    If there's an active video-audio merge session, routes to handle_merge_audio_received.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    if await _route_incoming_audio(update, context):
        return

    audio = update.message.audio
    if not audio:
        logger.warning(
            f"No audio found in message from user {update.effective_user.id}"
        )
        await update.message.reply_text("No encontré un archivo de audio en tu mensaje.")
        return

    await _show_audio_menu_for_file(
        update,
        context,
        audio.file_id,
        audio.file_size,
        source_label="Audio file",
    )


async def handle_audio_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio files sent as Telegram documents (e.g. MP3 attachments).

    Reuses the same routing and menu flow as handle_audio_file for join sessions,
    video-audio merge sessions, and the standard audio processing menu.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    document = update.message.document
    if not document or not _is_audio_document(document):
        return

    if await _route_incoming_audio(update, context):
        return

    await _show_audio_menu_for_file(
        update,
        context,
        document.file_id,
        document.file_size,
        source_label="Audio document",
    )


async def handle_merge_audio_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio file received during video-audio merge process.

    Downloads the video and audio files, merges them, and sends the result.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = context.user_data.get("merge_video_correlation_id", str(uuid.uuid4())[:8])
    logger.info(f"[{correlation_id}] Audio received for merge from user {user_id}")

    # Get audio from message (native audio or document attachment)
    audio_file_id, audio_file_size, _ = _get_message_audio_source(update.message)
    if not audio_file_id:
        logger.warning(f"[{correlation_id}] No audio found in message from user {user_id}")
        await update.message.reply_text("No encontré un archivo de audio en tu mensaje.")
        # Clean up merge context
        context.user_data.pop("merge_video_file_id", None)
        context.user_data.pop("merge_video_correlation_id", None)
        return

    # Validate file size
    if audio_file_size:
        is_valid, error_msg = validate_file_size(audio_file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] Audio file size validation failed: {error_msg}")
            await update.message.reply_text(error_msg)
            context.user_data.pop("merge_video_file_id", None)
            context.user_data.pop("merge_video_correlation_id", None)
            return

    # Send processing message
    processing_message = None
    try:
        processing_message = await update.message.reply_text(
            "Uniendo video con audio..."
        )
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not send processing message: {e}")

    # Process with TempManager
    with TempManager() as temp_mgr:
        try:
            # Retrieve video file_id from context
            video_file_id = context.user_data.get("merge_video_file_id")
            if not video_file_id:
                logger.error(f"[{correlation_id}] No video file_id in context")
                raise DownloadError("No encontré el video original. Intenta de nuevo.")

            # Generate safe filenames
            video_filename = f"merge_video_{user_id}_{correlation_id}.mp4"
            audio_filename = f"merge_audio_{user_id}_{correlation_id}.audio"
            output_filename = f"merged_{user_id}_{correlation_id}.mp4"

            video_path = temp_mgr.get_temp_path(video_filename)
            audio_path = temp_mgr.get_temp_path(audio_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download video
            logger.info(f"[{correlation_id}] Downloading video for merge")
            try:
                file = await context.bot.get_file(video_file_id)
                await _download_with_retry(file, video_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Video downloaded to {video_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download video: {e}")
                raise DownloadError("No pude descargar el video") from e

            # Download audio
            logger.info(f"[{correlation_id}] Downloading audio for merge")
            try:
                file = await context.bot.get_file(audio_file_id)
                await _download_with_retry(file, audio_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Audio downloaded to {audio_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download audio: {e}")
                raise DownloadError("No pude descargar el audio") from e

            # Validate files
            is_valid, error_msg = validate_video_file(str(video_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Video validation failed: {error_msg}")
                raise ValidationError(error_msg)

            is_valid, error_msg = validate_audio_file(str(audio_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Audio validation failed: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space
            video_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
            audio_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(video_size_mb + audio_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"[{correlation_id}] Disk space check failed: {space_error}")
                raise ValidationError(space_error)

            # Merge video and audio
            logger.info(f"[{correlation_id}] Merging video and audio")
            try:
                loop = asyncio.get_event_loop()
                merger = VideoAudioMerger(str(video_path), str(audio_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, merger.merge),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not success:
                    logger.error(f"[{correlation_id}] Video-audio merge failed")
                    raise VideoMergeError("No pude unir el video con el audio")

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Merge timed out")
                raise ProcessingTimeoutError("La unión tardó demasiado") from e

            # Send merged video
            logger.info(f"[{correlation_id}] Sending merged video")
            try:
                with open(output_path, "rb") as video_file:
                    await update.message.reply_video(video=video_file)
                logger.info(f"[{correlation_id}] Merged video sent successfully")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to send merged video: {e}")
                raise

            # Delete processing message
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Could not delete processing message: {e}")

            # Clean up context
            context.user_data.pop("merge_video_file_id", None)
            context.user_data.pop("merge_video_correlation_id", None)

        except (DownloadError, VideoMergeError, ProcessingTimeoutError, ValidationError) as e:
            logger.error(f"[{correlation_id}] Merge processing error: {e}")
            await handle_processing_error(update, e, user_id)
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Could not delete processing message: {e}")
            # Clean up context
            context.user_data.pop("merge_video_file_id", None)
            context.user_data.pop("merge_video_correlation_id", None)

        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error in merge: {e}")
            await handle_processing_error(update, e, user_id)
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Could not delete processing message: {e}")
            # Clean up context
            context.user_data.pop("merge_video_file_id", None)
            context.user_data.pop("merge_video_correlation_id", None)


# Video Split Interactive Handlers

async def handle_video_split_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start interactive video split process.

    Downloads the video, gets its duration, and asks user for start time.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    query = update.callback_query
    correlation_id = str(uuid.uuid4())[:8]

    # Respond to callback immediately to avoid timeout
    await query.answer()

    # Get file_id from context (set by handle_video_menu_callback)
    file_id = context.user_data.get("video_menu_file_id")
    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for video split")
        await query.edit_message_text("Error: no se encontró el video.")
        return

    logger.info(f"[{correlation_id}] Video split started by user {user_id}")

    # Store file info in context
    context.user_data["split_video_session"] = {
        "file_id": file_id,
        "correlation_id": correlation_id,
        "state": SPLIT_WAITING_START_TIME,
        "type": "video",
    }

    await query.edit_message_text(
        "✂️ *Dividir Video*\n\n"
        "⏳ Descargando y analizando el video...\n"
        "Este proceso puede tomar unos segundos.",
        parse_mode="Markdown"
    )

    # Download video and get duration
    temp_mgr = TempManager()
    try:
        input_filename = f"split_video_{user_id}_{correlation_id}.mp4"
        input_path = temp_mgr.get_temp_path(input_filename)

        # Download video
        file = await context.bot.get_file(file_id)
        await _download_with_retry(file, input_path, correlation_id=correlation_id)

        # Get duration
        splitter = VideoSplitter(str(input_path), str(temp_mgr.get_temp_path("output")))
        duration = splitter.get_video_duration()

        # Store in session and keep temp_mgr reference
        context.user_data["split_video_session"]["duration"] = duration
        context.user_data["split_video_session"]["input_path"] = input_path
        context.user_data["split_video_session"]["temp_mgr"] = temp_mgr

        minutes = int(duration // 60)
        seconds = int(duration % 60)

        await query.edit_message_text(
            f"✂️ *Dividir Video*\n\n"
            f"📊 Duración del video: *{minutes}m {seconds}s*\n\n"
            f"Envía el tiempo de *inicio* en segundos (ej: 30 para 30 segundos).\n\n"
            f"Puede ser un número decimal (ej: 30.5)",
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.error(f"[{correlation_id}] Error preparing video split: {e}")
        await query.edit_message_text(
            "Error al preparar el video. Intenta de nuevo."
        )
        context.user_data.pop("split_video_session", None)
        temp_mgr.cleanup()


async def handle_video_split_start_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle start time input for video split.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    session = context.user_data.get("split_video_session")

    if not session or session.get("type") != "video":
        # Not a video split session, ignore
        return

    correlation_id = session.get("correlation_id", "unknown")
    logger.info(f"[{correlation_id}] Start time input received from user {user_id}")

    try:
        start_time = float(update.message.text.strip())
    except (ValueError, AttributeError):
        await update.message.reply_text(
            "Por favor envía un número válido (ej: 30 o 30.5)"
        )
        return

    duration = session.get("duration", 0)
    if start_time < 0:
        await update.message.reply_text(
            "El tiempo de inicio no puede ser negativo."
        )
        return

    if start_time >= duration:
        await update.message.reply_text(
            f"El tiempo de inicio debe ser menor a la duración del video ({duration}s)."
        )
        return

    # Store start time
    session["start_time"] = start_time
    session["state"] = SPLIT_WAITING_END_TIME

    remaining = duration - start_time
    await update.message.reply_text(
        f"✅ Tiempo de inicio: *{start_time}s*\n\n"
        f"Ahora envía el tiempo *final* en segundos.\n"
        f"Debe ser mayor a {start_time}s y menor a {duration}s.\n\n"
        f"Tiempo máximo disponible: *{remaining:.1f}s*",
        parse_mode="Markdown"
    )


async def handle_video_split_end_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle end time input for video split and process the cut.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    session = context.user_data.get("split_video_session")

    if not session or session.get("type") != "video":
        return

    correlation_id = session.get("correlation_id", "unknown")
    logger.info(f"[{correlation_id}] End time input received from user {user_id}")

    try:
        end_time = float(update.message.text.strip())
    except (ValueError, AttributeError):
        await update.message.reply_text(
            "Por favor envía un número válido (ej: 60 o 90.5)"
        )
        return

    start_time = session.get("start_time", 0)
    duration = session.get("duration", 0)

    if end_time <= start_time:
        await update.message.reply_text(
            f"El tiempo final debe ser mayor al tiempo de inicio ({start_time}s)."
        )
        return

    if end_time > duration:
        await update.message.reply_text(
            f"El tiempo final no puede exceder la duración del video ({duration}s)."
        )
        return

    segment_duration = end_time - start_time
    if segment_duration < 1:
        await update.message.reply_text(
            "La duración mínima del segmento es 1 segundo."
        )
        return

    # Store end time and proceed to cut
    session["end_time"] = end_time
    session["state"] = SPLIT_CONFIRMING

    # Send processing message
    processing_message = await update.message.reply_text(
        f"✂️ Extrayendo segmento de {start_time}s a {end_time}s...\n"
        f"Duración: {segment_duration:.1f}s"
    )

    with TempManager() as temp_mgr:
        try:
            input_path = session["input_path"]
            output_dir = temp_mgr.get_temp_path(f"split_output_{correlation_id}")
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            splitter = VideoSplitter(str(input_path), str(output_dir))
            output_path = splitter.split_by_time_range(start_time, end_time)

            # Send video segment
            await update.message.reply_video(
                video=open(output_path, "rb"),
                caption=f"Segmento extraído ({start_time}s - {end_time}s)"
            )

            await processing_message.delete()
            logger.info(f"[{correlation_id}] Video segment sent successfully")

        except Exception as e:
            logger.error(f"[{correlation_id}] Error splitting video: {e}")
            await processing_message.delete()
            await update.message.reply_text(
                "Error al extraer el segmento. Intenta de nuevo."
            )
        finally:
            context.user_data.pop("split_video_session", None)


async def handle_audio_split_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start interactive audio split process.

    Downloads the audio, gets its duration, and asks user for start time.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    query = update.callback_query
    correlation_id = str(uuid.uuid4())[:8]

    # Respond to callback immediately to avoid timeout
    await query.answer()

    # Get file_id from context (set by handle_audio_menu_callback)
    file_id = context.user_data.get("audio_menu_file_id")
    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for audio split")
        await query.edit_message_text("Error: no se encontró el audio.")
        return

    logger.info(f"[{correlation_id}] Audio split started by user {user_id}")

    # Store file info in context
    context.user_data["split_audio_session"] = {
        "file_id": file_id,
        "correlation_id": correlation_id,
        "state": SPLIT_WAITING_START_TIME,
        "type": "audio",
    }

    await query.edit_message_text(
        "✂️ *Dividir Audio*\n\n"
        "⏳ Descargando y analizando el audio...\n"
        "Este proceso puede tomar unos segundos.",
        parse_mode="Markdown"
    )

    # Download audio and get duration
    temp_mgr = TempManager()
    try:
        input_filename = f"split_audio_{user_id}_{correlation_id}.mp3"
        input_path = temp_mgr.get_temp_path(input_filename)

        # Download audio
        file = await context.bot.get_file(file_id)
        await _download_with_retry(file, input_path, correlation_id=correlation_id)

        # Get duration
        splitter = AudioSplitter(str(input_path), str(temp_mgr.get_temp_path("output")))
        duration = splitter.get_audio_duration()

        # Store in session and keep temp_mgr reference
        context.user_data["split_audio_session"]["duration"] = duration
        context.user_data["split_audio_session"]["input_path"] = input_path
        context.user_data["split_audio_session"]["temp_mgr"] = temp_mgr

        minutes = int(duration // 60)
        seconds = int(duration % 60)

        await query.edit_message_text(
            f"✂️ *Dividir Audio*\n\n"
            f"📊 Duración del audio: *{minutes}m {seconds}s*\n\n"
            f"Envía el tiempo de *inicio* en segundos (ej: 30 para 30 segundos).\n\n"
            f"Puede ser un número decimal (ej: 30.5)",
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.error(f"[{correlation_id}] Error preparing audio split: {e}")
        await query.edit_message_text(
            "Error al preparar el audio. Intenta de nuevo."
        )
        context.user_data.pop("split_audio_session", None)
        temp_mgr.cleanup()


async def handle_audio_split_start_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle start time input for audio split.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    session = context.user_data.get("split_audio_session")

    if not session:
        return

    correlation_id = session.get("correlation_id", "unknown")
    logger.info(f"[{correlation_id}] Audio start time input received from user {user_id}")

    try:
        start_time = float(update.message.text.strip())
    except (ValueError, AttributeError):
        await update.message.reply_text(
            "Por favor envía un número válido (ej: 30 o 30.5)"
        )
        return

    duration = session.get("duration", 0)
    if start_time < 0:
        await update.message.reply_text(
            "El tiempo de inicio no puede ser negativo."
        )
        return

    if start_time >= duration:
        await update.message.reply_text(
            f"El tiempo de inicio debe ser menor a la duración del audio ({duration}s)."
        )
        return

    # Store start time
    session["start_time"] = start_time
    session["state"] = SPLIT_WAITING_END_TIME

    remaining = duration - start_time
    await update.message.reply_text(
        f"✅ Tiempo de inicio: *{start_time}s*\n\n"
        f"Ahora envía el tiempo *final* en segundos.\n"
        f"Debe ser mayor a {start_time}s y menor a {duration}s.\n\n"
        f"Tiempo máximo disponible: *{remaining:.1f}s*",
        parse_mode="Markdown"
    )


async def handle_audio_split_end_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle end time input for audio split and process the cut.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    session = context.user_data.get("split_audio_session")

    if not session:
        return

    correlation_id = session.get("correlation_id", "unknown")
    logger.info(f"[{correlation_id}] Audio end time input received from user {user_id}")

    try:
        end_time = float(update.message.text.strip())
    except (ValueError, AttributeError):
        await update.message.reply_text(
            "Por favor envía un número válido (ej: 60 o 90.5)"
        )
        return

    start_time = session.get("start_time", 0)
    duration = session.get("duration", 0)

    if end_time <= start_time:
        await update.message.reply_text(
            f"El tiempo final debe ser mayor al tiempo de inicio ({start_time}s)."
        )
        return

    if end_time > duration:
        await update.message.reply_text(
            f"El tiempo final no puede exceder la duración del audio ({duration}s)."
        )
        return

    segment_duration = end_time - start_time
    if segment_duration < 1:
        await update.message.reply_text(
            "La duración mínima del segmento es 1 segundo."
        )
        return

    # Store end time and proceed to cut
    session["end_time"] = end_time
    session["state"] = SPLIT_CONFIRMING

    # Send processing message
    processing_message = await update.message.reply_text(
        f"✂️ Extrayendo segmento de {start_time}s a {end_time}s...\n"
        f"Duración: {segment_duration:.1f}s"
    )

    with TempManager() as temp_mgr:
        try:
            input_path = session["input_path"]
            output_dir = temp_mgr.get_temp_path(f"split_output_{correlation_id}")
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            splitter = AudioSplitter(str(input_path), str(output_dir))
            output_path = splitter.split_by_time_range(start_time, end_time)

            # Send audio segment
            await update.message.reply_audio(
                audio=open(output_path, "rb"),
                caption=f"Segmento extraído ({start_time}s - {end_time}s)"
            )

            await processing_message.delete()
            logger.info(f"[{correlation_id}] Audio segment sent successfully")

        except Exception as e:
            logger.error(f"[{correlation_id}] Error splitting audio: {e}")
            await processing_message.delete()
            await update.message.reply_text(
                "Error al extraer el segmento. Intenta de nuevo."
            )
        finally:
            context.user_data.pop("split_audio_session", None)


async def handle_split_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages during active split or screenshot sessions.

    Routes to appropriate handler based on active session type.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    # Check for video split session
    if context.user_data.get("split_video_session"):
        session = context.user_data["split_video_session"]
        if session.get("state") == SPLIT_WAITING_START_TIME:
            await handle_video_split_start_time(update, context)
        elif session.get("state") == SPLIT_WAITING_END_TIME:
            await handle_video_split_end_time(update, context)
        return

    # Check for audio split session
    if context.user_data.get("split_audio_session"):
        session = context.user_data["split_audio_session"]
        if session.get("state") == SPLIT_WAITING_START_TIME:
            await handle_audio_split_start_time(update, context)
        elif session.get("state") == SPLIT_WAITING_END_TIME:
            await handle_audio_split_end_time(update, context)
        return

    # Check for screenshot session (automatic count or manual times)
    screenshot_state = context.user_data.get("screenshot_state")
    if screenshot_state in ("waiting_count", "waiting_times"):
        await handle_screenshot_text_input(update, context)
        return


async def handle_screenshot_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text input during screenshot sessions.

    For automatic mode: accepts a number for count.
    For manual mode: accepts timestamps in various formats.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = context.user_data.get("screenshot_correlation_id", str(uuid.uuid4())[:8])
    screenshot_state = context.user_data.get("screenshot_state")
    mode = context.user_data.get("screenshot_mode")

    text = update.message.text.strip()
    logger.info(f"[{correlation_id}] Screenshot text input: '{text}' (state={screenshot_state}, mode={mode})")

    if screenshot_state == "waiting_count" and mode == "auto":
        # Parse count from text
        try:
            count = int(text)
            if count < 1 or count > 100:
                await update.message.reply_text(
                    "❌ Cantidad inválida. Ingresa un número entre 1 y 100."
                )
                return

            # Process screenshots
            await _process_screenshots(update, context, count)

        except ValueError:
            await update.message.reply_text(
                "❌ Número inválido. Por favor ingresa un número entero (ej: 12)"
            )

    elif screenshot_state == "waiting_times" and mode == "manual":
        # Parse timestamps from text
        timestamps, error = await _parse_screenshot_times(text)

        if error:
            await update.message.reply_text(f"❌ {error}")
            return

        # Store timestamps and process
        context.user_data["screenshot_timestamps"] = timestamps

        # Show confirmation and process
        count = len(timestamps)
        await update.message.reply_text(f"✓ Generando {count} captura(s) en los tiempos especificados...")

        # Process screenshots with stored timestamps
        await _process_screenshots(update, context, None)

    else:
        # Not in a screenshot session, ignore
        logger.debug(f"[{correlation_id}] Ignoring text input, not in screenshot session")
        return


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages by converting them to MP3 format with automatic processing.

    Flow: voice note → MP3 → normalize to -16 LUFS (podcast) → bass boost (intensity 4) → send processed audio.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Voice message received from user {user_id} - running podcast pipeline")

    # Get voice from message
    voice = update.message.voice
    if not voice:
        logger.warning(f"[{correlation_id}] No voice found in message from user {user_id}")
        await update.message.reply_text("No encontré una nota de voz en tu mensaje.")
        return

    # Validate file size before downloading
    if voice.file_size:
        logger.debug(f"[{correlation_id}] Voice file size: {voice.file_size} bytes")
        is_valid, error_msg = validate_file_size(voice.file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Send "processing" message with cancel button
    keyboard = [[InlineKeyboardButton("❌ Cancelar", callback_data=f"voice_cancel:{correlation_id}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    processing_message = None
    try:
        processing_message = await update.message.reply_text(
            "🎙️ Procesando nota de voz...\n\n"
            "1️⃣ Convirtiendo a MP3...\n"
            "⏳ Espera por favor",
            reply_markup=reply_markup
        )
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not send processing message to user {user_id}: {e}")

    # Store correlation_id for cancel handling
    context.user_data["voice_pipeline_correlation_id"] = correlation_id

    # Use TempManager as context manager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"voice_{user_id}_{voice.file_unique_id}.oga"
            mp3_filename = f"voice_{user_id}_{voice.file_unique_id}.mp3"
            normalized_filename = f"normalized_{user_id}_{correlation_id}.mp3"
            output_filename = f"podcast_{user_id}_{correlation_id}.mp3"

            input_path = temp_mgr.get_temp_path(input_filename)
            mp3_path = temp_mgr.get_temp_path(mp3_filename)
            normalized_path = temp_mgr.get_temp_path(normalized_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download voice file
            logger.info(f"[{correlation_id}] Downloading voice from user {user_id}")
            try:
                file = await voice.get_file()
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download voice for user {user_id}: {e}")
                raise DownloadError("No pude descargar la nota de voz") from e

            # Validate audio integrity after download
            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Audio validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space before processing
            voice_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(voice_size_mb)) * 3  # 3x for pipeline
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"[{correlation_id}] Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Step 1: Convert to MP3
            logger.info(f"[{correlation_id}] Converting voice to MP3 for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                converter = VoiceToMp3Converter(str(input_path), str(mp3_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, converter.process),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not success:
                    logger.error(f"[{correlation_id}] Voice to MP3 conversion failed for user {user_id}")
                    raise VoiceToMp3Error("No pude convertir la nota de voz a MP3")

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Voice to MP3 conversion timed out for user {user_id}")
                raise ProcessingTimeoutError("La conversión tardó demasiado") from e

            # Update progress message
            try:
                await processing_message.edit_text(
                    "🎙️ Procesando nota de voz...\n\n"
                    "✅ Convertido a MP3\n"
                    "2️⃣ Normalizando a -16 LUFS (podcast)...\n"
                    "⏳ Espera por favor",
                    reply_markup=reply_markup
                )
            except Exception as e:
                logger.warning(f"[{correlation_id}] Could not update message: {e}")

            # Step 2: Normalize to -16 LUFS (podcast preset)
            logger.info(f"[{correlation_id}] Normalizing to -16 LUFS (podcast) for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                effects = AudioEffects(str(mp3_path), str(normalized_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, effects.normalize, -16.0),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not success:
                    logger.error(f"[{correlation_id}] Normalization failed for user {user_id}")
                    raise AudioEffectsError("No pude normalizar el audio")

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Normalization timed out for user {user_id}")
                raise ProcessingTimeoutError("La normalización tardó demasiado") from e

            # Update progress message
            try:
                await processing_message.edit_text(
                    "🎙️ Procesando nota de voz...\n\n"
                    "✅ Convertido a MP3\n"
                    "✅ Normalizado a -16 LUFS\n"
                    "3️⃣ Aplicando bass boost (intensidad 4)...\n"
                    "⏳ Espera por favor",
                    reply_markup=reply_markup
                )
            except Exception as e:
                logger.warning(f"[{correlation_id}] Could not update message: {e}")

            # Step 3: Apply bass boost (intensity 4)
            logger.info(f"[{correlation_id}] Applying bass boost (intensity 4) for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                enhancer = AudioEnhancer(str(normalized_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: enhancer.bass_boost(4.0)),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not success:
                    logger.error(f"[{correlation_id}] Bass boost failed for user {user_id}")
                    raise AudioEnhancementError("No pude aplicar el bass boost")

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Bass boost timed out for user {user_id}")
                raise ProcessingTimeoutError("El bass boost tardó demasiado") from e

            # Update progress message
            try:
                await processing_message.edit_text(
                    "🎙️ Procesando nota de voz...\n\n"
                    "✅ Convertido a MP3\n"
                    "✅ Normalizado a -16 LUFS\n"
                    "✅ Bass boost aplicado\n"
                    "📤 Enviando archivo...",
                    reply_markup=reply_markup
                )
            except Exception as e:
                logger.warning(f"[{correlation_id}] Could not update message: {e}")

            # Send as audio file with metadata
            logger.info(f"[{correlation_id}] Sending processed audio to user {user_id}")
            try:
                with open(output_path, "rb") as audio_file:
                    await update.message.reply_audio(
                        audio=audio_file,
                        title="Nota de voz procesada",
                        performer="Podcast Pipeline",
                        filename=f"voice_{user_id}_processed.mp3"
                    )
                logger.info(f"[{correlation_id}] Processed audio sent successfully to user {user_id}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to send processed audio to user {user_id}: {e}")
                raise

            # Clean up correlation_id
            context.user_data.pop("voice_pipeline_correlation_id", None)

            # Delete processing message on success
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Could not delete processing message: {e}")

        except (DownloadError, ValidationError, VoiceToMp3Error, AudioEffectsError, AudioEnhancementError, ProcessingTimeoutError) as e:
            # Handle known processing errors
            logger.error(f"[{correlation_id}] Processing error: {e}")
            await handle_processing_error(update, e, user_id)

            # Clean up correlation_id
            context.user_data.pop("voice_pipeline_correlation_id", None)

            # Delete processing message on error
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Could not delete processing message: {e}")

        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"[{correlation_id}] Unexpected error processing voice for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)

            # Delete processing message on error
            if processing_message:
                try:
                    await processing_message.delete()
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Could not delete processing message: {e}")

        # TempManager cleanup happens automatically on context exit
        logger.debug(f"[{correlation_id}] Cleanup completed for user {user_id}")


async def handle_convert_audio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /convert_audio command to convert audio to different format.

    Usage: /convert_audio (when replying to an audio or with audio attached)
    Shows inline keyboard with format options for user to select.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Convert audio command received from user {user_id}")

    # Get audio from message or reply
    audio, is_reply = await _get_audio_from_message(update)

    if not audio:
        await update.message.reply_text(
            "Envía /convert_audio respondiendo a un archivo de audio o adjunta el audio al mensaje."
        )
        return

    # Validate file size before downloading
    if audio.file_size:
        is_valid, error_msg = validate_file_size(audio.file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Store file_id in context for later retrieval
    context.user_data["convert_audio_file_id"] = audio.file_id
    context.user_data["convert_audio_correlation_id"] = correlation_id

    # Create inline keyboard with format options (3 + 2 layout)
    keyboard = [
        [
            InlineKeyboardButton("MP3", callback_data="format:mp3"),
            InlineKeyboardButton("WAV", callback_data="format:wav"),
            InlineKeyboardButton("OGG", callback_data="format:ogg"),
        ],
        [
            InlineKeyboardButton("AAC", callback_data="format:aac"),
            InlineKeyboardButton("FLAC", callback_data="format:flac"),
        ],
        [InlineKeyboardButton("❌ Cancelar", callback_data="cancel")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Selecciona el formato de salida:",
        reply_markup=reply_markup
    )
    logger.info(f"[{correlation_id}] Format selection keyboard sent to user {user_id}")


async def handle_format_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle format selection callback from inline keyboard.

    Downloads the audio, converts it to selected format, and sends back.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Extract format from callback data (e.g., "format:mp3" -> "mp3")
    callback_data = query.data
    if not callback_data.startswith("format:"):
        logger.warning(f"Invalid callback data received: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    output_format = callback_data.split(":")[1]

    # Retrieve file_id from context
    file_id = context.user_data.get("convert_audio_file_id")
    correlation_id = context.user_data.get("convert_audio_correlation_id", str(uuid.uuid4())[:8])

    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for user {user_id}")
        await query.edit_message_text("Error: no se encontró el archivo de audio. Intenta de nuevo.")
        return

    logger.info(f"[{correlation_id}] Format {output_format} selected by user {user_id}")

    # Update message to show processing
    try:
        await query.edit_message_text(f"Convirtiendo a {output_format.upper()}...")
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not update message: {e}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_{user_id}_{correlation_id}.audio"
            output_filename = f"converted_{user_id}_{correlation_id}.{output_format}"

            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download audio file
            logger.info(f"[{correlation_id}] Downloading audio from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Audio downloaded to {input_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download audio for user {user_id}: {e}")
                raise DownloadError("No pude descargar el audio") from e

            # Validate audio integrity after download
            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Audio validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Detect input format
            input_format = detect_audio_format(str(input_path))
            if input_format:
                logger.info(f"[{correlation_id}] Detected input format: {input_format}")
                # Check if input format equals output format
                if input_format == output_format:
                    await query.edit_message_text(
                        f"El archivo ya está en formato {output_format.upper()}. No es necesario convertir."
                    )
                    return
            else:
                logger.warning(f"[{correlation_id}] Could not detect input format for user {user_id}")

            # Check disk space before processing
            audio_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(audio_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"[{correlation_id}] Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Convert audio with timeout
            logger.info(f"[{correlation_id}] Converting audio to {output_format} for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                converter = AudioFormatConverter(str(input_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, converter.convert, output_format),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not success:
                    logger.error(f"[{correlation_id}] Audio format conversion failed for user {user_id}")
                    raise AudioFormatConversionError(f"No pude convertir el audio a {output_format.upper()}")

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Audio conversion timed out for user {user_id}")
                raise ProcessingTimeoutError("La conversión tardó demasiado") from e

            # Send converted audio
            logger.info(f"[{correlation_id}] Sending converted audio to user {user_id}")
            try:
                with open(output_path, "rb") as audio_file:
                    await context.bot.send_audio(
                        chat_id=update.effective_chat.id,
                        audio=audio_file,
                        filename=f"converted.{output_format}",
                        title=f"Audio convertido a {output_format.upper()}"
                    )
                logger.info(f"[{correlation_id}] Converted audio sent successfully to user {user_id}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to send converted audio to user {user_id}: {e}")
                raise

            # Update message on success
            try:
                await query.edit_message_text(f"Audio convertido a {output_format.upper()} exitosamente.")
            except Exception as e:
                logger.warning(f"[{correlation_id}] Could not update final message: {e}")

            # Clean up user_data
            context.user_data.pop("convert_audio_file_id", None)
            context.user_data.pop("convert_audio_correlation_id", None)

        except (DownloadError, ValidationError, AudioFormatConversionError, ProcessingTimeoutError) as e:
            # Handle known processing errors
            logger.error(f"[{correlation_id}] Processing error: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text(f"Error: {str(e)}")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"[{correlation_id}] Unexpected error converting audio for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        # TempManager cleanup happens automatically on context exit
        logger.debug(f"[{correlation_id}] Cleanup completed for user {user_id}")


async def handle_bass_boost_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /bass_boost command to apply bass boost enhancement.

    Usage: /bass_boost (when replying to an audio or with audio attached)
    Shows inline keyboard with intensity options (1-10) for user to select.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Bass boost command received from user {user_id}")

    # Get audio from message or reply
    audio, is_reply = await _get_audio_from_message(update)

    if not audio:
        await update.message.reply_text(
            "Envía /bass_boost respondiendo a un archivo de audio o adjunta el audio al mensaje."
        )
        return

    # Validate file size before downloading
    if audio.file_size:
        is_valid, error_msg = validate_file_size(audio.file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Store file_id in context for later retrieval
    context.user_data["enhance_audio_file_id"] = audio.file_id
    context.user_data["enhance_audio_correlation_id"] = correlation_id
    context.user_data["enhance_type"] = "bass"

    # Create inline keyboard with intensity options (5 + 5 layout)
    keyboard = [
        [
            InlineKeyboardButton("1", callback_data="bass:1"),
            InlineKeyboardButton("2", callback_data="bass:2"),
            InlineKeyboardButton("3", callback_data="bass:3"),
            InlineKeyboardButton("4", callback_data="bass:4"),
            InlineKeyboardButton("5", callback_data="bass:5"),
        ],
        [
            InlineKeyboardButton("6", callback_data="bass:6"),
            InlineKeyboardButton("7", callback_data="bass:7"),
            InlineKeyboardButton("8", callback_data="bass:8"),
            InlineKeyboardButton("9", callback_data="bass:9"),
            InlineKeyboardButton("10", callback_data="bass:10"),
        ],
        [InlineKeyboardButton("❌ Cancelar", callback_data="cancel")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Selecciona la intensidad del bass boost (1-10):",
        reply_markup=reply_markup
    )
    logger.info(f"[{correlation_id}] Intensity selection keyboard sent to user {user_id}")


async def handle_treble_boost_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /treble_boost command to apply treble boost enhancement.

    Usage: /treble_boost (when replying to an audio or with audio attached)
    Shows inline keyboard with intensity options (1-10) for user to select.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Treble boost command received from user {user_id}")

    # Get audio from message or reply
    audio, is_reply = await _get_audio_from_message(update)

    if not audio:
        await update.message.reply_text(
            "Envía /treble_boost respondiendo a un archivo de audio o adjunta el audio al mensaje."
        )
        return

    # Validate file size before downloading
    if audio.file_size:
        is_valid, error_msg = validate_file_size(audio.file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Store file_id in context for later retrieval
    context.user_data["enhance_audio_file_id"] = audio.file_id
    context.user_data["enhance_audio_correlation_id"] = correlation_id
    context.user_data["enhance_type"] = "treble"

    # Create inline keyboard with intensity options (5 + 5 layout)
    keyboard = [
        [
            InlineKeyboardButton("1", callback_data="treble:1"),
            InlineKeyboardButton("2", callback_data="treble:2"),
            InlineKeyboardButton("3", callback_data="treble:3"),
            InlineKeyboardButton("4", callback_data="treble:4"),
            InlineKeyboardButton("5", callback_data="treble:5"),
        ],
        [
            InlineKeyboardButton("6", callback_data="treble:6"),
            InlineKeyboardButton("7", callback_data="treble:7"),
            InlineKeyboardButton("8", callback_data="treble:8"),
            InlineKeyboardButton("9", callback_data="treble:9"),
            InlineKeyboardButton("10", callback_data="treble:10"),
        ],
        [InlineKeyboardButton("❌ Cancelar", callback_data="cancel")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Selecciona la intensidad del treble boost (1-10):",
        reply_markup=reply_markup
    )
    logger.info(f"[{correlation_id}] Intensity selection keyboard sent to user {user_id}")


async def handle_intensity_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle intensity selection callback from inline keyboard.

    Downloads the audio, applies the selected enhancement (bass or treble),
    and sends back the enhanced audio.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Parse callback data (e.g., "bass:5" or "treble:8")
    callback_data = query.data
    if not callback_data or ":" not in callback_data:
        logger.warning(f"Invalid callback data received: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    parts = callback_data.split(":")
    if len(parts) != 2:
        logger.warning(f"Invalid callback data format: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    enhance_type = parts[0]
    try:
        intensity = int(parts[1])
    except ValueError:
        logger.warning(f"Invalid intensity value: {parts[1]}")
        await query.edit_message_text("Error: intensidad inválida.")
        return

    if enhance_type not in ("bass", "treble"):
        logger.warning(f"Invalid enhancement type: {enhance_type}")
        await query.edit_message_text("Error: tipo de mejora inválido.")
        return

    if intensity < 1 or intensity > 10:
        logger.warning(f"Invalid intensity range: {intensity}")
        await query.edit_message_text("Error: intensidad debe estar entre 1 y 10.")
        return

    # Retrieve file_id from context
    file_id = context.user_data.get("enhance_audio_file_id")
    correlation_id = context.user_data.get("enhance_audio_correlation_id", str(uuid.uuid4())[:8])
    stored_enhance_type = context.user_data.get("enhance_type")

    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for user {user_id}")
        await query.edit_message_text("Error: no se encontró el archivo de audio. Intenta de nuevo.")
        return

    # Verify enhance_type matches stored type
    if stored_enhance_type and stored_enhance_type != enhance_type:
        logger.warning(f"[{correlation_id}] Mismatch: stored={stored_enhance_type}, callback={enhance_type}")

    effect_name = "bass" if enhance_type == "bass" else "treble"
    logger.info(f"[{correlation_id}] {effect_name.capitalize()} boost intensity {intensity} selected by user {user_id}")

    # Update message to show processing
    try:
        await query.edit_message_text(f"Aplicando {effect_name} boost (intensidad {intensity})...")
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not update message: {e}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_{user_id}_{correlation_id}.audio"
            output_filename = f"enhanced_{user_id}_{correlation_id}.mp3"

            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download audio file
            logger.info(f"[{correlation_id}] Downloading audio from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Audio downloaded to {input_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download audio for user {user_id}: {e}")
                raise DownloadError("No pude descargar el audio") from e

            # Validate audio integrity after download
            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Audio validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space before processing
            audio_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(audio_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"[{correlation_id}] Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Apply enhancement with timeout
            logger.info(f"[{correlation_id}] Applying {effect_name} boost (intensity {intensity}) for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                enhancer = AudioEnhancer(str(input_path), str(output_path))

                if enhance_type == "bass":
                    success = await asyncio.wait_for(
                        loop.run_in_executor(None, enhancer.bass_boost, intensity),
                        timeout=config.PROCESSING_TIMEOUT
                    )
                else:  # treble
                    success = await asyncio.wait_for(
                        loop.run_in_executor(None, enhancer.treble_boost, intensity),
                        timeout=config.PROCESSING_TIMEOUT
                    )

                if not success:
                    logger.error(f"[{correlation_id}] Audio enhancement failed for user {user_id}")
                    raise AudioEnhancementError(f"No pude aplicar el {effect_name} boost")

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Audio enhancement timed out for user {user_id}")
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            # Send enhanced audio
            logger.info(f"[{correlation_id}] Sending enhanced audio to user {user_id}")
            try:
                with open(output_path, "rb") as audio_file:
                    await context.bot.send_audio(
                        chat_id=update.effective_chat.id,
                        audio=audio_file,
                        filename=f"enhanced_{effect_name}.mp3",
                        title=f"Audio mejorado ({effect_name.capitalize()} Boost)"
                    )
                logger.info(f"[{correlation_id}] Enhanced audio sent successfully to user {user_id}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to send enhanced audio to user {user_id}: {e}")
                raise

            # Update message on success
            try:
                await query.edit_message_text(
                    f"¡Listo! Audio mejorado con {effect_name} boost (intensidad {intensity}/10)."
                )
            except Exception as e:
                logger.warning(f"[{correlation_id}] Could not update final message: {e}")

            # Clean up user_data
            context.user_data.pop("enhance_audio_file_id", None)
            context.user_data.pop("enhance_audio_correlation_id", None)
            context.user_data.pop("enhance_type", None)

        except (DownloadError, ValidationError, AudioEnhancementError, ProcessingTimeoutError) as e:
            # Handle known processing errors
            logger.error(f"[{correlation_id}] Processing error: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text(f"Error: {str(e)}")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"[{correlation_id}] Unexpected error enhancing audio for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        # TempManager cleanup happens automatically on context exit
        logger.debug(f"[{correlation_id}] Cleanup completed for user {user_id}")


# =============================================================================
# Equalizer Handlers
# =============================================================================


def _get_video_menu_keyboard() -> InlineKeyboardMarkup:
    """Generate inline keyboard for video menu options."""
    keyboard = [
        [
            InlineKeyboardButton("Nota de Video", callback_data="video_action:videonote"),
            InlineKeyboardButton("Extraer Audio", callback_data="video_action:extract_audio"),
        ],
        [
            InlineKeyboardButton("Convertir Formato", callback_data="video_action:convert"),
            InlineKeyboardButton("Dividir Video", callback_data="video_action:split"),
        ],
        [
            InlineKeyboardButton("Unir Videos", callback_data="video_action:join"),
            InlineKeyboardButton("Merge con Audio", callback_data="video_action:merge_audio"),
        ],
        [
            InlineKeyboardButton("📸 Capturas de Pantalla", callback_data="video_action:screenshots"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_video_format_keyboard() -> InlineKeyboardMarkup:
    """Generate inline keyboard for video format selection."""
    keyboard = [
        [
            InlineKeyboardButton("MP4", callback_data="video_format:mp4"),
            InlineKeyboardButton("AVI", callback_data="video_format:avi"),
            InlineKeyboardButton("MOV", callback_data="video_format:mov"),
        ],
        [
            InlineKeyboardButton("MKV", callback_data="video_format:mkv"),
            InlineKeyboardButton("WEBM", callback_data="video_format:webm"),
        ],
        [
            InlineKeyboardButton("← Volver", callback_data="back:video"),
            InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_video_audio_format_keyboard() -> InlineKeyboardMarkup:
    """Generate inline keyboard for audio extraction format selection."""
    keyboard = [
        [
            InlineKeyboardButton("MP3", callback_data="video_audio_format:mp3"),
            InlineKeyboardButton("AAC", callback_data="video_audio_format:aac"),
        ],
        [
            InlineKeyboardButton("WAV", callback_data="video_audio_format:wav"),
            InlineKeyboardButton("OGG", callback_data="video_audio_format:ogg"),
        ],
        [
            InlineKeyboardButton("← Volver", callback_data="back:video"),
            InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_screenshot_mode_keyboard() -> InlineKeyboardMarkup:
    """Generate inline keyboard for screenshot mode selection."""
    keyboard = [
        [
            InlineKeyboardButton("🔢 Automático", callback_data="screenshot:auto"),
            InlineKeyboardButton("✏️ Manual", callback_data="screenshot:manual"),
        ],
        [
            InlineKeyboardButton("← Volver", callback_data="back:video"),
            InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_screenshot_count_keyboard() -> InlineKeyboardMarkup:
    """Generate inline keyboard for automatic screenshot count selection."""
    keyboard = [
        [
            InlineKeyboardButton("5", callback_data="screenshot_count:5"),
            InlineKeyboardButton("10", callback_data="screenshot_count:10"),
            InlineKeyboardButton("15", callback_data="screenshot_count:15"),
        ],
        [
            InlineKeyboardButton("20", callback_data="screenshot_count:20"),
            InlineKeyboardButton("25", callback_data="screenshot_count:25"),
            InlineKeyboardButton("30", callback_data="screenshot_count:30"),
        ],
        [
            InlineKeyboardButton("← Volver", callback_data="screenshot:back_to_mode"),
            InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_screenshot_manual_nav_keyboard() -> InlineKeyboardMarkup:
    """Generate inline keyboard for manual screenshot time input navigation."""
    keyboard = [
        [
            InlineKeyboardButton("← Volver", callback_data="screenshot:back_to_mode"),
            InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_equalizer_keyboard(bass: int, mid: int, treble: int) -> InlineKeyboardMarkup:
    """Generate inline keyboard for 3-band equalizer.

    Args:
        bass: Current bass value (-10 to +10)
        mid: Current mid value (-10 to +10)
        treble: Current treble value (-10 to +10)

    Returns:
        InlineKeyboardMarkup with equalizer controls
    """
    # Format values with sign for positive numbers
    bass_str = f"{bass:+d}" if bass != 0 else "0"
    mid_str = f"{mid:+d}" if mid != 0 else "0"
    treble_str = f"{treble:+d}" if treble != 0 else "0"

    keyboard = [
        # Bass row
        [
            InlineKeyboardButton("Bass", callback_data="eq_noop"),
            InlineKeyboardButton("-", callback_data="eq_bass_down"),
            InlineKeyboardButton(bass_str, callback_data="eq_noop"),
            InlineKeyboardButton("+", callback_data="eq_bass_up"),
        ],
        # Mid row
        [
            InlineKeyboardButton("Mid", callback_data="eq_noop"),
            InlineKeyboardButton("-", callback_data="eq_mid_down"),
            InlineKeyboardButton(mid_str, callback_data="eq_noop"),
            InlineKeyboardButton("+", callback_data="eq_mid_up"),
        ],
        # Treble row
        [
            InlineKeyboardButton("Treble", callback_data="eq_noop"),
            InlineKeyboardButton("-", callback_data="eq_treble_down"),
            InlineKeyboardButton(treble_str, callback_data="eq_noop"),
            InlineKeyboardButton("+", callback_data="eq_treble_up"),
        ],
        # Reset and Apply row
        [
            InlineKeyboardButton("Reset", callback_data="eq_reset_all"),
            InlineKeyboardButton("Aplicar", callback_data="eq_apply"),
            InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


async def handle_equalize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /equalize command to show 3-band equalizer interface.

    Usage: /equalize (when replying to an audio or with audio attached)

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Equalize command received from user {user_id}")

    # Get audio from message or reply
    audio, is_reply = await _get_audio_from_message(update)

    if not audio:
        await update.message.reply_text(
            "Envía /equalize respondiendo a un archivo de audio o adjunta el audio al mensaje."
        )
        return

    # Validate file size before downloading
    if audio.file_size:
        is_valid, error_msg = validate_file_size(audio.file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Initialize equalizer state in context.user_data
    context.user_data["eq_file_id"] = audio.file_id
    context.user_data["eq_correlation_id"] = correlation_id
    context.user_data["eq_bass"] = 0
    context.user_data["eq_mid"] = 0
    context.user_data["eq_treble"] = 0

    # Create inline keyboard
    reply_markup = _get_equalizer_keyboard(0, 0, 0)

    await update.message.reply_text(
        "Ecualizador de 3 bandas:\n"
        "🎵 Bass: 0\n"
        "🎵 Mid: 0\n"
        "🎵 Treble: 0\n\n"
        "Ajusta cada banda y presiona Aplicar.",
        reply_markup=reply_markup
    )
    logger.info(f"[{correlation_id}] Equalizer interface sent to user {user_id}")


async def handle_equalizer_adjustment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle equalizer adjustment callbacks from inline keyboard.

    Handles up/down adjustments for bass/mid/treble, reset, and apply.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Handle noop callbacks (display buttons)
    if callback_data == "eq_noop":
        return

    # Get current values from context
    bass = context.user_data.get("eq_bass", 0)
    mid = context.user_data.get("eq_mid", 0)
    treble = context.user_data.get("eq_treble", 0)
    correlation_id = context.user_data.get("eq_correlation_id", str(uuid.uuid4())[:8])

    # Process callback
    if callback_data == "eq_apply":
        await _handle_equalizer_apply(update, context, bass, mid, treble)
        return

    # Step size for adjustments
    STEP = 2
    MIN_VAL = -10
    MAX_VAL = 10

    if callback_data == "eq_bass_up":
        bass = min(MAX_VAL, bass + STEP)
    elif callback_data == "eq_bass_down":
        bass = max(MIN_VAL, bass - STEP)
    elif callback_data == "eq_mid_up":
        mid = min(MAX_VAL, mid + STEP)
    elif callback_data == "eq_mid_down":
        mid = max(MIN_VAL, mid - STEP)
    elif callback_data == "eq_treble_up":
        treble = min(MAX_VAL, treble + STEP)
    elif callback_data == "eq_treble_down":
        treble = max(MIN_VAL, treble - STEP)
    elif callback_data == "eq_reset_all":
        bass = 0
        mid = 0
        treble = 0
    else:
        logger.warning(f"[{correlation_id}] Unknown equalizer callback: {callback_data}")
        return

    # Store updated values
    context.user_data["eq_bass"] = bass
    context.user_data["eq_mid"] = mid
    context.user_data["eq_treble"] = treble

    # Format values for display
    bass_display = f"{bass:+d}" if bass != 0 else "0"
    mid_display = f"{mid:+d}" if mid != 0 else "0"
    treble_display = f"{treble:+d}" if treble != 0 else "0"

    # Update message with new values
    reply_markup = _get_equalizer_keyboard(bass, mid, treble)

    try:
        await query.edit_message_text(
            f"Ecualizador de 3 bandas:\n"
            f"🎵 Bass: {bass_display}\n"
            f"🎵 Mid: {mid_display}\n"
            f"🎵 Treble: {treble_display}\n\n"
            f"Ajusta cada banda y presiona Aplicar.",
            reply_markup=reply_markup
        )
        logger.info(f"[{correlation_id}] Equalizer updated: bass={bass}, mid={mid}, treble={treble}")
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not update equalizer message: {e}")


async def _handle_equalizer_apply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    bass: int,
    mid: int,
    treble: int
) -> None:
    """Apply equalizer settings and process audio.

    Args:
        update: Telegram update object
        context: Telegram context object
        bass: Bass gain value (-10 to +10)
        mid: Mid gain value (-10 to +10)
        treble: Treble gain value (-10 to +10)
    """
    query = update.callback_query
    user_id = update.effective_user.id
    correlation_id = context.user_data.get("eq_correlation_id", str(uuid.uuid4())[:8])

    # Check if any adjustments were made
    if bass == 0 and mid == 0 and treble == 0:
        await query.edit_message_text(
            "No has hecho ajustes. Modifica al menos una banda antes de aplicar."
        )
        return

    # Retrieve file_id from context
    file_id = context.user_data.get("eq_file_id")
    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for user {user_id}")
        await query.edit_message_text("Error: no se encontró el archivo de audio. Intenta de nuevo.")
        return

    # Format values for display
    bass_display = f"{bass:+d}" if bass != 0 else "0"
    mid_display = f"{mid:+d}" if mid != 0 else "0"
    treble_display = f"{treble:+d}" if treble != 0 else "0"

    # Update message to show processing
    try:
        await query.edit_message_text(
            f"Aplicando ecualización (Bass: {bass_display}, Mid: {mid_display}, Treble: {treble_display})..."
        )
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not update message: {e}")

    logger.info(f"[{correlation_id}] Applying equalizer: bass={bass}, mid={mid}, treble={treble}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_eq_{user_id}_{correlation_id}.audio"
            output_filename = f"equalized_{user_id}_{correlation_id}.mp3"

            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download audio file
            logger.info(f"[{correlation_id}] Downloading audio from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Audio downloaded to {input_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download audio for user {user_id}: {e}")
                raise DownloadError("No pude descargar el audio") from e

            # Validate audio integrity after download
            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Audio validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space before processing
            audio_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(audio_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"[{correlation_id}] Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Apply equalization with timeout
            logger.info(f"[{correlation_id}] Applying equalization for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                enhancer = AudioEnhancer(str(input_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, enhancer.equalize, bass, mid, treble),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not success:
                    logger.error(f"[{correlation_id}] Equalization failed for user {user_id}")
                    raise AudioEnhancementError("No pude aplicar la ecualización")

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Equalization timed out for user {user_id}")
                raise ProcessingTimeoutError("La ecualización tardó demasiado") from e

            # Send equalized audio
            logger.info(f"[{correlation_id}] Sending equalized audio to user {user_id}")
            try:
                with open(output_path, "rb") as audio_file:
                    await context.bot.send_audio(
                        chat_id=update.effective_chat.id,
                        audio=audio_file,
                        filename=f"equalized.mp3",
                        title=f"Audio ecualizado"
                    )
                logger.info(f"[{correlation_id}] Equalized audio sent successfully to user {user_id}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to send equalized audio to user {user_id}: {e}")
                raise

            # Update message on success
            try:
                await query.edit_message_text(
                    f"¡Listo! Ecualización aplicada:\n"
                    f"🎵 Bass: {bass_display}\n"
                    f"🎵 Mid: {mid_display}\n"
                    f"🎵 Treble: {treble_display}"
                )
            except Exception as e:
                logger.warning(f"[{correlation_id}] Could not update final message: {e}")

            # Clean up user_data
            context.user_data.pop("eq_file_id", None)
            context.user_data.pop("eq_correlation_id", None)
            context.user_data.pop("eq_bass", None)
            context.user_data.pop("eq_mid", None)
            context.user_data.pop("eq_treble", None)

        except (DownloadError, ValidationError, AudioEnhancementError, ProcessingTimeoutError) as e:
            # Handle known processing errors
            logger.error(f"[{correlation_id}] Processing error: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text(f"Error: {str(e)}")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"[{correlation_id}] Unexpected error applying equalizer for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        # TempManager cleanup happens automatically on context exit


async def handle_denoise_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /denoise command to apply noise reduction.

    Usage: /denoise (when replying to an audio or with audio attached)
    Shows inline keyboard with strength options (1-10) for user to select.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Denoise command received from user {user_id}")

    # Get audio from message or reply
    audio, is_reply = await _get_audio_from_message(update)

    if not audio:
        await update.message.reply_text(
            "Envía /denoise respondiendo a un archivo de audio o adjunta el audio al mensaje."
        )
        return

    # Validate file size before downloading
    if audio.file_size:
        is_valid, error_msg = validate_file_size(audio.file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Store file_id in context for later retrieval
    context.user_data["effect_audio_file_id"] = audio.file_id
    context.user_data["effect_audio_correlation_id"] = correlation_id
    context.user_data["effect_type"] = "denoise"

    # Create inline keyboard with strength options (5 + 5 layout)
    keyboard = [
        [
            InlineKeyboardButton("1", callback_data="denoise:1"),
            InlineKeyboardButton("2", callback_data="denoise:2"),
            InlineKeyboardButton("3", callback_data="denoise:3"),
            InlineKeyboardButton("4", callback_data="denoise:4"),
            InlineKeyboardButton("5", callback_data="denoise:5"),
        ],
        [
            InlineKeyboardButton("6", callback_data="denoise:6"),
            InlineKeyboardButton("7", callback_data="denoise:7"),
            InlineKeyboardButton("8", callback_data="denoise:8"),
            InlineKeyboardButton("9", callback_data="denoise:9"),
            InlineKeyboardButton("10", callback_data="denoise:10"),
        ],
        [InlineKeyboardButton("❌ Cancelar", callback_data="cancel")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Selecciona la intensidad de reducción de ruido (1-10):\n\n"
        "1 = Reducción ligera\n"
        "10 = Reducción máxima",
        reply_markup=reply_markup
    )
    logger.info(f"[{correlation_id}] Denoise strength selection keyboard sent to user {user_id}")


async def handle_compress_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /compress command to apply dynamic range compression.

    Usage: /compress (when replying to an audio or with audio attached)
    Shows inline keyboard with compression ratio presets for user to select.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Compress command received from user {user_id}")

    # Get audio from message or reply
    audio, is_reply = await _get_audio_from_message(update)

    if not audio:
        await update.message.reply_text(
            "Envía /compress respondiendo a un archivo de audio o adjunta el audio al mensaje."
        )
        return

    # Validate file size before downloading
    if audio.file_size:
        is_valid, error_msg = validate_file_size(audio.file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Store file_id in context for later retrieval
    context.user_data["effect_audio_file_id"] = audio.file_id
    context.user_data["effect_audio_correlation_id"] = correlation_id
    context.user_data["effect_type"] = "compress"

    # Create inline keyboard with compression ratio presets (2 + 2 layout)
    keyboard = [
        [
            InlineKeyboardButton("Compresión ligera", callback_data="compress:light"),
            InlineKeyboardButton("Compresión media", callback_data="compress:medium"),
        ],
        [
            InlineKeyboardButton("Compresión fuerte", callback_data="compress:heavy"),
            InlineKeyboardButton("Compresión extrema", callback_data="compress:extreme"),
        ],
        [InlineKeyboardButton("❌ Cancelar", callback_data="cancel")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Selecciona el nivel de compresión:\n\n"
        "La compresión reduce la diferencia entre sonidos fuertes y débiles.",
        reply_markup=reply_markup
    )
    logger.info(f"[{correlation_id}] Compression ratio selection keyboard sent to user {user_id}")


async def handle_effect_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle effect selection callback from inline keyboard.

    Downloads the audio, applies the selected effect (denoise or compress),
    and sends back the processed audio.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Parse callback data (e.g., "denoise:5" or "compress:medium")
    callback_data = query.data
    if not callback_data or ":" not in callback_data:
        logger.warning(f"Invalid callback data received: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    parts = callback_data.split(":")
    if len(parts) != 2:
        logger.warning(f"Invalid callback data format: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    effect_type = parts[0]
    parameter = parts[1]

    if effect_type not in ("denoise", "compress"):
        logger.warning(f"Invalid effect type: {effect_type}")
        await query.edit_message_text("Error: tipo de efecto inválido.")
        return

    # Validate and convert parameter
    if effect_type == "denoise":
        try:
            strength = int(parameter)
            if strength < 1 or strength > 10:
                logger.warning(f"Invalid denoise strength: {strength}")
                await query.edit_message_text("Error: intensidad debe estar entre 1 y 10.")
                return
        except ValueError:
            logger.warning(f"Invalid denoise strength value: {parameter}")
            await query.edit_message_text("Error: intensidad inválida.")
            return
    else:  # compress
        preset_map = {
            "light": (2.0, "ligera"),
            "medium": (4.0, "media"),
            "heavy": (8.0, "fuerte"),
            "extreme": (12.0, "extrema"),
        }
        if parameter not in preset_map:
            logger.warning(f"Invalid compress preset: {parameter}")
            await query.edit_message_text("Error: nivel de compresión inválido.")
            return
        ratio, preset_name = preset_map[parameter]

    # Retrieve file_id from context
    file_id = context.user_data.get("effect_audio_file_id")
    correlation_id = context.user_data.get("effect_audio_correlation_id", str(uuid.uuid4())[:8])
    stored_effect_type = context.user_data.get("effect_type")

    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for user {user_id}")
        await query.edit_message_text("Error: no se encontró el archivo de audio. Intenta de nuevo.")
        return

    # Verify effect_type matches stored type
    if stored_effect_type and stored_effect_type != effect_type:
        logger.warning(f"[{correlation_id}] Mismatch: stored={stored_effect_type}, callback={effect_type}")

    # Update message to show processing
    if effect_type == "denoise":
        processing_text = f"Aplicando reducción de ruido (intensidad {strength})..."
        effect_name = "reducción de ruido"
        success_text = f"¡Listo! Reducción de ruido aplicada (intensidad {strength}/10)."
    else:
        processing_text = f"Aplicando compresión ({preset_name})..."
        effect_name = "compresión"
        success_text = f"¡Listo! Compresión aplicada (nivel: {preset_name})."

    logger.info(f"[{correlation_id}] {effect_name.capitalize()} selected by user {user_id}")

    try:
        await query.edit_message_text(processing_text)
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not update message: {e}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_{user_id}_{correlation_id}.audio"
            output_filename = f"effect_{user_id}_{correlation_id}.mp3"

            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download audio file
            logger.info(f"[{correlation_id}] Downloading audio from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Audio downloaded to {input_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download audio for user {user_id}: {e}")
                raise DownloadError("No pude descargar el audio") from e

            # Validate audio integrity after download
            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Audio validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space before processing
            audio_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(audio_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"[{correlation_id}] Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Apply effect with timeout
            logger.info(f"[{correlation_id}] Applying {effect_name} for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                effects = AudioEffects(str(input_path), str(output_path))

                if effect_type == "denoise":
                    await asyncio.wait_for(
                        loop.run_in_executor(None, effects.denoise, float(strength)),
                        timeout=config.PROCESSING_TIMEOUT
                    )
                else:  # compress
                    await asyncio.wait_for(
                        loop.run_in_executor(None, effects.compress, ratio, -20.0),
                        timeout=config.PROCESSING_TIMEOUT
                    )

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Audio effect timed out for user {user_id}")
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            # Send processed audio
            logger.info(f"[{correlation_id}] Sending processed audio to user {user_id}")
            try:
                with open(output_path, "rb") as audio_file:
                    await context.bot.send_audio(
                        chat_id=update.effective_chat.id,
                        audio=audio_file,
                        filename=f"{effect_type}_audio.mp3",
                        title=f"Audio con {effect_name.capitalize()}"
                    )
                logger.info(f"[{correlation_id}] Processed audio sent successfully to user {user_id}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to send processed audio to user {user_id}: {e}")
                raise

            # Update message on success
            try:
                await query.edit_message_text(success_text)
            except Exception as e:
                logger.warning(f"[{correlation_id}] Could not update final message: {e}")

            # Clean up user_data
            context.user_data.pop("effect_audio_file_id", None)
            context.user_data.pop("effect_audio_correlation_id", None)
            context.user_data.pop("effect_type", None)

        except (DownloadError, ValidationError, AudioEffectsError, ProcessingTimeoutError) as e:
            # Handle known processing errors
            logger.error(f"[{correlation_id}] Processing error: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text(f"Error: {str(e)}")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"[{correlation_id}] Unexpected error applying effect for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        # TempManager cleanup happens automatically on context exit
        logger.debug(f"[{correlation_id}] Cleanup completed for user {user_id}")


# =============================================================================
# Normalize Handler
# =============================================================================


async def handle_normalize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /normalize command to apply loudness normalization.

    Usage: /normalize (when replying to an audio or with audio attached)
    Shows inline keyboard with normalization preset options for user to select.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Normalize command received from user {user_id}")

    # Get audio from message or reply
    audio, is_reply = await _get_audio_from_message(update)

    if not audio:
        await update.message.reply_text(
            "Envía /normalize respondiendo a un archivo de audio o adjunta el audio al mensaje."
        )
        return

    # Validate file size before downloading
    if audio.file_size:
        is_valid, error_msg = validate_file_size(audio.file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Store file_id in context for later retrieval
    context.user_data["effect_audio_file_id"] = audio.file_id
    context.user_data["effect_audio_correlation_id"] = correlation_id
    context.user_data["effect_type"] = "normalize"

    # Create inline keyboard with normalization presets (1 per row for clarity)
    keyboard = [
        [
            InlineKeyboardButton("Música/General (-14 LUFS)", callback_data="normalize:music"),
        ],
        [
            InlineKeyboardButton("Podcast/Voz (-16 LUFS)", callback_data="normalize:podcast"),
        ],
        [
            InlineKeyboardButton("Streaming/Broadcast (-23 LUFS)", callback_data="normalize:streaming"),
        ],
        [InlineKeyboardButton("❌ Cancelar", callback_data="cancel")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Selecciona el perfil de normalización:\n\n"
        "La normalización ajusta el volumen al estándar EBU R128.",
        reply_markup=reply_markup
    )
    logger.info(f"[{correlation_id}] Normalization preset keyboard sent to user {user_id}")


async def handle_normalize_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle normalization preset selection callback from inline keyboard.

    Downloads the audio, applies loudness normalization with the selected preset,
    and sends back the normalized audio.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Parse callback data (e.g., "normalize:music", "normalize:podcast", "normalize:streaming")
    callback_data = query.data
    if not callback_data or not callback_data.startswith("normalize:"):
        logger.warning(f"Invalid callback data received: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    parts = callback_data.split(":")
    if len(parts) != 2:
        logger.warning(f"Invalid callback data format: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    preset = parts[1]

    # Map preset to target LUFS value
    preset_map = {
        "music": (-14.0, "Música/General", "reproducción general"),
        "podcast": (-16.0, "Podcast/Voz", "contenido de voz"),
        "streaming": (-23.0, "Streaming/Broadcast", "plataformas de streaming"),
    }

    if preset not in preset_map:
        logger.warning(f"Invalid normalization preset: {preset}")
        await query.edit_message_text("Error: perfil de normalización inválido.")
        return

    target_lufs, preset_name, use_case = preset_map[preset]

    # Retrieve file_id from context
    file_id = context.user_data.get("effect_audio_file_id")
    correlation_id = context.user_data.get("effect_audio_correlation_id", str(uuid.uuid4())[:8])
    stored_effect_type = context.user_data.get("effect_type")

    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for user {user_id}")
        await query.edit_message_text("Error: no se encontró el archivo de audio. Intenta de nuevo.")
        return

    # Verify effect_type matches stored type
    if stored_effect_type and stored_effect_type != "normalize":
        logger.warning(f"[{correlation_id}] Mismatch: stored={stored_effect_type}, callback=normalize")

    logger.info(f"[{correlation_id}] Normalization preset '{preset}' ({target_lufs} LUFS) selected by user {user_id}")

    # Update message to show processing
    try:
        await query.edit_message_text(f"Normalizando audio a {preset_name} ({target_lufs} LUFS)...")
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not update message: {e}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_{user_id}_{correlation_id}.audio"
            output_filename = f"normalized_{user_id}_{correlation_id}.mp3"

            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download audio file
            logger.info(f"[{correlation_id}] Downloading audio from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Audio downloaded to {input_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download audio for user {user_id}: {e}")
                raise DownloadError("No pude descargar el audio") from e

            # Validate audio integrity after download
            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Audio validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space before processing
            audio_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(audio_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"[{correlation_id}] Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Apply normalization with timeout
            logger.info(f"[{correlation_id}] Applying normalization ({target_lufs} LUFS) for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                effects = AudioEffects(str(input_path), str(output_path))

                success = await asyncio.wait_for(
                    loop.run_in_executor(None, effects.normalize, target_lufs),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not success:
                    logger.error(f"[{correlation_id}] Normalization failed for user {user_id}")
                    raise AudioEffectsError("No pude normalizar el audio")

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Normalization timed out for user {user_id}")
                raise ProcessingTimeoutError("La normalización tardó demasiado") from e

            # Send normalized audio
            logger.info(f"[{correlation_id}] Sending normalized audio to user {user_id}")
            try:
                with open(output_path, "rb") as audio_file:
                    await context.bot.send_audio(
                        chat_id=update.effective_chat.id,
                        audio=audio_file,
                        filename=f"normalized.mp3",
                        title=f"Audio normalizado ({preset_name})"
                    )
                logger.info(f"[{correlation_id}] Normalized audio sent successfully to user {user_id}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to send normalized audio to user {user_id}: {e}")
                raise

            # Update message on success
            try:
                await query.edit_message_text(
                    f"¡Listo! Audio normalizado a {preset_name} ({target_lufs} LUFS).\n\n"
                    f"El volumen ahora está optimizado para {use_case}."
                )
            except Exception as e:
                logger.warning(f"[{correlation_id}] Could not update final message: {e}")

            # Clean up user_data
            context.user_data.pop("effect_audio_file_id", None)
            context.user_data.pop("effect_audio_correlation_id", None)
            context.user_data.pop("effect_type", None)

        except (DownloadError, ValidationError, AudioEffectsError, ProcessingTimeoutError) as e:
            # Handle known processing errors
            logger.error(f"[{correlation_id}] Processing error: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text(f"Error: {str(e)}")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"[{correlation_id}] Unexpected error normalizing audio for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        # TempManager cleanup happens automatically on context exit
        logger.debug(f"[{correlation_id}] Cleanup completed for user {user_id}")


async def handle_audio_3d_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle stereo 3D intensity selection callback from inline keyboard."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    if not callback_data or not callback_data.startswith("audio_3d:"):
        logger.warning(f"Invalid callback data received: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    parts = callback_data.split(":")
    if len(parts) != 2:
        await query.edit_message_text("Error: selección inválida.")
        return

    intensity = parts[1]
    intensity_labels = {
        "suave": "Suave",
        "medio": "Medio",
        "intenso": "Intenso",
    }

    if intensity not in intensity_labels:
        await query.edit_message_text("Error: intensidad inválida.")
        return

    file_id = context.user_data.get("effect_audio_file_id")
    correlation_id = context.user_data.get("effect_audio_correlation_id", str(uuid.uuid4())[:8])
    stored_effect_type = context.user_data.get("effect_type")

    if not file_id:
        await query.edit_message_text("Error: no se encontró el archivo de audio. Intenta de nuevo.")
        return

    if stored_effect_type and stored_effect_type != "stereo_3d":
        logger.warning(
            f"[{correlation_id}] Mismatch: stored={stored_effect_type}, callback=stereo_3d"
        )

    intensity_label = intensity_labels[intensity]
    logger.info(
        f"[{correlation_id}] Stereo 3D intensity '{intensity}' selected by user {user_id}"
    )

    try:
        await query.edit_message_text(f"Aplicando efecto 3D ({intensity_label})...")
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not update message: {e}")

    with TempManager() as temp_mgr:
        try:
            input_filename = f"input_{user_id}_{correlation_id}.audio"
            output_filename = f"stereo3d_{user_id}_{correlation_id}.mp3"
            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Downloading audio from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download audio: {e}")
                raise DownloadError("No pude descargar el audio") from e

            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                raise ValidationError(error_msg)

            audio_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(audio_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                raise ValidationError(space_error)

            logger.info(f"[{correlation_id}] Applying stereo 3D ({intensity}) for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                effects = AudioEffects(str(input_path), str(output_path))
                await asyncio.wait_for(
                    loop.run_in_executor(None, effects.stereo_3d, intensity),
                    timeout=config.PROCESSING_TIMEOUT,
                )
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El efecto 3D tardó demasiado") from e

            doc_filename = f"stereo_3d_{intensity}_{correlation_id}.mp3"
            document_sent = False
            with open(output_path, "rb") as audio_file:
                await context.bot.send_audio(
                    chat_id=update.effective_chat.id,
                    audio=audio_file,
                    filename="stereo_3d.mp3",
                    title=f"Audio con efecto 3D ({intensity_label})",
                )
                try:
                    audio_file.seek(0)
                    await context.bot.send_document(
                        chat_id=update.effective_chat.id,
                        document=audio_file,
                        filename=doc_filename,
                        caption=(
                            f"Archivo MP3 con efecto 3D ({intensity_label}) "
                            "para editores de video"
                        ),
                    )
                    document_sent = True
                except Exception as doc_error:
                    logger.warning(
                        f"[{correlation_id}] Audio sent but document delivery failed: {doc_error}"
                    )

            success_msg = f"¡Listo! Efecto 3D aplicado con intensidad {intensity_label}."
            if not document_sent:
                success_msg += (
                    "\n\n(No pude enviar el archivo MP3 como documento; "
                    "usa el audio de arriba.)"
                )
            await query.edit_message_text(success_msg)

            context.user_data.pop("effect_audio_file_id", None)
            context.user_data.pop("effect_audio_correlation_id", None)
            context.user_data.pop("effect_type", None)

        except (DownloadError, ValidationError, AudioEffectsError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Stereo 3D processing error: {e}")
            try:
                await query.edit_message_text(f"Error: {get_user_error_message(e)}")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error applying stereo 3D: {e}")
            try:
                await query.edit_message_text(DEFAULT_ERROR_MESSAGE)
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")


# =============================================================================
# Audio Inline Menu Handler
# =============================================================================


async def handle_audio_pitch_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle pitch shift intensity selection callback from inline keyboard."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    if not callback_data or not callback_data.startswith("audio_pitch:"):
        logger.warning(f"Invalid callback data received: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    parts = callback_data.split(":")
    if len(parts) != 2:
        await query.edit_message_text("Error: selección inválida.")
        return

    intensity = parts[1]
    intensity_labels = {
        "grave": "Grave",
        "agudo": "Agudo",
        "muy_agudo": "Muy agudo",
    }

    if intensity not in intensity_labels:
        await query.edit_message_text("Error: intensidad inválida.")
        return

    file_id = context.user_data.get("effect_audio_file_id")
    correlation_id = context.user_data.get("effect_audio_correlation_id", str(uuid.uuid4())[:8])
    stored_effect_type = context.user_data.get("effect_type")

    if not file_id:
        await query.edit_message_text("Error: no se encontró el archivo de audio. Intenta de nuevo.")
        return

    if stored_effect_type and stored_effect_type != "pitch_shift":
        logger.warning(
            f"[{correlation_id}] Mismatch: stored={stored_effect_type}, callback=pitch_shift"
        )

    intensity_label = intensity_labels[intensity]
    logger.info(
        f"[{correlation_id}] Pitch shift intensity '{intensity}' selected by user {user_id}"
    )

    try:
        await query.edit_message_text(f"Aplicando cambio de tono ({intensity_label})...")
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not update message: {e}")

    with TempManager() as temp_mgr:
        try:
            input_filename = f"input_{user_id}_{correlation_id}.audio"
            output_filename = f"pitch_{user_id}_{correlation_id}.mp3"
            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Downloading audio from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download audio: {e}")
                raise DownloadError("No pude descargar el audio") from e

            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                raise ValidationError(error_msg)

            audio_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(audio_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                raise ValidationError(space_error)

            logger.info(f"[{correlation_id}] Applying pitch shift ({intensity}) for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                effects = AudioEffects(str(input_path), str(output_path))
                await asyncio.wait_for(
                    loop.run_in_executor(None, effects.pitch_shift, intensity),
                    timeout=config.PROCESSING_TIMEOUT,
                )
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El cambio de tono tardó demasiado") from e

            doc_filename = f"pitch_shift_{intensity}_{correlation_id}.mp3"
            document_sent = False
            with open(output_path, "rb") as audio_file:
                await context.bot.send_audio(
                    chat_id=update.effective_chat.id,
                    audio=audio_file,
                    filename="pitch_shift.mp3",
                    title=f"Audio con cambio de tono ({intensity_label})",
                )
                try:
                    audio_file.seek(0)
                    await context.bot.send_document(
                        chat_id=update.effective_chat.id,
                        document=audio_file,
                        filename=doc_filename,
                        caption=(
                            f"Archivo MP3 con cambio de tono ({intensity_label}) "
                            "para editores de video"
                        ),
                    )
                    document_sent = True
                except Exception as doc_error:
                    logger.warning(
                        f"[{correlation_id}] Audio sent but document delivery failed: {doc_error}"
                    )

            success_msg = f"¡Listo! Cambio de tono aplicado con intensidad {intensity_label}."
            if not document_sent:
                success_msg += (
                    "\n\n(No pude enviar el archivo MP3 como documento; "
                    "usa el audio de arriba.)"
                )
            await query.edit_message_text(success_msg)

            context.user_data.pop("effect_audio_file_id", None)
            context.user_data.pop("effect_audio_correlation_id", None)
            context.user_data.pop("effect_type", None)

        except (DownloadError, ValidationError, AudioEffectsError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Pitch shift processing error: {e}")
            try:
                await query.edit_message_text(f"Error: {get_user_error_message(e)}")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error applying pitch shift: {e}")
            try:
                await query.edit_message_text(DEFAULT_ERROR_MESSAGE)
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")


def _get_audio_menu_keyboard() -> InlineKeyboardMarkup:
    """Generate inline keyboard for audio menu options.

    Returns:
        InlineKeyboardMarkup with audio action buttons
    """
    keyboard = [
        [
            InlineKeyboardButton("Nota de Voz", callback_data="audio_action:voicenote"),
            InlineKeyboardButton("Convertir Formato", callback_data="audio_action:convert"),
        ],
        [
            InlineKeyboardButton("Bass Boost", callback_data="audio_action:bass_boost"),
            InlineKeyboardButton("Treble Boost", callback_data="audio_action:treble_boost"),
            InlineKeyboardButton("Ecualizar", callback_data="audio_action:equalize"),
        ],
        [
            InlineKeyboardButton("Reducir Ruido", callback_data="audio_action:denoise"),
            InlineKeyboardButton("Comprimir", callback_data="audio_action:compress"),
            InlineKeyboardButton("Normalizar", callback_data="audio_action:normalize"),
        ],
        [
            InlineKeyboardButton("Efecto 3D", callback_data="audio_action:stereo_3d"),
            InlineKeyboardButton("Cambiar Pitch", callback_data="audio_action:pitch_shift"),
        ],
        [
            InlineKeyboardButton("Dividir Audio", callback_data="audio_action:split"),
            InlineKeyboardButton("Unir Audios", callback_data="audio_action:join"),
        ],
        [
            InlineKeyboardButton("Pipeline de Efectos", callback_data="audio_action:effects"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


async def handle_audio_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio menu selection callbacks from inline keyboard.

    Routes to appropriate action based on user selection:
    - voicenote: Convert to voice note
    - convert: Show format selection
    - bass_boost/treble_boost/equalize: Show enhancement options
    - denoise/compress/normalize: Show effect options
    - effects: Show pipeline builder

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Parse callback data (format: "audio_action:<action>")
    callback_data = query.data
    if not callback_data or not callback_data.startswith("audio_action:"):
        logger.warning(f"Invalid callback data received: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    action = callback_data.split(":")[1]

    # Retrieve file_id from context
    file_id = context.user_data.get("audio_menu_file_id")
    correlation_id = context.user_data.get("audio_menu_correlation_id", str(uuid.uuid4())[:8])

    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for user {user_id}")
        await query.edit_message_text("Error: no se encontró el archivo de audio. Intenta de nuevo.")
        return

    logger.info(f"[{correlation_id}] Audio menu action '{action}' selected by user {user_id}")

    # Route to appropriate action
    if action == "voicenote":
        await _handle_audio_menu_voicenote(update, context, file_id, correlation_id)

    elif action == "convert":
        # Store action and show format selection
        context.user_data["audio_menu_action"] = "convert"
        keyboard = [
            [
                InlineKeyboardButton("MP3", callback_data="audio_menu_format:mp3"),
                InlineKeyboardButton("WAV", callback_data="audio_menu_format:wav"),
                InlineKeyboardButton("OGG", callback_data="audio_menu_format:ogg"),
            ],
            [
                InlineKeyboardButton("AAC", callback_data="audio_menu_format:aac"),
                InlineKeyboardButton("FLAC", callback_data="audio_menu_format:flac"),
            ],
            [
                InlineKeyboardButton("← Volver", callback_data="back:audio"),
                InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona el formato de conversión:",
            reply_markup=reply_markup
        )

    elif action == "bass_boost":
        # Store file info for enhancement handler
        context.user_data["enhance_audio_file_id"] = file_id
        context.user_data["enhance_audio_correlation_id"] = correlation_id
        context.user_data["enhance_type"] = "bass"
        # Show intensity selection keyboard
        keyboard = [
            [
                InlineKeyboardButton("1", callback_data="bass:1"),
                InlineKeyboardButton("2", callback_data="bass:2"),
                InlineKeyboardButton("3", callback_data="bass:3"),
                InlineKeyboardButton("4", callback_data="bass:4"),
                InlineKeyboardButton("5", callback_data="bass:5"),
            ],
            [
                InlineKeyboardButton("6", callback_data="bass:6"),
                InlineKeyboardButton("7", callback_data="bass:7"),
                InlineKeyboardButton("8", callback_data="bass:8"),
                InlineKeyboardButton("9", callback_data="bass:9"),
                InlineKeyboardButton("10", callback_data="bass:10"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona la intensidad del Bass Boost (1-10):",
            reply_markup=reply_markup
        )

    elif action == "treble_boost":
        # Store file info for enhancement handler
        context.user_data["enhance_audio_file_id"] = file_id
        context.user_data["enhance_audio_correlation_id"] = correlation_id
        context.user_data["enhance_type"] = "treble"
        # Show intensity selection keyboard
        keyboard = [
            [
                InlineKeyboardButton("1", callback_data="treble:1"),
                InlineKeyboardButton("2", callback_data="treble:2"),
                InlineKeyboardButton("3", callback_data="treble:3"),
                InlineKeyboardButton("4", callback_data="treble:4"),
                InlineKeyboardButton("5", callback_data="treble:5"),
            ],
            [
                InlineKeyboardButton("6", callback_data="treble:6"),
                InlineKeyboardButton("7", callback_data="treble:7"),
                InlineKeyboardButton("8", callback_data="treble:8"),
                InlineKeyboardButton("9", callback_data="treble:9"),
                InlineKeyboardButton("10", callback_data="treble:10"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona la intensidad del Treble Boost (1-10):",
            reply_markup=reply_markup
        )

    elif action == "equalize":
        # Store file info for equalizer
        context.user_data["eq_file_id"] = file_id
        context.user_data["eq_correlation_id"] = correlation_id
        context.user_data["eq_bass"] = 0
        context.user_data["eq_mid"] = 0
        context.user_data["eq_treble"] = 0
        # Show equalizer keyboard
        reply_markup = _get_equalizer_keyboard(0, 0, 0)
        await query.edit_message_text(
            "Ecualizador de 3 bandas:\n"
            "🎵 Bass: 0\n"
            "🎵 Mid: 0\n"
            "🎵 Treble: 0\n\n"
            "Ajusta cada banda y presiona Aplicar.",
            reply_markup=reply_markup
        )

    elif action == "denoise":
        # Store file info for effect handler
        context.user_data["effect_audio_file_id"] = file_id
        context.user_data["effect_audio_correlation_id"] = correlation_id
        context.user_data["effect_type"] = "denoise"
        # Show strength selection keyboard
        keyboard = [
            [
                InlineKeyboardButton("1", callback_data="denoise:1"),
                InlineKeyboardButton("2", callback_data="denoise:2"),
                InlineKeyboardButton("3", callback_data="denoise:3"),
                InlineKeyboardButton("4", callback_data="denoise:4"),
                InlineKeyboardButton("5", callback_data="denoise:5"),
            ],
            [
                InlineKeyboardButton("6", callback_data="denoise:6"),
                InlineKeyboardButton("7", callback_data="denoise:7"),
                InlineKeyboardButton("8", callback_data="denoise:8"),
                InlineKeyboardButton("9", callback_data="denoise:9"),
                InlineKeyboardButton("10", callback_data="denoise:10"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona la intensidad de reducción de ruido (1-10):\n\n"
            "1 = Reducción ligera\n"
            "10 = Reducción máxima",
            reply_markup=reply_markup
        )

    elif action == "compress":
        # Store file info for effect handler
        context.user_data["effect_audio_file_id"] = file_id
        context.user_data["effect_audio_correlation_id"] = correlation_id
        context.user_data["effect_type"] = "compress"
        # Show compression preset keyboard
        keyboard = [
            [
                InlineKeyboardButton("Ligera", callback_data="compress:light"),
                InlineKeyboardButton("Media", callback_data="compress:medium"),
            ],
            [
                InlineKeyboardButton("Fuerte", callback_data="compress:heavy"),
                InlineKeyboardButton("Extrema", callback_data="compress:extreme"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona el nivel de compresión:",
            reply_markup=reply_markup
        )

    elif action == "normalize":
        # Store file info for effect handler
        context.user_data["effect_audio_file_id"] = file_id
        context.user_data["effect_audio_correlation_id"] = correlation_id
        context.user_data["effect_type"] = "normalize"
        # Show normalization preset keyboard
        keyboard = [
            [
                InlineKeyboardButton("Música", callback_data="normalize:music"),
            ],
            [
                InlineKeyboardButton("Podcast", callback_data="normalize:podcast"),
            ],
            [
                InlineKeyboardButton("Streaming", callback_data="normalize:streaming"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona el perfil de normalización:",
            reply_markup=reply_markup
        )

    elif action == "stereo_3d":
        context.user_data["effect_audio_file_id"] = file_id
        context.user_data["effect_audio_correlation_id"] = correlation_id
        context.user_data["effect_type"] = "stereo_3d"
        keyboard = [
            [
                InlineKeyboardButton("Suave", callback_data="audio_3d:suave"),
                InlineKeyboardButton("Medio", callback_data="audio_3d:medio"),
                InlineKeyboardButton("Intenso", callback_data="audio_3d:intenso"),
            ],
            [
                InlineKeyboardButton("← Volver", callback_data="back:audio"),
                InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona la intensidad del efecto 3D:\n\n"
            "• Suave - ampliación estéreo ligera\n"
            "• Medio - efecto equilibrado\n"
            "• Intenso - ampliación estéreo marcada",
            reply_markup=reply_markup
        )

    elif action == "pitch_shift":
        context.user_data["effect_audio_file_id"] = file_id
        context.user_data["effect_audio_correlation_id"] = correlation_id
        context.user_data["effect_type"] = "pitch_shift"
        keyboard = [
            [
                InlineKeyboardButton("Grave", callback_data="audio_pitch:grave"),
                InlineKeyboardButton("Agudo", callback_data="audio_pitch:agudo"),
                InlineKeyboardButton("Muy Agudo", callback_data="audio_pitch:muy_agudo"),
            ],
            [
                InlineKeyboardButton("<- Volver", callback_data="back:audio"),
                InlineKeyboardButton("Cancelar", callback_data="cancel"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona el cambio de tono:\n\n"
            "• Grave - tono más grave (-3.5 semitonos)\n"
            "• Agudo - tono más agudo (+3.5 semitonos)\n"
            "• Muy Agudo - tono muy agudo (+6.5 semitonos)",
            reply_markup=reply_markup
        )

    elif action == "effects":
        # Store file info for pipeline builder
        context.user_data["pipeline_file_id"] = file_id
        context.user_data["pipeline_correlation_id"] = correlation_id
        context.user_data["pipeline_effects"] = []
        # Show pipeline builder keyboard
        reply_markup = _get_pipeline_keyboard([])
        await query.edit_message_text(
            _format_pipeline_message([]),
            reply_markup=reply_markup
        )

    elif action == "split":
        # Start interactive audio split process
        await handle_audio_split_start(update, context)

    elif action == "join":
        # Start audio join session
        temp_mgr = TempManager()
        context.user_data["join_audio_session"] = {
            "audios": [],
            "correlation_id": correlation_id,
            "last_activity": time.time(),
            "temp_mgr": temp_mgr,
        }
        await query.edit_message_text(
            "¡Perfecto! Ahora envíame los archivos de audio que quieres unir (uno por uno).\n\n"
            "Cuando termines, envía /done para procesar.\n"
            "Envía /cancel para cancelar."
        )
        logger.info(f"[{correlation_id}] Started audio join session for user {user_id}")

    else:
        logger.warning(f"[{correlation_id}] Unknown audio action: {action}")
        await query.edit_message_text("Error: acción no reconocida.")


async def _handle_audio_menu_voicenote(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
    correlation_id: str
) -> None:
    """Handle voice note conversion from audio menu.

    Args:
        update: Telegram update object
        context: Telegram context object
        file_id: Telegram file ID of the audio
        correlation_id: Correlation ID for tracing
    """
    query = update.callback_query
    user_id = update.effective_user.id

    # Update message to show processing
    try:
        await query.edit_message_text("Convirtiendo a nota de voz...")
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not update message: {e}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_{user_id}_{correlation_id}.audio"
            output_filename = f"voice_{user_id}_{correlation_id}.ogg"

            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download audio file
            logger.info(f"[{correlation_id}] Downloading audio from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Audio downloaded to {input_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download audio for user {user_id}: {e}")
                raise DownloadError("No pude descargar el audio") from e

            # Validate audio integrity after download
            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Audio validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space before processing
            audio_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(audio_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"[{correlation_id}] Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Convert to voice note with timeout
            logger.info(f"[{correlation_id}] Converting audio to voice note for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                converter = VoiceNoteConverter(str(input_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, converter.process),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not success:
                    logger.error(f"[{correlation_id}] Voice note conversion failed for user {user_id}")
                    raise VoiceConversionError("No pude convertir el audio a nota de voz")

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Voice note conversion timed out for user {user_id}")
                raise ProcessingTimeoutError("El audio tardó demasiado en procesarse") from e

            # Send as voice note
            logger.info(f"[{correlation_id}] Sending voice note to user {user_id}")
            try:
                with open(output_path, "rb") as voice_file:
                    await context.bot.send_voice(
                        chat_id=update.effective_chat.id,
                        voice=voice_file
                    )
                logger.info(f"[{correlation_id}] Voice note sent successfully to user {user_id}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to send voice note to user {user_id}: {e}")
                raise

            # Update message on success
            try:
                await query.edit_message_text("¡Listo! Audio convertido a nota de voz.")
            except Exception as e:
                logger.warning(f"[{correlation_id}] Could not update final message: {e}")

            # Clean up user_data
            context.user_data.pop("audio_menu_file_id", None)
            context.user_data.pop("audio_menu_correlation_id", None)

        except (DownloadError, ValidationError, VoiceConversionError, ProcessingTimeoutError) as e:
            # Handle known processing errors
            logger.error(f"[{correlation_id}] Processing error: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text(f"Error: {str(e)}")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"[{correlation_id}] Unexpected error converting audio for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")


async def handle_audio_menu_format_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio format selection callback from inline menu.

    Downloads the audio, converts it to selected format, and sends back.
    This is specifically for the menu flow (audio_menu_format:* pattern).

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Extract format from callback data (e.g., "audio_menu_format:mp3" -> "mp3")
    callback_data = query.data
    if not callback_data.startswith("audio_menu_format:"):
        logger.warning(f"Invalid callback data received: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    output_format = callback_data.split(":")[1]

    # Retrieve file_id from context
    file_id = context.user_data.get("audio_menu_file_id")
    correlation_id = context.user_data.get("audio_menu_correlation_id", str(uuid.uuid4())[:8])

    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for user {user_id}")
        await query.edit_message_text("Error: no se encontró el archivo de audio. Intenta de nuevo.")
        return

    logger.info(f"[{correlation_id}] Format {output_format} selected by user {user_id} (from menu)")

    # Update message to show processing
    try:
        await query.edit_message_text(f"Convirtiendo audio a {output_format.upper()}...")
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not update message: {e}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_{user_id}_{correlation_id}.audio"
            output_filename = f"converted_{user_id}_{correlation_id}.{output_format}"

            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download audio file
            logger.info(f"[{correlation_id}] Downloading audio from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Audio downloaded to {input_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download audio for user {user_id}: {e}")
                raise DownloadError("No pude descargar el audio") from e

            # Validate audio integrity after download
            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Audio validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Detect input format
            input_format = detect_audio_format(str(input_path))
            if input_format:
                logger.info(f"[{correlation_id}] Detected input format: {input_format}")
                # Check if input format equals output format
                if input_format == output_format:
                    await query.edit_message_text(
                        f"El archivo ya está en formato {output_format.upper()}. No es necesario convertir."
                    )
                    return
            else:
                logger.warning(f"[{correlation_id}] Could not detect input format for user {user_id}")

            # Check disk space before processing
            audio_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(audio_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"[{correlation_id}] Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Convert audio with timeout
            logger.info(f"[{correlation_id}] Converting audio to {output_format} for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                converter = AudioFormatConverter(str(input_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, converter.convert, output_format),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not success:
                    logger.error(f"[{correlation_id}] Audio format conversion failed for user {user_id}")
                    raise AudioFormatConversionError(f"No pude convertir el audio a {output_format.upper()}")

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Audio conversion timed out for user {user_id}")
                raise ProcessingTimeoutError("La conversión tardó demasiado") from e

            # Send converted audio
            logger.info(f"[{correlation_id}] Sending converted audio to user {user_id}")
            try:
                with open(output_path, "rb") as audio_file:
                    await context.bot.send_audio(
                        chat_id=update.effective_chat.id,
                        audio=audio_file,
                        filename=f"converted.{output_format}",
                        title=f"Audio convertido a {output_format.upper()}"
                    )
                logger.info(f"[{correlation_id}] Converted audio sent successfully to user {user_id}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to send converted audio to user {user_id}: {e}")
                raise

            # Update message on success
            try:
                await query.edit_message_text(f"Audio convertido a {output_format.upper()} exitosamente.")
            except Exception as e:
                logger.warning(f"[{correlation_id}] Could not update final message: {e}")

            # Clean up user_data
            context.user_data.pop("audio_menu_file_id", None)
            context.user_data.pop("audio_menu_correlation_id", None)
            context.user_data.pop("audio_menu_action", None)

        except (DownloadError, ValidationError, AudioFormatConversionError, ProcessingTimeoutError) as e:
            # Handle known processing errors
            logger.error(f"[{correlation_id}] Processing error: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text(f"Error: {str(e)}")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"[{correlation_id}] Unexpected error converting audio for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")


# Effects Pipeline Handler
# =============================================================================


def _get_pipeline_keyboard(pipeline_effects: list) -> InlineKeyboardMarkup:
    """Generate inline keyboard for pipeline builder.

    Args:
        pipeline_effects: List of effect configs in the pipeline

    Returns:
        InlineKeyboardMarkup with add/preview/apply/cancel buttons
    """
    # Add effect buttons row
    add_buttons = [
        InlineKeyboardButton("+ Denoise", callback_data="pipeline_add:denoise"),
        InlineKeyboardButton("+ Compress", callback_data="pipeline_add:compress"),
        InlineKeyboardButton("+ Normalize", callback_data="pipeline_add:normalize"),
    ]

    # Preview button row
    preview_button = [InlineKeyboardButton("Ver Pipeline", callback_data="pipeline_preview")]

    # Action buttons row
    action_buttons = [
        InlineKeyboardButton("Aplicar", callback_data="pipeline_apply"),
        InlineKeyboardButton("Cancelar", callback_data="pipeline_cancel"),
    ]

    keyboard = [add_buttons, preview_button, action_buttons]
    return InlineKeyboardMarkup(keyboard)


def _format_pipeline_message(pipeline_effects: list) -> str:
    """Format pipeline display message.

    Args:
        pipeline_effects: List of effect configs in the pipeline

    Returns:
        Formatted message string showing current pipeline
    """
    if not pipeline_effects:
        return (
            "Constructor de efectos de audio:\n\n"
            "Efectos en pipeline: (ninguno)\n\n"
            "Agrega efectos en el orden que deseas aplicarlos.\n"
            "Orden recomendado: Denoise → Compress → Normalize"
        )

    effect_lines = []
    for i, effect in enumerate(pipeline_effects, 1):
        effect_type = effect.get("type", "unknown")
        params = effect.get("params", {})

        if effect_type == "denoise":
            strength = params.get("strength", 5)
            effect_lines.append(f"{i}. Denoise (intensidad: {strength})")
        elif effect_type == "compress":
            ratio = params.get("ratio", 4.0)
            preset_name = params.get("preset_name", "media")
            effect_lines.append(f"{i}. Compress (ratio: {preset_name})")
        elif effect_type == "normalize":
            target_lufs = params.get("target_lufs", -14.0)
            preset_name = params.get("preset_name", "música")
            effect_lines.append(f"{i}. Normalize (perfil: {preset_name})")
        else:
            effect_lines.append(f"{i}. {effect_type}")

    pipeline_text = "\n".join(effect_lines)
    return (
        f"Constructor de efectos de audio:\n\n"
        f"Pipeline ({len(pipeline_effects)} efectos):\n"
        f"{pipeline_text}\n\n"
        f"Agrega más efectos o aplica el pipeline."
    )


async def handle_effects_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /effects command to show pipeline builder interface.

    Usage: /effects (when replying to an audio or with audio attached)

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Effects command received from user {user_id}")

    # Get audio from message or reply
    audio, is_reply = await _get_audio_from_message(update)

    if not audio:
        await update.message.reply_text(
            "Envía /effects respondiendo a un archivo de audio o adjunta el audio al mensaje."
        )
        return

    # Validate file size before downloading
    if audio.file_size:
        is_valid, error_msg = validate_file_size(audio.file_size, config.max_incoming_audio_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed for user {user_id}: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    # Initialize pipeline state in context.user_data
    context.user_data["pipeline_file_id"] = audio.file_id
    context.user_data["pipeline_correlation_id"] = correlation_id
    context.user_data["pipeline_effects"] = []

    # Create inline keyboard
    reply_markup = _get_pipeline_keyboard([])

    await update.message.reply_text(
        _format_pipeline_message([]),
        reply_markup=reply_markup
    )
    logger.info(f"[{correlation_id}] Pipeline builder interface sent to user {user_id}")


async def handle_pipeline_builder(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle pipeline builder callbacks from inline keyboard.

    Handles add effect, preview, apply, and cancel actions.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data
    correlation_id = context.user_data.get("pipeline_correlation_id", str(uuid.uuid4())[:8])

    # Get current pipeline state
    pipeline_effects = context.user_data.get("pipeline_effects", [])

    # Handle cancel
    if callback_data == "pipeline_cancel":
        # Clear all pipeline state
        context.user_data.pop("pipeline_file_id", None)
        context.user_data.pop("pipeline_correlation_id", None)
        context.user_data.pop("pipeline_effects", None)
        context.user_data.pop("pipeline_selecting_effect", None)

        await query.edit_message_text("Pipeline cancelado.")
        logger.info(f"[{correlation_id}] Pipeline cancelled by user {user_id}")
        return

    # Handle preview
    if callback_data == "pipeline_preview":
        if not pipeline_effects:
            await query.answer("No hay efectos en el pipeline", show_alert=True)
        else:
            preview_text = "Pipeline actual:\n\n"
            for i, effect in enumerate(pipeline_effects, 1):
                effect_type = effect.get("type", "unknown")
                params = effect.get("params", {})

                if effect_type == "denoise":
                    strength = params.get("strength", 5)
                    preview_text += f"{i}. Denoise (intensidad: {strength})\n"
                elif effect_type == "compress":
                    preset_name = params.get("preset_name", "media")
                    preview_text += f"{i}. Compress (ratio: {preset_name})\n"
                elif effect_type == "normalize":
                    preset_name = params.get("preset_name", "música")
                    preview_text += f"{i}. Normalize (perfil: {preset_name})\n"

            await query.answer(preview_text, show_alert=True)
        return

    # Handle add effect selection
    if callback_data.startswith("pipeline_add:"):
        effect_type = callback_data.split(":")[1]

        if effect_type == "denoise":
            # Show denoise strength selection keyboard
            keyboard = [
                [
                    InlineKeyboardButton("1", callback_data="pipeline_denoise:1"),
                    InlineKeyboardButton("2", callback_data="pipeline_denoise:2"),
                    InlineKeyboardButton("3", callback_data="pipeline_denoise:3"),
                    InlineKeyboardButton("4", callback_data="pipeline_denoise:4"),
                    InlineKeyboardButton("5", callback_data="pipeline_denoise:5"),
                ],
                [
                    InlineKeyboardButton("6", callback_data="pipeline_denoise:6"),
                    InlineKeyboardButton("7", callback_data="pipeline_denoise:7"),
                    InlineKeyboardButton("8", callback_data="pipeline_denoise:8"),
                    InlineKeyboardButton("9", callback_data="pipeline_denoise:9"),
                    InlineKeyboardButton("10", callback_data="pipeline_denoise:10"),
                ],
                [InlineKeyboardButton("Volver", callback_data="pipeline_back")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "Selecciona la intensidad de reducción de ruido (1-10):\n\n"
                "1 = Reducción ligera\n"
                "10 = Reducción máxima",
                reply_markup=reply_markup
            )

        elif effect_type == "compress":
            # Show compress ratio selection keyboard
            keyboard = [
                [
                    InlineKeyboardButton("Ligera", callback_data="pipeline_compress:light"),
                    InlineKeyboardButton("Media", callback_data="pipeline_compress:medium"),
                ],
                [
                    InlineKeyboardButton("Fuerte", callback_data="pipeline_compress:heavy"),
                    InlineKeyboardButton("Extrema", callback_data="pipeline_compress:extreme"),
                ],
                [InlineKeyboardButton("Volver", callback_data="pipeline_back")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "Selecciona el nivel de compresión:",
                reply_markup=reply_markup
            )

        elif effect_type == "normalize":
            # Show normalize preset selection keyboard
            keyboard = [
                [
                    InlineKeyboardButton("Música (-14 LUFS)", callback_data="pipeline_normalize:music"),
                ],
                [
                    InlineKeyboardButton("Podcast (-16 LUFS)", callback_data="pipeline_normalize:podcast"),
                ],
                [
                    InlineKeyboardButton("Streaming (-23 LUFS)", callback_data="pipeline_normalize:streaming"),
                ],
                [InlineKeyboardButton("Volver", callback_data="pipeline_back")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "Selecciona el perfil de normalización:",
                reply_markup=reply_markup
            )

        return

    # Handle back button
    if callback_data == "pipeline_back":
        reply_markup = _get_pipeline_keyboard(pipeline_effects)
        await query.edit_message_text(
            _format_pipeline_message(pipeline_effects),
            reply_markup=reply_markup
        )
        return

    # Handle denoise parameter selection
    if callback_data.startswith("pipeline_denoise:"):
        strength = int(callback_data.split(":")[1])
        effect_config = {
            "type": "denoise",
            "params": {"strength": strength}
        }
        pipeline_effects.append(effect_config)
        context.user_data["pipeline_effects"] = pipeline_effects

        reply_markup = _get_pipeline_keyboard(pipeline_effects)
        await query.edit_message_text(
            _format_pipeline_message(pipeline_effects),
            reply_markup=reply_markup
        )
        logger.info(f"[{correlation_id}] Denoise (strength={strength}) added to pipeline by user {user_id}")
        return

    # Handle compress parameter selection
    if callback_data.startswith("pipeline_compress:"):
        preset = callback_data.split(":")[1]
        preset_map = {
            "light": (2.0, "ligera"),
            "medium": (4.0, "media"),
            "heavy": (8.0, "fuerte"),
            "extreme": (12.0, "extrema"),
        }
        ratio, preset_name = preset_map.get(preset, (4.0, "media"))
        effect_config = {
            "type": "compress",
            "params": {"ratio": ratio, "preset_name": preset_name}
        }
        pipeline_effects.append(effect_config)
        context.user_data["pipeline_effects"] = pipeline_effects

        reply_markup = _get_pipeline_keyboard(pipeline_effects)
        await query.edit_message_text(
            _format_pipeline_message(pipeline_effects),
            reply_markup=reply_markup
        )
        logger.info(f"[{correlation_id}] Compress (ratio={preset_name}) added to pipeline by user {user_id}")
        return

    # Handle normalize parameter selection
    if callback_data.startswith("pipeline_normalize:"):
        preset = callback_data.split(":")[1]
        preset_map = {
            "music": (-14.0, "música", "streaming y música"),
            "podcast": (-16.0, "podcast", "podcasts y voz"),
            "streaming": (-23.0, "streaming", "broadcast profesional"),
        }
        target_lufs, preset_name, use_case = preset_map.get(preset, (-14.0, "música", "streaming y música"))
        effect_config = {
            "type": "normalize",
            "params": {"target_lufs": target_lufs, "preset_name": preset_name, "use_case": use_case}
        }
        pipeline_effects.append(effect_config)
        context.user_data["pipeline_effects"] = pipeline_effects

        reply_markup = _get_pipeline_keyboard(pipeline_effects)
        await query.edit_message_text(
            _format_pipeline_message(pipeline_effects),
            reply_markup=reply_markup
        )
        logger.info(f"[{correlation_id}] Normalize (profile={preset_name}) added to pipeline by user {user_id}")
        return

    # Handle apply pipeline
    if callback_data == "pipeline_apply":
        await _handle_pipeline_apply(update, context, pipeline_effects)
        return

    logger.warning(f"[{correlation_id}] Unknown pipeline callback: {callback_data}")


async def _handle_pipeline_apply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    pipeline_effects: list
) -> None:
    """Apply the effect pipeline and process audio.

    Args:
        update: Telegram update object
        context: Telegram context object
        pipeline_effects: List of effect configs to apply
    """
    query = update.callback_query
    user_id = update.effective_user.id
    correlation_id = context.user_data.get("pipeline_correlation_id", str(uuid.uuid4())[:8])

    # Validate pipeline
    if not pipeline_effects:
        await query.answer("No has agregado efectos. Agrega al menos uno antes de aplicar.", show_alert=True)
        return

    # Retrieve file_id from context
    file_id = context.user_data.get("pipeline_file_id")
    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for user {user_id}")
        await query.edit_message_text("Error: no se encontró el archivo de audio. Intenta de nuevo.")
        return

    # Update message to show processing
    try:
        await query.edit_message_text(f"Aplicando pipeline ({len(pipeline_effects)} efectos)...")
    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not update message: {e}")

    logger.info(f"[{correlation_id}] Applying pipeline with {len(pipeline_effects)} effects for user {user_id}")

    # Process with TempManager for automatic cleanup
    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_pipeline_{user_id}_{correlation_id}.audio"
            output_filename = f"pipeline_{user_id}_{correlation_id}.mp3"

            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download audio file
            logger.info(f"[{correlation_id}] Downloading audio from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Audio downloaded to {input_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download audio for user {user_id}: {e}")
                raise DownloadError("No pude descargar el audio") from e

            # Validate audio integrity after download
            is_valid, error_msg = validate_audio_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Audio validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space before processing (estimate based on number of effects)
            audio_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(audio_size_mb * (1 + len(pipeline_effects) * 0.5)))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"[{correlation_id}] Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            # Apply effects in chain using AudioEffects
            logger.info(f"[{correlation_id}] Processing pipeline with {len(pipeline_effects)} effects for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                effects = AudioEffects(str(input_path), str(output_path))

                # Build method chain based on pipeline_effects order
                for effect in pipeline_effects:
                    effect_type = effect.get("type")
                    params = effect.get("params", {})

                    if effect_type == "denoise":
                        strength = params.get("strength", 5)
                        await asyncio.wait_for(
                            loop.run_in_executor(None, effects.denoise, float(strength)),
                            timeout=config.PROCESSING_TIMEOUT
                        )
                    elif effect_type == "compress":
                        ratio = params.get("ratio", 4.0)
                        await asyncio.wait_for(
                            loop.run_in_executor(None, effects.compress, ratio, -20.0),
                            timeout=config.PROCESSING_TIMEOUT
                        )
                    elif effect_type == "normalize":
                        target_lufs = params.get("target_lufs", -14.0)
                        await asyncio.wait_for(
                            loop.run_in_executor(None, effects.normalize, target_lufs),
                            timeout=config.PROCESSING_TIMEOUT
                        )

                # Finalize the effect chain
                final_output = await asyncio.wait_for(
                    loop.run_in_executor(None, effects.finalize),
                    timeout=config.PROCESSING_TIMEOUT
                )

                if not final_output or not Path(final_output).exists():
                    logger.error(f"[{correlation_id}] Pipeline processing failed for user {user_id}")
                    raise AudioEffectsError("No pude procesar el pipeline de efectos")

            except asyncio.TimeoutError as e:
                logger.error(f"[{correlation_id}] Pipeline processing timed out for user {user_id}")
                raise ProcessingTimeoutError("El procesamiento del pipeline tardó demasiado") from e

            # Send processed audio
            logger.info(f"[{correlation_id}] Sending pipeline result to user {user_id}")
            try:
                with open(output_path, "rb") as audio_file:
                    await context.bot.send_audio(
                        chat_id=update.effective_chat.id,
                        audio=audio_file,
                        filename=f"pipeline_audio.mp3",
                        title=f"Audio con pipeline ({len(pipeline_effects)} efectos)"
                    )
                logger.info(f"[{correlation_id}] Pipeline result sent successfully to user {user_id}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to send pipeline result to user {user_id}: {e}")
                raise

            # Build effect list for success message
            effect_list = []
            for effect in pipeline_effects:
                effect_type = effect.get("type", "unknown")
                params = effect.get("params", {})
                if effect_type == "denoise":
                    effect_list.append(f"Denoise ({params.get('strength', 5)})")
                elif effect_type == "compress":
                    effect_list.append(f"Compress ({params.get('preset_name', 'media')})")
                elif effect_type == "normalize":
                    effect_list.append(f"Normalize ({params.get('preset_name', 'música')})")

            # Update message on success
            try:
                await query.edit_message_text(
                    f"¡Listo! Pipeline aplicado ({len(pipeline_effects)} efectos):\n"
                    + "\n".join(f"  {i+1}. {name}" for i, name in enumerate(effect_list))
                )
            except Exception as e:
                logger.warning(f"[{correlation_id}] Could not update final message: {e}")

            # Clean up user_data
            context.user_data.pop("pipeline_file_id", None)
            context.user_data.pop("pipeline_correlation_id", None)
            context.user_data.pop("pipeline_effects", None)

        except (DownloadError, ValidationError, AudioEffectsError, ProcessingTimeoutError) as e:
            # Handle known processing errors
            logger.error(f"[{correlation_id}] Pipeline processing error: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error (keep state so user can retry)
            try:
                await query.edit_message_text(f"Error: {str(e)}\n\nPuedes intentar aplicar el pipeline de nuevo.")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"[{correlation_id}] Unexpected error applying pipeline for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)

            # Update message on error
            try:
                await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")
            except Exception as edit_error:
                logger.warning(f"[{correlation_id}] Could not update error message: {edit_error}")

        # TempManager cleanup happens automatically on context exit


async def handle_video_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle video menu selections from inline keyboard.

    Routes to appropriate action based on user selection:
    - videonote: Convert video to circular video note
    - extract_audio: Show format selection for audio extraction
    - convert: Show format selection for video conversion
    - split: Show split options or prompt for parameters

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data
    correlation_id = context.user_data.get("video_menu_correlation_id", str(uuid.uuid4())[:8])

    # Parse action from callback data (format: video_action:<action>)
    if not callback_data.startswith("video_action:"):
        logger.warning(f"[{correlation_id}] Unexpected callback data: {callback_data}")
        return

    action = callback_data.split(":")[1]
    logger.info(f"[{correlation_id}] Video menu action selected: {action} by user {user_id}")

    # Retrieve file_id from context
    file_id = context.user_data.get("video_menu_file_id")
    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for user {user_id}")
        await query.edit_message_text("Error: no se encontró el video. Intenta de nuevo.")
        return

    if action == "videonote":
        # Process video to video note
        await query.edit_message_text("Procesando video a nota de video...")

        with TempManager() as temp_mgr:
            try:
                # Generate safe filenames
                input_filename = f"input_videonote_{user_id}_{correlation_id}.mp4"
                output_filename = f"videonote_{user_id}_{correlation_id}.mp4"

                input_path = temp_mgr.get_temp_path(input_filename)
                output_path = temp_mgr.get_temp_path(output_filename)

                # Download video
                logger.info(f"[{correlation_id}] Downloading video for videonote from user {user_id}")
                try:
                    file = await context.bot.get_file(file_id)
                    await _download_with_retry(file, input_path, correlation_id=correlation_id)
                    logger.info(f"[{correlation_id}] Video downloaded to {input_path}")
                except Exception as e:
                    logger.error(f"[{correlation_id}] Failed to download video for user {user_id}: {e}")
                    raise DownloadError("No pude descargar el video") from e

                # Validate video integrity
                is_valid, error_msg = validate_video_file(str(input_path))
                if not is_valid:
                    logger.warning(f"[{correlation_id}] Video validation failed for user {user_id}: {error_msg}")
                    raise ValidationError(error_msg)

                # Check disk space
                video_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
                required_space = estimate_required_space(int(video_size_mb))
                has_space, space_error = check_disk_space(required_space)
                if not has_space:
                    logger.warning(f"[{correlation_id}] Disk space check failed for user {user_id}: {space_error}")
                    raise ValidationError(space_error)

                # Process video with timeout
                logger.info(f"[{correlation_id}] Processing video to video note for user {user_id}")
                try:
                    loop = asyncio.get_event_loop()
                    success = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            VideoProcessor.process_video,
                            str(input_path),
                            str(output_path)
                        ),
                        timeout=config.PROCESSING_TIMEOUT
                    )

                    if not success:
                        logger.error(f"[{correlation_id}] Video processing failed for user {user_id}")
                        raise FFmpegError("El procesamiento de video falló")

                except asyncio.TimeoutError as e:
                    logger.error(f"[{correlation_id}] Video processing timed out for user {user_id}")
                    raise ProcessingTimeoutError("El video tardó demasiado en procesarse") from e

                # Send as video note
                logger.info(f"[{correlation_id}] Sending video note to user {user_id}")
                try:
                    with open(output_path, "rb") as video_file:
                        await query.message.reply_video_note(video_note=video_file)
                    logger.info(f"[{correlation_id}] Video note sent successfully to user {user_id}")
                except Exception as e:
                    logger.error(f"[{correlation_id}] Failed to send video note to user {user_id}: {e}")
                    raise

                # Update message to confirm completion
                await query.edit_message_text("¡Listo! Nota de video enviada.")

                # Clean up context
                context.user_data.pop("video_menu_file_id", None)
                context.user_data.pop("video_menu_correlation_id", None)

            except (DownloadError, FFmpegError, ProcessingTimeoutError, ValidationError) as e:
                logger.error(f"[{correlation_id}] Video note processing error: {e}")
                await handle_processing_error(update, e, user_id)
                await query.edit_message_text(f"Error: {str(e)}")

                # Clean up context on error
                context.user_data.pop("video_menu_file_id", None)
                context.user_data.pop("video_menu_correlation_id", None)

            except Exception as e:
                logger.exception(f"[{correlation_id}] Unexpected error processing video note for user {user_id}: {e}")
                await handle_processing_error(update, e, user_id)
                await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")

                # Clean up context on error
                context.user_data.pop("video_menu_file_id", None)
                context.user_data.pop("video_menu_correlation_id", None)

    elif action == "extract_audio":
        # Store action type and show format selection
        context.user_data["video_menu_action"] = "extract_audio"

        reply_markup = _get_video_audio_format_keyboard()
        await query.edit_message_text(
            "Selecciona el formato de audio:",
            reply_markup=reply_markup
        )
        logger.info(f"[{correlation_id}] Showing audio format selection to user {user_id}")

    elif action == "convert":
        # Store action type and show format selection
        context.user_data["video_menu_action"] = "convert"

        reply_markup = _get_video_format_keyboard()
        await query.edit_message_text(
            "Selecciona el formato de video:",
            reply_markup=reply_markup
        )
        logger.info(f"[{correlation_id}] Showing video format selection to user {user_id}")

    elif action == "split":
        # Start interactive video split process
        # Send callback data with file_id to the split handler
        await query.edit_message_text(
            "✂️ Iniciando proceso para dividir video..."
        )
        
        # Create callback data with file_id
        split_callback_data = f"video_split:{file_id}"
        
        # Store original context for later
        context.user_data["video_menu_file_id"] = file_id
        context.user_data["video_menu_correlation_id"] = correlation_id
        
        # Call the split start handler
        # Create a fake update with the callback data
        await handle_video_split_start(update, context)

    elif action == "merge_audio":
        # Store video info and prompt user to send audio
        context.user_data["merge_video_file_id"] = file_id
        context.user_data["merge_video_correlation_id"] = correlation_id
        await query.edit_message_text(
            "¡Perfecto! Ahora envíame el archivo de audio que quieres agregar al video.\n\n"
            "Puede ser MP3, WAV, OGG, AAC, etc.\n\n"
            "Envía /cancel para cancelar."
        )
        logger.info(f"[{correlation_id}] Waiting for audio file from user {user_id} for merge")

    elif action == "join":
        # Start video join session with current video as first video
        await query.edit_message_text("🎬 Iniciando sesión de unión de videos...")

        # Check if there's already an active join session
        if context.user_data.get("join_session"):
            await query.edit_message_text(
                "Ya tienes una sesión de unión de videos activa. "
                "Usa /done para unir o /cancel para cancelarla primero."
            )
            return

        # Download the current video to temp directory
        # Note: Don't use 'with' context manager as the session needs to persist
        temp_mgr = TempManager()
        try:
            input_filename = f"join_{user_id}_video01_{correlation_id}.mp4"
            input_path = temp_mgr.get_temp_path(input_filename)

            # Download video
            logger.info(f"[{correlation_id}] Downloading video for join session from user {user_id}")
            file = await context.bot.get_file(file_id)
            await _download_with_retry(file, input_path, correlation_id=correlation_id)

            # Validate video
            is_valid, error_msg = validate_video_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Video validation failed: {error_msg}")
                temp_mgr.cleanup()
                await query.edit_message_text(f"Error: {error_msg}")
                return

            # Initialize join session
            context.user_data["join_session"] = {
                "videos": [str(input_path)],
                "temp_mgr": temp_mgr,
                "last_activity": asyncio.get_event_loop().time(),
                "correlation_id": correlation_id,
            }

            # Track the file
            temp_mgr.track_file(str(input_path))

            await query.edit_message_text(
                "🎬 *Modo unión de videos activado*\n\n"
                "El video actual es el **primer video** en la lista.\n"
                "Envíame más videos para unir (máximo 10 en total).\n"
                "Los videos se unirán en el orden en que los envíes.\n\n"
                f"Actualmente tienes: *1 video*",
                parse_mode="Markdown",
                reply_markup=_get_join_video_keyboard(1)
            )
            logger.info(f"[{correlation_id}] Join session started for user {user_id} with current video")

        except Exception as e:
            logger.error(f"[{correlation_id}] Failed to start join session: {e}")
            temp_mgr.cleanup()
            await query.edit_message_text(
                "Error al iniciar la sesión de unión. Por favor intenta de nuevo."
            )

    elif action == "screenshots":
        # Store file_id and show mode selection
        context.user_data["screenshot_file_id"] = file_id
        context.user_data["screenshot_correlation_id"] = correlation_id
        context.user_data["screenshot_mode"] = None
        context.user_data["screenshot_state"] = "select_mode"

        await query.edit_message_text(
            "📸 *Capturas de Pantalla*\n\n"
            "Elige el modo de captura:\n\n"
            "🔢 *Automático*: Genera capturas uniformemente distribuidas en el video\n"
            "✏️ *Manual*: Especifica los tiempos exactos para cada captura",
            parse_mode="Markdown",
            reply_markup=_get_screenshot_mode_keyboard()
        )
        logger.info(f"[{correlation_id}] Screenshot mode selection shown to user {user_id}")

    else:
        logger.warning(f"[{correlation_id}] Unknown video menu action: {action}")
        await query.edit_message_text("Acción no reconocida. Por favor intenta de nuevo.")

        # Clean up context
        context.user_data.pop("video_menu_file_id", None)
        context.user_data.pop("video_menu_correlation_id", None)


async def handle_video_format_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle video format selection callbacks.

    Processes video conversion or audio extraction based on the
    previously stored action type and selected format.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data
    correlation_id = context.user_data.get("video_menu_correlation_id", str(uuid.uuid4())[:8])

    # Retrieve file_id and action from context
    file_id = context.user_data.get("video_menu_file_id")
    action = context.user_data.get("video_menu_action")

    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for user {user_id}")
        await query.edit_message_text("Error: no se encontró el video. Intenta de nuevo.")
        return

    if not action:
        logger.error(f"[{correlation_id}] No action found in context for user {user_id}")
        await query.edit_message_text("Error: acción no encontrada. Intenta de nuevo.")
        return

    # Parse format from callback data
    if callback_data.startswith("video_format:"):
        output_format = callback_data.split(":")[1]
    elif callback_data.startswith("video_audio_format:"):
        output_format = callback_data.split(":")[1]
    else:
        logger.warning(f"[{correlation_id}] Unexpected callback data: {callback_data}")
        return

    logger.info(f"[{correlation_id}] Format selected: {output_format} for action: {action} by user {user_id}")

    with TempManager() as temp_mgr:
        try:
            # Generate safe filenames
            input_filename = f"input_{action}_{user_id}_{correlation_id}.mp4"
            input_path = temp_mgr.get_temp_path(input_filename)

            # Download video
            logger.info(f"[{correlation_id}] Downloading video for {action} from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Video downloaded to {input_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download video for user {user_id}: {e}")
                raise DownloadError("No pude descargar el video") from e

            # Validate video integrity
            is_valid, error_msg = validate_video_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Video validation failed for user {user_id}: {error_msg}")
                raise ValidationError(error_msg)

            # Check disk space
            video_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
            required_space = estimate_required_space(int(video_size_mb))
            has_space, space_error = check_disk_space(required_space)
            if not has_space:
                logger.warning(f"[{correlation_id}] Disk space check failed for user {user_id}: {space_error}")
                raise ValidationError(space_error)

            if action == "convert":
                # Process video conversion
                await query.edit_message_text(f"Convirtiendo video a {output_format.upper()}...")

                output_filename = f"converted_{user_id}_{correlation_id}.{output_format}"
                output_path = temp_mgr.get_temp_path(output_filename)

                logger.info(f"[{correlation_id}] Converting video to {output_format} for user {user_id}")
                try:
                    loop = asyncio.get_event_loop()
                    converter = FormatConverter(str(input_path), str(output_path))
                    success = await asyncio.wait_for(
                        loop.run_in_executor(None, converter.convert, output_format),
                        timeout=config.PROCESSING_TIMEOUT
                    )

                    if not success:
                        logger.error(f"[{correlation_id}] Format conversion failed for user {user_id}")
                        raise FormatConversionError(f"No pude convertir el video a {output_format.upper()}")

                except asyncio.TimeoutError as e:
                    logger.error(f"[{correlation_id}] Format conversion timed out for user {user_id}")
                    raise ProcessingTimeoutError("La conversión tardó demasiado") from e

                # Send converted video
                logger.info(f"[{correlation_id}] Sending converted video to user {user_id}")
                try:
                    with open(output_path, "rb") as video_file:
                        await query.message.reply_video(video=video_file)
                    logger.info(f"[{correlation_id}] Converted video sent successfully to user {user_id}")
                except Exception as e:
                    logger.error(f"[{correlation_id}] Failed to send converted video to user {user_id}: {e}")
                    raise

                # Update message to confirm completion
                await query.edit_message_text(f"¡Listo! Video convertido a {output_format.upper()}.")

            elif action == "extract_audio":
                # Process audio extraction
                await query.edit_message_text(f"Extrayendo audio como {output_format.upper()}...")

                output_filename = f"audio_{user_id}_{correlation_id}.{output_format}"
                output_path = temp_mgr.get_temp_path(output_filename)

                logger.info(f"[{correlation_id}] Extracting audio as {output_format} for user {user_id}")
                try:
                    loop = asyncio.get_event_loop()
                    extractor = AudioExtractor(str(input_path), str(output_path))
                    success = await asyncio.wait_for(
                        loop.run_in_executor(None, extractor.extract, output_format),
                        timeout=config.PROCESSING_TIMEOUT
                    )

                    if not success:
                        logger.error(f"[{correlation_id}] Audio extraction failed for user {user_id}")
                        raise AudioExtractionError(f"No pude extraer el audio en formato {output_format.upper()}")

                except asyncio.TimeoutError as e:
                    logger.error(f"[{correlation_id}] Audio extraction timed out for user {user_id}")
                    raise ProcessingTimeoutError("La extracción de audio tardó demasiado") from e

                # Send extracted audio
                logger.info(f"[{correlation_id}] Sending extracted audio to user {user_id}")
                try:
                    with open(output_path, "rb") as audio_file:
                        await query.message.reply_audio(audio=audio_file)
                    logger.info(f"[{correlation_id}] Audio sent successfully to user {user_id}")
                except Exception as e:
                    logger.error(f"[{correlation_id}] Failed to send audio to user {user_id}: {e}")
                    raise

                # Update message to confirm completion
                await query.edit_message_text(f"¡Listo! Audio extraído como {output_format.upper()}.")

            # Clean up context
            context.user_data.pop("video_menu_file_id", None)
            context.user_data.pop("video_menu_correlation_id", None)
            context.user_data.pop("video_menu_action", None)

        except (DownloadError, FormatConversionError, AudioExtractionError, ProcessingTimeoutError, ValidationError) as e:
            logger.error(f"[{correlation_id}] Format selection processing error: {e}")
            await handle_processing_error(update, e, user_id)
            await query.edit_message_text(f"Error: {str(e)}")

            # Clean up context on error
            context.user_data.pop("video_menu_file_id", None)
            context.user_data.pop("video_menu_correlation_id", None)
            context.user_data.pop("video_menu_action", None)

        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error in format selection for user {user_id}: {e}")
            await handle_processing_error(update, e, user_id)
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")

            # Clean up context on error
            context.user_data.pop("video_menu_file_id", None)
            context.user_data.pop("video_menu_correlation_id", None)
            context.user_data.pop("video_menu_action", None)


async def handle_screenshot_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle screenshot-related callback queries.

    Routes to appropriate action based on callback data:
    - screenshot:auto: Show count selection keyboard
    - screenshot:manual: Prompt for timestamps
    - screenshot:back_to_mode: Return to mode selection
    - screenshot_count:*: Process automatic count selection

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data
    correlation_id = context.user_data.get("screenshot_correlation_id", str(uuid.uuid4())[:8])

    logger.info(f"[{correlation_id}] Screenshot callback: {callback_data} from user {user_id}")

    # Get file_id from context
    file_id = context.user_data.get("screenshot_file_id")
    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found for screenshots")
        await query.edit_message_text("Error: no se encontró el video. Por favor envía el video de nuevo.")
        return

    # Route based on callback data
    if callback_data == "screenshot:auto":
        # Show count selection keyboard
        context.user_data["screenshot_mode"] = "auto"
        context.user_data["screenshot_state"] = "waiting_count"

        await query.edit_message_text(
            "🔢 *Modo Automático*\n\n"
            "Selecciona la cantidad de capturas o envía un número directamente.\n\n"
            "Las capturas se distribuirán均匀e a lo largo del video.",
            parse_mode="Markdown",
            reply_markup=_get_screenshot_count_keyboard()
        )

    elif callback_data == "screenshot:manual":
        # Prompt for timestamps
        context.user_data["screenshot_mode"] = "manual"
        context.user_data["screenshot_state"] = "waiting_times"

        await query.edit_message_text(
            "✏️ *Modo Manual*\n\n"
            "Ingresa los tiempos exactos para las capturas.\n\n"
            "Formatos aceptados:\n"
            "• Minutos:segundos → `1:30`, `3:45`\n"
            "• Segundos → `85`, `160`\n"
            "• Combinados → `1:30, 85, 3:45`\n\n"
            "Ejemplo: `1:25, 2:40, 5:10`",
            parse_mode="Markdown",
            reply_markup=_get_screenshot_manual_nav_keyboard()
        )

    elif callback_data == "screenshot:back_to_mode":
        # Return to mode selection
        context.user_data["screenshot_mode"] = None
        context.user_data["screenshot_state"] = "select_mode"

        await query.edit_message_text(
            "📸 *Capturas de Pantalla*\n\n"
            "Elige el modo de captura:\n\n"
            "🔢 *Automático*: Genera capturas uniformemente distribuidas en el video\n"
            "✏️ *Manual*: Especifica los tiempos exactos para cada captura",
            parse_mode="Markdown",
            reply_markup=_get_screenshot_mode_keyboard()
        )

    elif callback_data.startswith("screenshot_count:"):
        # Process automatic count selection
        try:
            count = int(callback_data.split(":")[1])
            await _process_screenshots(update, context, count)
        except ValueError:
            await query.edit_message_text("Cantidad inválida.")
            logger.warning(f"[{correlation_id}] Invalid screenshot count: {callback_data}")


async def _process_screenshots(update: Update, context: ContextTypes.DEFAULT_TYPE, count: int) -> None:
    """Process screenshot extraction and sending.

    Args:
        update: Telegram update object
        context: Telegram context object
        count: Number of screenshots (for auto mode) or None for manual
    """
    query = update.callback_query
    correlation_id = context.user_data.get("screenshot_correlation_id", str(uuid.uuid4())[:8])
    user_id = update.effective_user.id
    file_id = context.user_data.get("screenshot_file_id")

    mode = context.user_data.get("screenshot_mode")

    # Show processing message
    processing_msg = None
    if hasattr(query, 'message') and query.message:
        try:
            processing_msg = await query.message.reply_text("⏳ Procesando capturas de pantalla...")
        except Exception:
            processing_msg = None

    with TempManager(correlation_id) as temp_mgr:
        try:
            # Download video
            input_filename = f"screenshot_input_{correlation_id}.mp4"
            input_path = temp_mgr.get_temp_path(input_filename)

            logger.info(f"[{correlation_id}] Downloading video for screenshots from user {user_id}")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
                logger.info(f"[{correlation_id}] Video downloaded to {input_path}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to download video: {e}")
                raise DownloadError("No pude descargar el video") from e

            # Validate video
            is_valid, error_msg = validate_video_file(str(input_path))
            if not is_valid:
                logger.warning(f"[{correlation_id}] Video validation failed: {error_msg}")
                raise ValidationError(error_msg)

            # Create screenshot processor
            processor = ScreenshotProcessor(str(input_path), correlation_id)

            # Extract screenshots
            if mode == "auto":
                screenshot_paths = await processor.extract_auto(count)
            elif mode == "manual":
                timestamps = context.user_data.get("screenshot_timestamps", [])
                screenshot_paths = await processor.extract_at_times(timestamps)
            else:
                raise ValueError(f"Unknown screenshot mode: {mode}")

            # Delete processing message if exists
            if processing_msg:
                try:
                    await processing_msg.delete()
                except Exception:
                    pass

            # Send screenshots in albums of 10
            await _send_screenshots_in_albums(update, context, screenshot_paths, correlation_id)

            # Cleanup
            processor.cleanup()

            # Clear screenshot context
            context.user_data.pop("screenshot_file_id", None)
            context.user_data.pop("screenshot_correlation_id", None)
            context.user_data.pop("screenshot_mode", None)
            context.user_data.pop("screenshot_state", None)
            context.user_data.pop("screenshot_timestamps", None)

            logger.info(f"[{correlation_id}] Screenshots sent successfully to user {user_id}")

        except Exception as e:
            logger.exception(f"[{correlation_id}] Screenshot processing error: {e}")
            if processing_msg:
                try:
                    await processing_msg.delete()
                except Exception:
                    pass
            error_text = str(e) if e else "Error desconocido"
            if hasattr(query, 'message') and query.message:
                await query.message.reply_text(f"❌ Error: {error_text}")


async def _send_images_in_albums(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    image_paths: list,
    correlation_id: str,
    caption_prefix: str = "Imagen",
) -> None:
    """Send images grouped in albums of max 10.

    Args:
        update: Telegram update object
        context: Telegram context object
        image_paths: List of paths to image files
        correlation_id: Correlation ID for logging
        caption_prefix: Prefix for album captions (e.g. "Imagen", "Captura")
    """
    from telegram import InputMediaPhoto

    user_id = update.effective_user.id
    total = len(image_paths)
    album_size = min(config.MAX_IMAGE_BATCH_SIZE, 10)

    logger.info(f"[{correlation_id}] Sending {total} images in albums to user {user_id}")

    for i in range(0, total, album_size):
        album_paths = image_paths[i:i + album_size]
        album_num = (i // album_size) + 1
        total_albums = (total + album_size - 1) // album_size

        file_handles = []
        try:
            media_group = []
            for j, path in enumerate(album_paths):
                global_idx = i + j + 1
                caption = f"{caption_prefix} {global_idx}/{total}" if j == 0 else None
                fh = open(path, "rb")
                file_handles.append(fh)
                media_group.append(InputMediaPhoto(media=fh, caption=caption))

            if hasattr(update, 'callback_query') and update.callback_query:
                await update.callback_query.message.reply_media_group(media=media_group)
            elif hasattr(update, 'message') and update.message:
                await update.message.reply_media_group(media=media_group)

            logger.info(
                f"[{correlation_id}] Sent album {album_num}/{total_albums} "
                f"({len(album_paths)} images)"
            )

        except Exception as e:
            logger.error(f"[{correlation_id}] Failed to send album {album_num}: {e}")
            for path in album_paths:
                try:
                    with open(path, 'rb') as f:
                        if hasattr(update, 'callback_query') and update.callback_query:
                            await update.callback_query.message.reply_photo(photo=f)
                        elif hasattr(update, 'message') and update.message:
                            await update.message.reply_photo(photo=f)
                except Exception as ex:
                    logger.error(f"[{correlation_id}] Failed to send individual image: {ex}")
        finally:
            for fh in file_handles:
                fh.close()


async def _send_screenshots_in_albums(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    screenshot_paths: list,
    correlation_id: str
) -> None:
    """Send screenshots grouped in albums of max 10.

    Args:
        update: Telegram update object
        context: Telegram context object
        screenshot_paths: List of paths to screenshot files
        correlation_id: Correlation ID for logging
    """
    await _send_images_in_albums(
        update, context, screenshot_paths, correlation_id, caption_prefix="Captura"
    )


async def _parse_screenshot_times(text: str) -> tuple[list[float], str]:
    """Parse screenshot timestamps from user input.

    Supports formats:
    - MM:SS or M:SS -> 1:30, 3:45
    - Seconds only -> 85, 160
    - Mixed/combined -> 1:30, 85, 3:45

    Args:
        text: User input text

    Returns:
        Tuple of (list of timestamps in seconds, error message or empty string)
    """
    timestamps = []
    parts = text.replace(' ', '').split(',')

    for part in parts:
        if not part:
            continue

        if ':' in part:
            # Format: MM:SS or M:SS
            try:
                if part.count(':') == 1:
                    minutes, seconds = part.split(':')
                    ts = int(minutes) * 60 + int(seconds)
                else:
                    return [], f"Formato inválido: {part}. Usa M:SS o MM:SS"
                timestamps.append(ts)
            except ValueError:
                return [], f"No pude entender el tiempo: {part}"
        else:
            # Format: seconds only
            try:
                timestamps.append(int(part))
            except ValueError:
                return [], f"No pude entender el número: {part}"

    if not timestamps:
        return [], "No se encontraron tiempos válidos. Usa el formato: 1:30, 85, 3:45"

    return timestamps, ""


async def handle_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle cancel callback from inline keyboard.

    Cleans up all user context data related to ongoing operations
    and shows a cancellation confirmation message.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]

    # Clear video menu keys
    context.user_data.pop("video_menu_file_id", None)
    context.user_data.pop("video_menu_correlation_id", None)
    context.user_data.pop("video_menu_action", None)

    # Clear audio menu keys
    context.user_data.pop("audio_menu_file_id", None)
    context.user_data.pop("audio_menu_correlation_id", None)
    context.user_data.pop("audio_menu_action", None)

    # Clear convert keys
    context.user_data.pop("convert_audio_file_id", None)
    context.user_data.pop("convert_audio_correlation_id", None)

    # Clear enhance keys
    context.user_data.pop("enhance_audio_file_id", None)
    context.user_data.pop("enhance_audio_correlation_id", None)
    context.user_data.pop("enhance_type", None)

    # Clear EQ keys
    context.user_data.pop("eq_file_id", None)
    context.user_data.pop("eq_correlation_id", None)
    context.user_data.pop("eq_bass", None)
    context.user_data.pop("eq_mid", None)
    context.user_data.pop("eq_treble", None)

    # Clear effect keys
    context.user_data.pop("effect_audio_file_id", None)
    context.user_data.pop("effect_audio_correlation_id", None)
    context.user_data.pop("effect_type", None)

    # Clear pipeline keys
    context.user_data.pop("pipeline_file_id", None)
    context.user_data.pop("pipeline_correlation_id", None)
    context.user_data.pop("pipeline_effects", None)
    context.user_data.pop("pipeline_selecting_effect", None)

    # Clear voice pipeline keys
    context.user_data.pop("voice_pipeline_correlation_id", None)

    # Clear merge keys
    context.user_data.pop("merge_video_file_id", None)
    context.user_data.pop("merge_video_correlation_id", None)

    # Clear screenshot keys
    context.user_data.pop("screenshot_file_id", None)
    context.user_data.pop("screenshot_correlation_id", None)
    context.user_data.pop("screenshot_mode", None)
    context.user_data.pop("screenshot_state", None)

    await query.edit_message_text("Operación cancelada.")
    logger.info(f"[{correlation_id}] Operation cancelled by user {user_id}")


async def handle_voice_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle cancel callback for voice processing pipeline.

    Cancels the voice note processing pipeline and cleans up resources.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Parse correlation_id from callback data
    if not callback_data.startswith("voice_cancel:"):
        logger.warning(f"Invalid voice cancel callback data: {callback_data}")
        return

    correlation_id = callback_data.split(":")[1]

    # Clear voice pipeline keys
    context.user_data.pop("voice_pipeline_correlation_id", None)

    await query.edit_message_text("❌ Procesamiento de nota de voz cancelado.")
    logger.info(f"[{correlation_id}] Voice pipeline cancelled by user {user_id}")


async def handle_back_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle back navigation callback from inline keyboard.

    Returns the user to the appropriate parent menu based on context.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Parse callback data: back:video or back:audio
    if not callback_data.startswith("back:"):
        logger.warning(f"Invalid back callback data: {callback_data}")
        return

    menu_type = callback_data.split(":")[1]

    if menu_type == "video":
        # Re-show video menu with stored file_id
        file_id = context.user_data.get("video_menu_file_id")
        if not file_id:
            await query.edit_message_text(
                "Error: no se encontró el archivo de video. Por favor envía el video de nuevo."
            )
            logger.warning(f"Back to video menu failed: no file_id for user {user_id}")
            return

        reply_markup = _get_video_menu_keyboard()
        await query.edit_message_text(
            "¿Qué quieres hacer con este video?",
            reply_markup=reply_markup
        )
        logger.info(f"User {user_id} navigated back to video menu")

    elif menu_type == "audio":
        # Re-show audio menu with stored file_id
        file_id = context.user_data.get("audio_menu_file_id")
        if not file_id:
            await query.edit_message_text(
                "Error: no se encontró el archivo de audio. Por favor envía el audio de nuevo."
            )
            logger.warning(f"Back to audio menu failed: no file_id for user {user_id}")
            return

        reply_markup = _get_audio_menu_keyboard()
        await query.edit_message_text(
            "¿Qué quieres hacer con este audio?",
            reply_markup=reply_markup
        )
        logger.info(f"User {user_id} navigated back to audio menu")

    elif menu_type == "image":
        file_ids = context.user_data.get("image_menu_file_ids")
        file_id = context.user_data.get("image_menu_file_id")
        if not file_ids and not file_id:
            await query.edit_message_text(
                "Error: no se encontró la imagen. Por favor envía la imagen de nuevo."
            )
            logger.warning(f"Back to image menu failed: no file_id for user {user_id}")
            return

        count = len(file_ids) if file_ids else 1
        truncated = context.user_data.get("image_menu_truncated", False)
        reply_markup = _get_image_menu_keyboard(count)
        if count > 1:
            menu_text = (
                f"{count} imágenes recibidas.\n\n"
                "«Mejorar», «Naturalizar» y «Agrupar» procesan todas las imágenes del álbum. "
                "Selecciona una acción:"
            )
        else:
            menu_text = "Imagen recibida. Selecciona una acción:"
        if truncated:
            menu_text += (
                f"\n\n⚠️ Solo se procesarán las primeras "
                f"{config.MAX_IMAGE_BATCH_SIZE} imágenes."
            )
        await query.edit_message_text(menu_text, reply_markup=reply_markup)
        logger.info(f"User {user_id} navigated back to image menu")

    else:
        logger.warning(f"Unknown back menu type: {menu_type}")


# =============================================================================
# URL Download Handlers
# =============================================================================

# Import PlatformRouter for metadata extraction
from bot.downloaders.platform_router import PlatformRouter

# Telegram upload limit (50MB cloud, up to 2000MB with local Bot API)
TELEGRAM_MAX_FILE_SIZE = config.telegram_max_upload_bytes


@contextmanager
def _open_file_for_send(file_path: str) -> Iterator[Any]:
    """Yield an open file handle for Telegram uploads.

    Local Bot API raises the upload limit to 2000MB via multipart uploads.
    File paths are only used when bot and API share the same filesystem.
    """
    abs_path = os.path.abspath(file_path)
    file_handle = open(abs_path, "rb")
    try:
        yield file_handle
    finally:
        file_handle.close()


def _get_download_max_filesize_mb() -> int:
    """Return the configured download size limit in megabytes."""
    return config.DOWNLOAD_MAX_SIZE_MB


def _media_input(file_path: str) -> Any:
    """Return an open file handle for Telegram media uploads."""
    return open(os.path.abspath(file_path), "rb")


def _detect_platform_for_display(url: str) -> str:
    """Detect platform name from URL for display purposes.

    Args:
        url: The URL to analyze

    Returns:
        Platform name for display, or empty string if unknown
    """
    url_lower = url.lower()

    if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return 'YouTube'
    elif 'instagram.com' in url_lower:
        return 'Instagram'
    elif 'tiktok.com' in url_lower:
        return 'TikTok'
    elif 'twitter.com' in url_lower or 'x.com' in url_lower:
        return 'Twitter/X'
    elif 'facebook.com' in url_lower or 'fb.watch' in url_lower:
        return 'Facebook'
    else:
        return ''


def _get_error_message_for_exception(e: Exception, url: str, correlation_id: str) -> str:
    """Get user-friendly error message for download exceptions.

    Handles network errors, platform-specific errors, file system errors,
    and Telegram errors with appropriate Spanish messages.

    Args:
        e: The exception that occurred
        url: The URL being downloaded
        correlation_id: Unique download ID for logging

    Returns:
        User-friendly error message in Spanish
    """
    import errno
    from telegram.error import NetworkError as TelegramNetworkError, RetryAfter, TimedOut as TelegramTimedOut

    error_msg = str(e).lower()
    platform = _detect_platform_for_display(url)

    # Network errors
    if isinstance(e, (ConnectionResetError, BrokenPipeError)):
        logger.warning(f"[{correlation_id}] Connection reset during download: {e}")
        return "La conexión se interrumpió. Intenta de nuevo."

    if isinstance(e, TimeoutError) or "timeout" in error_msg:
        logger.warning(f"[{correlation_id}] Download timeout: {e}")
        return "La descarga tardó demasiado, intenta de nuevo."

    if "dns" in error_msg or "name resolution" in error_msg or "getaddrinfo" in error_msg:
        logger.warning(f"[{correlation_id}] DNS failure: {e}")
        return "No se pudo conectar al servidor. Verifica la URL."

    # Platform-specific errors
    if platform == "YouTube":
        if "age" in error_msg or "restricted" in error_msg:
            return "Este video tiene restricción de edad."
        if "unavailable" in error_msg or "not available" in error_msg:
            return "Este video no está disponible."
        if "private" in error_msg:
            return "Este video es privado."
        if "sign in" in error_msg or "bot" in error_msg or "confirm" in error_msg:
            return "YouTube requiere verificación. Contacta al administrador para configurar cookies."
        if "cookie" in error_msg or "authentication" in error_msg:
            return "YouTube requiere autenticación. Contacta al administrador."

    if platform == "Instagram":
        if "private" in error_msg:
            return "Este contenido de Instagram es privado."
        if "story" in error_msg and ("expired" in error_msg or "unavailable" in error_msg):
            return "Esta historia de Instagram ha expirado."
        if "login" in error_msg or "authent" in error_msg:
            return "Este contenido requiere inicio de sesión en Instagram."

    if platform == "TikTok":
        if "unexpected response" in error_msg:
            return (
                "TikTok cambió su estructura y no está disponible temporalmente. "
                "Intenta de nuevo más tarde o descarga el video manualmente."
            )
        if "slideshow" in error_msg or "carousel" in error_msg:
            return "Los slideshows de TikTok no son soportados."
        if "watermark" in error_msg:
            return "No se pudo descargar el video sin marca de agua."

    if platform == "Twitter/X":
        if "restricted" in error_msg or "sensitive" in error_msg:
            return "Este contenido está restringido."
        if "deleted" in error_msg or "not found" in error_msg:
            return "Este tweet no existe o fue eliminado."
        if "suspended" in error_msg:
            return "Esta cuenta de X/Twitter está suspendida."

    if platform == "Facebook":
        if "login" in error_msg or "authent" in error_msg:
            return "Este video requiere inicio de sesión en Facebook."
        if "private" in error_msg:
            return "Este video de Facebook es privado."

    # File system errors
    if isinstance(e, OSError):
        if e.errno == errno.ENOSPC:
            logger.error(f"[{correlation_id}] Disk full: {e}")
            return "No hay espacio suficiente en el servidor."
        if e.errno == errno.EACCES or e.errno == errno.EPERM:
            logger.error(f"[{correlation_id}] Permission denied: {e}")
            return "Error de permisos al guardar archivo."
        if e.errno == errno.ENOSPC or "no space" in error_msg:
            logger.error(f"[{correlation_id}] Disk full: {e}")
            return "No hay espacio suficiente en el servidor."

    # Telegram errors (TimedOut is a NetworkError subclass — check it first)
    if isinstance(e, TelegramTimedOut):
        logger.warning(f"[{correlation_id}] Telegram timeout: {e}")
        return "El envío tardó demasiado, intenta de nuevo."

    if isinstance(e, TelegramNetworkError):
        logger.warning(f"[{correlation_id}] Telegram network error: {e}")
        return "Error de red al enviar el archivo, intenta de nuevo."

    if isinstance(e, RetryAfter):
        retry_after = getattr(e, 'retry_after', 30)
        logger.warning(f"[{correlation_id}] Rate limited: retry after {retry_after}s")
        return f"Demasiadas solicitudes, espera {retry_after} segundos."

    # File too large for Telegram
    if "file is too big" in error_msg or "too large" in error_msg or "entity too large" in error_msg:
        logger.warning(f"[{correlation_id}] File too large for Telegram: {e}")
        max_mb = config.TELEGRAM_MAX_UPLOAD_SIZE_MB
        return f"El archivo excede el límite de Telegram ({max_mb}MB)."

    # Generic download errors
    if "404" in error_msg or "not found" in error_msg:
        return "No se encontró el contenido en la URL proporcionada."

    if "403" in error_msg or "forbidden" in error_msg:
        return "Acceso denegado al contenido."

    # Default error
    logger.error(f"[{correlation_id}] Unhandled error: {type(e).__name__}: {e}")
    return "Ocurrió un error inesperado. Por favor intenta de nuevo."


def _get_download_format_keyboard(correlation_id: str, url_metadata: dict = None) -> InlineKeyboardMarkup:
    """Generate inline keyboard for download format selection.

    Args:
        correlation_id: Unique ID for this download request
        url_metadata: Optional metadata about the URL (platform, content type, etc.)

    Returns:
        InlineKeyboardMarkup with video/audio options and combined actions
    """
    # Determine available options based on content type
    is_video_content = True  # Default to showing video options
    is_audio_content = False

    if url_metadata:
        # Check if content is audio-only (e.g., YouTube audio, SoundCloud)
        content_type = url_metadata.get('content_type', 'video')
        is_audio_content = content_type == 'audio' or url_metadata.get('is_audio_only', False)
        is_video_content = not is_audio_content or url_metadata.get('has_video', True)

    keyboard = []

    # Basic download options
    basic_row = []
    if is_video_content:
        basic_row.append(InlineKeyboardButton("Video", callback_data=f"download:video:{correlation_id}"))
    if is_audio_content or is_video_content:
        basic_row.append(InlineKeyboardButton("Audio", callback_data=f"download:audio:{correlation_id}"))
    if basic_row:
        keyboard.append(basic_row)

    # Combined action options (video content only)
    if is_video_content:
        keyboard.append([
            InlineKeyboardButton("Video + Nota de Video", callback_data=f"download:video:videonote:{correlation_id}"),
            InlineKeyboardButton("Video + Extraer Audio", callback_data=f"download:video:extract:{correlation_id}"),
        ])

    # Combined action options for audio
    if is_audio_content or is_video_content:
        keyboard.append([
            InlineKeyboardButton("Audio + Nota de Voz", callback_data=f"download:audio:voicenote:{correlation_id}"),
        ])

    keyboard.append([InlineKeyboardButton("❌ Cancelar", callback_data="cancel")])

    return InlineKeyboardMarkup(keyboard)


def _get_large_download_confirmation_keyboard(correlation_id: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard for large download confirmation.

    Args:
        correlation_id: Unique ID for this download request

    Returns:
        InlineKeyboardMarkup with confirm/cancel options
    """
    keyboard = [
        [
            InlineKeyboardButton("Confirmar Descarga", callback_data=f"download:confirm:{correlation_id}"),
            InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_download_cancel_keyboard(correlation_id: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard with cancel button for active download.

    Args:
        correlation_id: Unique ID for this download request

    Returns:
        InlineKeyboardMarkup with cancel button
    """
    keyboard = [
        [
            InlineKeyboardButton("❌ Cancelar Descarga", callback_data=f"download:cancel:{correlation_id}"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


async def handle_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /download command to download a URL.

    Usage: /download <url>

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]

    # Parse URL from command arguments
    args = context.args
    if not args:
        await update.message.reply_text(
            "Por favor proporciona una URL para descargar.\n"
            "Ejemplo: /download https://youtube.com/watch?v=..."
        )
        return

    url = args[0]

    # Validate URL
    if not url_detector.validate_url(url):
        await update.message.reply_text(
            "La URL proporcionada no parece válida.\n"
            "Asegúrate de incluir http:// o https://"
        )
        return

    # Check if URL is supported
    if not url_detector.is_supported(url):
        await update.message.reply_text(
            "Esta URL no parece ser un video soportado.\n"
            "Soporto YouTube, Instagram, TikTok, Twitter/X, Facebook y URLs directas de video."
        )
        return

    logger.info(f"[{correlation_id}] Download command from user {user_id}: {url}")

    # Store URL and correlation_id in context
    context.user_data[f"download_url_{correlation_id}"] = url
    context.user_data[f"download_correlation_id_{user_id}"] = correlation_id

    # Show format selection menu
    reply_markup = _get_download_format_keyboard(correlation_id)
    await update.message.reply_text(
        "Selecciona formato:\n"
        "- Video: Solo descargar video\n"
        "- Audio: Solo extraer audio\n"
        "- Video + Nota de Video: Descargar y convertir a nota circular\n"
        "- Video + Extraer Audio: Descargar y extraer audio\n"
        "- Audio + Nota de Voz: Descargar y convertir a nota de voz",
        reply_markup=reply_markup
    )


async def handle_url_detection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle URL detection in regular text messages.

    Detects URLs and starts download immediately without showing a menu.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    if await _try_collect_caption_for_group_session(update, context):
        return

    message_text = update.message.text
    user_id = update.effective_user.id

    # Detect URLs in message
    urls = url_detector.extract_urls(message_text, update.message.entities)
    if not urls:
        # No URLs, delegate to split text input handler
        await handle_split_text_input(update, context)
        return

    # Process first URL
    url = urls[0]

    # Validate URL
    if not url_detector.validate_url(url):
        # Not a valid URL, delegate to split text input handler
        await handle_split_text_input(update, context)
        return

    # Check if URL is a valid http/https URL
    # Note: We accept any URL here because yt-dlp supports 1000+ sites
    # The actual download will fail gracefully if the URL is not supported
    url_type = url_detector.classify_url(url)
    if url_type == URLType.UNKNOWN:
        # For unknown URLs, we'll still try to process them with yt-dlp
        # yt-dlp has a generic extractor that works with many sites
        logger.debug(f"URL type UNKNOWN, will attempt generic extraction: {url}")

    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] URL detected in message from user {user_id}: {url}")

    # Store URL and correlation_id in context
    context.user_data[f"download_url_{correlation_id}"] = url
    context.user_data[f"download_correlation_id_{user_id}"] = correlation_id

    # YouTube URLs get a menu: screenshots or download
    if is_youtube_url(url):
        logger.info(f"[{correlation_id}] YouTube URL detected, showing menu")
        keyboard = [
            [
                InlineKeyboardButton("📸 Capturas", callback_data=f"youtube:screenshots:{correlation_id}"),
                InlineKeyboardButton("📥 Descargar Video", callback_data=f"youtube:download:{correlation_id}"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Link de YouTube detectado. ¿Qué quieres hacer?",
            reply_markup=reply_markup
        )
        return

    # Start download immediately (no menu)
    await _start_download_from_message(update, context, correlation_id, url)


async def _handle_youtube_auto_screenshot_flow(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    url: str,
    correlation_id: str
) -> None:
    """Handle YouTube URLs automatically: download + 10 screenshots + send + cleanup.

    When a YouTube URL is detected, this function automatically downloads the video,
    extracts 10 evenly distributed screenshots, sends them to the user,
    and cleans up all temporary files — no questions asked.
    """
    # Support both direct messages and callback queries
    is_callback = bool(update.callback_query)
    message = update.callback_query.message if is_callback else update.message
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if is_callback:
        query = update.callback_query
        await query.edit_message_text("Procesando solicitud...")
        # Use query.message for subsequent replies
        message = query.message

    facade = DownloadFacade()

    try:
        await facade.start()

        # Store for cancellation support (matches cancel handler expectations)
        context.user_data[f"download_facade_{correlation_id}"] = facade
        context.user_data[f"download_format_{correlation_id}"] = "video"
        context.user_data[f"download_status_{correlation_id}"] = "downloading"

        # Send initial progress message with cancel button
        reply_markup = _get_download_cancel_keyboard(correlation_id)
        progress_message = await message.reply_text(
            "Descargando video de YouTube...",
            reply_markup=reply_markup
        )

        # Progress tracking with rate limiting
        last_message_text = ["Descargando video de YouTube..."]
        last_update_time = [0.0]

        async def progress_callback(progress: dict) -> None:
            from bot.downloaders.progress_tracker import format_progress_message

            status = progress.get('status', 'downloading')

            current_time = time.time()
            if current_time - last_update_time[0] < 1.0 and status == 'downloading':
                return

            if status == 'downloading':
                msg = f"Descargando video de YouTube...\n{format_progress_message(progress)}"
                if msg != last_message_text[0]:
                    try:
                        await progress_message.edit_text(msg, reply_markup=reply_markup)
                        last_message_text[0] = msg
                        last_update_time[0] = current_time
                    except Exception:
                        pass

            elif status == 'waiting':
                wait_msg = progress.get('message', 'Aplicando delay...')
                try:
                    await progress_message.edit_text(wait_msg, reply_markup=reply_markup)
                except Exception:
                    pass

            elif status == 'completed':
                try:
                    await progress_message.edit_text("Descarga completada, procesando...")
                except Exception:
                    pass

            elif status == 'error':
                error_msg = progress.get('error', 'Error desconocido')
                try:
                    await progress_message.edit_text(f"Error en descarga: {error_msg}")
                except Exception:
                    pass

        # Create progress tracker
        from bot.downloaders.progress_tracker import ProgressTracker
        tracker = ProgressTracker(
            min_update_interval=3.0,
            min_percent_change=5.0,
            on_update=lambda p: asyncio.create_task(progress_callback(p))
        )

        # Download the video (keep file on success for screenshot extraction)
        config_overrides = {
            'extract_audio': False,
            'cleanup_on_success': False,
        }

        result = await facade.download(
            url=url,
            chat_id=chat_id,
            config_overrides=config_overrides
        )

        if not result.success:
            context.user_data[f"download_status_{correlation_id}"] = "error"
            await progress_message.edit_text(
                f"Error al descargar el video: {getattr(result, 'error_message', 'Error desconocido')}"
            )
            return

        context.user_data[f"download_status_{correlation_id}"] = "completed"

        if not result.file_path:
            await progress_message.edit_text("Error: No se encontró el archivo descargado")
            return

        video_path = result.file_path

        # Update message to show screenshot extraction is starting
        try:
            await progress_message.edit_text("Generando capturas de pantalla...")
        except Exception:
            pass

        # Extract 10 evenly distributed screenshots
        processor = ScreenshotProcessor(video_path, correlation_id)
        try:
            screenshot_paths = await processor.extract_auto(10)
        except Exception as e:
            logger.error(f"[{correlation_id}] Screenshot extraction failed: {e}")
            await progress_message.edit_text(f"Error al generar capturas: {str(e)}")
            processor.cleanup()
            return

        # Delete progress message before sending results
        try:
            await progress_message.delete()
        except Exception:
            pass

        # Send the 10 screenshots in albums
        await _send_screenshots_in_albums(update, context, screenshot_paths, correlation_id)

        logger.info(f"[{correlation_id}] YouTube auto screenshot flow completed for user {user_id}")

    except Exception as e:
        logger.error(f"[{correlation_id}] YouTube auto screenshot error: {type(e).__name__}: {e}")
        error_name = type(e).__name__
        try:
            await message.reply_text(
                f"Error inesperado: {error_name}. Intenta de nuevo más tarde."
            )
        except Exception:
            pass
    finally:
        # Clean up user_data
        context.user_data.pop(f"download_facade_{correlation_id}", None)
        context.user_data.pop(f"download_status_{correlation_id}", None)
        context.user_data.pop(f"download_format_{correlation_id}", None)
        # Keep download_url_{correlation_id} for potential retry

        # Stop facade (stops download manager)
        try:
            await facade.stop()
        except Exception:
            pass

        # Clean up downloaded video file
        if 'video_path' in locals() and video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.debug(f"[{correlation_id}] Cleaned up video file: {video_path}")
            except Exception as e:
                logger.warning(f"[{correlation_id}] Failed to clean up video file: {e}")

        # Clean up screenshot artifacts if processor was created
        if 'processor' in locals():
            try:
                processor.cleanup()
            except Exception:
                pass


async def handle_youtube_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle YouTube URL menu selection.

    Routes to screenshots or download based on user choice.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Parse callback data: youtube:screenshots:correlation_id or youtube:download:correlation_id
    callback_data = query.data
    if not callback_data.startswith("youtube:"):
        return

    parts = callback_data.split(":")
    if len(parts) != 3:
        return

    action = parts[1]
    correlation_id = parts[2]

    url = context.user_data.get(f"download_url_{correlation_id}")
    if not url:
        await query.edit_message_text("Error: No se encontró la URL. Intenta de nuevo.")
        return

    logger.info(f"[{correlation_id}] YouTube menu action '{action}' selected by user {user_id}")

    if action == "screenshots":
        context.user_data[f"download_format_{correlation_id}"] = "video"
        await _handle_youtube_auto_screenshot_flow(update, context, url, correlation_id)

    elif action == "download":
        context.user_data[f"download_format_{correlation_id}"] = "video"
        await _start_download(update, context, correlation_id, url, "video")


async def handle_download_format_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle format selection callback for downloads.

    Parses callback data, retrieves URL, checks file size,
    and either shows confirmation or starts download.
    Supports both simple format selection and combined download+process actions.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Parse callback data:
    # - download:format:correlation_id (simple)
    # - download:format:action:correlation_id (combined)
    if not callback_data.startswith("download:video:") and not callback_data.startswith("download:audio:"):
        logger.warning(f"Unexpected callback data: {callback_data}")
        return

    parts = callback_data.split(":")
    if len(parts) not in (3, 4):
        logger.warning(f"Invalid callback data format: {callback_data}")
        return

    format_type = parts[1]  # video or audio
    correlation_id = parts[-1]  # Last part is always correlation_id
    post_action = parts[2] if len(parts) == 4 else None  # videonote, extract, voicenote

    # Retrieve URL from context
    url = context.user_data.get(f"download_url_{correlation_id}")
    if not url:
        await query.edit_message_text(
            "Error: No se encontró la URL. Intenta de nuevo."
        )
        return

    if post_action:
        logger.info(f"[{correlation_id}] Combined action selected: {format_type} + {post_action} by user {user_id}")
    else:
        logger.info(f"[{correlation_id}] Format selected: {format_type} by user {user_id}")

    # Store format preference and post-action
    context.user_data[f"download_format_{correlation_id}"] = format_type
    if post_action:
        context.user_data[f"download_post_action_{correlation_id}"] = post_action

    # Check file size before downloading
    await query.edit_message_text("Analizando tamaño del archivo...")

    try:
        # Extract metadata using PlatformRouter
        from bot.downloaders import DownloadOptions
        router = PlatformRouter()
        route_result = await router.route(url)
        options = DownloadOptions(output_path="/tmp")
        metadata = await route_result.downloader.extract_metadata(url, options)

        # Get file size
        size = metadata.get('filesize') or metadata.get('filesize_approx', 0)

        # Store metadata for later use
        context.user_data[f"download_meta_{correlation_id}"] = metadata

        if size and size > TELEGRAM_MAX_FILE_SIZE:
            # Large file - show confirmation
            size_mb = size / (1024 * 1024)
            logger.info(f"[{correlation_id}] Large file detected: {size_mb:.1f} MB")

            # For combined actions, note that processing may change size
            action_note = ""
            if post_action:
                action_note = "\nNota: El procesamiento posterior puede cambiar el tamaño."

            reply_markup = _get_large_download_confirmation_keyboard(correlation_id)
            await query.edit_message_text(
                f"El archivo es grande (~{size_mb:.1f} MB).{action_note}\n\n"
                f"Esto puede tomar tiempo y consumir datos.\n"
                f"¿Deseas continuar?",
                reply_markup=reply_markup
            )
        else:
            # Small file or unknown size - proceed directly
            if size:
                size_mb = size / (1024 * 1024)
                logger.info(f"[{correlation_id}] File size: {size_mb:.1f} MB - proceeding")
            else:
                logger.info(f"[{correlation_id}] Unknown file size - proceeding")

            # Start download (combined flow if post_action specified)
            if post_action:
                await _start_combined_download(update, context, correlation_id, url, format_type, post_action)
            else:
                await _start_download(update, context, correlation_id, url, format_type)

    except Exception as e:
        logger.warning(f"[{correlation_id}] Could not get metadata: {e}")
        # If we can't get metadata, proceed anyway (will fail during download if too large)
        if post_action:
            await _start_combined_download(update, context, correlation_id, url, format_type, post_action)
        else:
            await _start_download(update, context, correlation_id, url, format_type)


async def handle_download_confirm_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle confirmation callback for large downloads.

    Starts the download after user confirms.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Parse callback data: download:confirm:correlation_id
    if not callback_data.startswith("download:confirm:"):
        return

    correlation_id = callback_data.split(":")[2]

    # Retrieve URL and format from context
    url = context.user_data.get(f"download_url_{correlation_id}")
    format_type = context.user_data.get(f"download_format_{correlation_id}", "video")

    if not url:
        await query.edit_message_text(
            "Error: No se encontró la información de descarga. Intenta de nuevo."
        )
        return

    logger.info(f"[{correlation_id}] Large download confirmed by user {user_id}")

    # Check for combined action
    post_action = context.user_data.get(f"download_post_action_{correlation_id}")

    # Start download (combined flow if post_action specified)
    if post_action:
        await _start_combined_download(update, context, correlation_id, url, format_type, post_action)
    else:
        await _start_download(update, context, correlation_id, url, format_type)


async def _start_download(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    correlation_id: str,
    url: str,
    format_type: str
) -> None:
    """Start the download process with progress updates.

    Args:
        update: Telegram update object
        context: Telegram context object
        correlation_id: Unique download ID
        url: URL to download
        format_type: 'video' or 'audio'
    """
    query = update.callback_query
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Detect platform for display
    platform = _detect_platform_for_display(url)

    # Create facade
    facade = DownloadFacade()

    try:
        await facade.start()

        # Store facade instance for cancellation support
        context.user_data[f"download_facade_{correlation_id}"] = facade
        context.user_data[f"download_url_{correlation_id}"] = url
        context.user_data[f"download_format_{correlation_id}"] = format_type
        context.user_data[f"download_status_{correlation_id}"] = "downloading"

        # Initial message with cancel button
        reply_markup = _get_download_cancel_keyboard(correlation_id)
        await query.edit_message_text(
            f"Analizando enlace de {platform}...",
            reply_markup=reply_markup
        )

        # Progress tracking with enhanced state management
        last_message_text = [f"Analizando enlace de {platform}..."]
        last_update_time = [0.0]  # Track last update time for rate limiting

        async def progress_callback(progress: dict) -> None:
            """Update download progress message with cancel button."""
            import time
            from bot.downloaders.progress_tracker import format_progress_message

            status = progress.get('status', 'downloading')
            percent = progress.get('percent', 0)

            # Rate limiting: only update every 1 second minimum
            current_time = time.time()
            if current_time - last_update_time[0] < 1.0 and status == 'downloading':
                return

            # Format message based on status
            if status == 'downloading':
                message = format_progress_message(progress)
                # Add platform info to message
                if platform:
                    message = f"Descargando de {platform}...\n{message}"

                if message != last_message_text[0]:
                    try:
                        await query.edit_message_text(
                            message,
                            reply_markup=reply_markup
                        )
                        last_message_text[0] = message
                        last_update_time[0] = current_time
                    except Exception as e:
                        logger.debug(f"Failed to update progress message: {e}")

            elif status == 'waiting':
                wait_msg = progress.get('message', 'Aplicando delay...')
                try:
                    await query.edit_message_text(wait_msg, reply_markup=reply_markup)
                except Exception:
                    pass

            elif status == 'completed':
                # Remove cancel button, show completed
                try:
                    await query.edit_message_text("Descarga completada")
                    context.user_data[f"download_status_{correlation_id}"] = "completed"
                except Exception:
                    pass

            elif status == 'error':
                error_msg = progress.get('error', 'Error desconocido')
                try:
                    await query.edit_message_text(f"Error: {error_msg}")
                    context.user_data[f"download_status_{correlation_id}"] = "error"
                except Exception:
                    pass

        # Create progress tracker with callback
        from bot.downloaders.progress_tracker import ProgressTracker
        tracker = ProgressTracker(
            min_update_interval=3.0,
            min_percent_change=5.0,
            on_update=lambda p: asyncio.create_task(progress_callback(p))
        )

        # Download with progress callback integration
        # IMPORTANT: cleanup_on_success=False so file remains for sending
        config_overrides = {
            'extract_audio': (format_type == 'audio'),
            'cleanup_on_success': False,
            'max_filesize_mb': _get_download_max_filesize_mb(),
        }

        # Capture pre-delay text to avoid "Analizando" -> "Descargando" jump on restore (Issue 11)
        pre_delay_text = last_message_text[0]

        # Apply Instagram inter-download delay before starting (notify user during wait)
        # (covers /download command + IG button flows for consistent behavior)
        if is_instagram_url(url):
            waited = await _apply_instagram_delay()
            if waited > 0:
                try:
                    await query.edit_message_text(
                        f"Aplicando delay de {waited:.1f}s para evitar detección de Instagram...",
                        reply_markup=reply_markup
                    )
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Failed to show Instagram delay notification: {e}")
                await asyncio.sleep(waited)
                # Restore normal message for seamless transition
                try:
                    await query.edit_message_text(
                        pre_delay_text,
                        reply_markup=reply_markup
                    )
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Failed to restore post-delay message: {e}")

        # Guard: respect cancel that arrived during the (potentially long) delay sleep (Issue 1).
        # Uses the exact download_status user_data pattern from handle_download_cancel_callback.
        if context.user_data.get(f"download_status_{correlation_id}") == "cancelled":
            return

        # Note on Issue 5: any hypothetical exception in this narrow window after _apply
        # (but before facade reaches InstagramDownloader.download's finally) would skip
        # the timestamp _mark. This is best-effort; normal errors are handled inside
        # the downloader (which always marks). The window is tiny and pre-existing
        # for waited==0 IG cases.
        result = await facade.download(
            url=url,
            chat_id=chat_id,
            config_overrides=config_overrides
        )

        if result.success:
            context.user_data[f"download_status_{correlation_id}"] = "completed"

            # Send downloaded file
            await _send_downloaded_file_with_menu(update, context, result, format_type, correlation_id)

            # Clean up status message
            try:
                await query.delete_message()
            except Exception:
                pass
        else:
            context.user_data[f"download_status_{correlation_id}"] = "error"
            await query.edit_message_text(
                f"Error en la descarga: {getattr(result, 'error_message', 'Error desconocido')}"
            )

    except FileTooLargeError as e:
        logger.warning(f"[{correlation_id}] File too large: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        await query.edit_message_text(e.to_user_message())
    except URLValidationError as e:
        logger.warning(f"[{correlation_id}] URL validation error: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        await query.edit_message_text(e.to_user_message())
    except UnsupportedURLError as e:
        logger.warning(f"[{correlation_id}] Unsupported URL: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        await query.edit_message_text(e.to_user_message())
    except DownloadError as e:
        logger.error(f"[{correlation_id}] Download error: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        error_msg = _get_error_message_for_exception(e, url, correlation_id)
        await query.edit_message_text(error_msg)
    except Exception as e:
        logger.error(f"[{correlation_id}] Unexpected error: {type(e).__name__}: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        error_msg = _get_error_message_for_exception(e, url, correlation_id)
        await query.edit_message_text(error_msg)
    finally:
        # Clean up facade reference but keep status for /downloads command
        context.user_data.pop(f"download_facade_{correlation_id}", None)
        try:
            await facade.stop()
        except Exception:
            pass


async def _start_download_from_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    correlation_id: str,
    url: str
) -> None:
    """Start download directly from a message (no menu).

    Downloads content immediately with default format (video),
    then shows post-download menu for further processing.

    Args:
        update: Telegram update object
        context: Telegram context object
        correlation_id: Unique download ID
        url: URL to download
    """
    message = update.message
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    format_type = "video"  # Default: download as video (works for photos too)

    # Detect platform for display
    platform = _detect_platform_for_display(url)

    # Create facade
    facade = DownloadFacade()

    try:
        await facade.start()

        # Store facade instance for cancellation support
        context.user_data[f"download_facade_{correlation_id}"] = facade
        context.user_data[f"download_url_{correlation_id}"] = url
        context.user_data[f"download_format_{correlation_id}"] = format_type
        context.user_data[f"download_status_{correlation_id}"] = "downloading"

        # Send initial progress message with cancel button
        reply_markup = _get_download_cancel_keyboard(correlation_id)
        progress_message = await message.reply_text(
            f"Descargando de {platform}...",
            reply_markup=reply_markup
        )

        # Progress tracking with enhanced state management
        last_message_text = [f"Descargando de {platform}..."]
        last_update_time = [0.0]  # Track last update time for rate limiting

        async def progress_callback(progress: dict) -> None:
            """Update download progress message with cancel button."""
            import time
            from bot.downloaders.progress_tracker import format_progress_message

            status = progress.get('status', 'downloading')
            percent = progress.get('percent', 0)

            # Rate limiting: only update every 1 second minimum
            current_time = time.time()
            if current_time - last_update_time[0] < 1.0 and status == 'downloading':
                return

            # Format message based on status
            if status == 'downloading':
                message_text = format_progress_message(progress)
                # Add platform info to message
                if platform:
                    message_text = f"Descargando de {platform}...\n{message_text}"

                if message_text != last_message_text[0]:
                    try:
                        await progress_message.edit_text(
                            message_text,
                            reply_markup=reply_markup
                        )
                        last_message_text[0] = message_text
                        last_update_time[0] = current_time
                    except Exception as e:
                        logger.debug(f"Failed to update progress message: {e}")

            elif status == 'waiting':
                wait_msg = progress.get('message', 'Aplicando delay...')
                try:
                    await progress_message.edit_text(wait_msg, reply_markup=reply_markup)
                except Exception:
                    pass

            elif status == 'completed':
                # Remove cancel button, show completed
                try:
                    await progress_message.edit_text("Descarga completada")
                    context.user_data[f"download_status_{correlation_id}"] = "completed"
                except Exception:
                    pass

            elif status == 'error':
                error_msg = progress.get('error', 'Error desconocido')
                try:
                    await progress_message.edit_text(f"Error: {error_msg}")
                    context.user_data[f"download_status_{correlation_id}"] = "error"
                except Exception:
                    pass

        # Create progress tracker with callback
        from bot.downloaders.progress_tracker import ProgressTracker
        tracker = ProgressTracker(
            min_update_interval=3.0,
            min_percent_change=5.0,
            on_update=lambda p: asyncio.create_task(progress_callback(p))
        )

        # Download with progress callback integration
        # IMPORTANT: cleanup_on_success=False so file remains for sending
        config_overrides = {
            'extract_audio': False,  # Download video by default
            'cleanup_on_success': False,
        }

        # Apply Instagram inter-download delay before starting (notify user during wait)
        if is_instagram_url(url):
            waited = await _apply_instagram_delay()
            if waited > 0:
                try:
                    await progress_message.edit_text(
                        f"Aplicando delay de {waited:.1f}s para evitar detección de Instagram...",
                        reply_markup=reply_markup,
                    )
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Failed to show Instagram delay notification: {e}")
                await asyncio.sleep(waited)
                # Restore normal start message for seamless transition after delay
                try:
                    await progress_message.edit_text(
                        f"Descargando de {platform}...",
                        reply_markup=reply_markup,
                    )
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Failed to restore post-delay message: {e}")

        # Guard: respect cancel that arrived during the (potentially long) delay sleep (Issue 1).
        # Uses the exact download_status user_data pattern from handle_download_cancel_callback.
        if context.user_data.get(f"download_status_{correlation_id}") == "cancelled":
            return

        # Note on Issue 5: any hypothetical exception in this narrow window after _apply
        # (but before facade reaches InstagramDownloader.download's finally) would skip
        # the timestamp _mark. This is best-effort; normal errors are handled inside
        # the downloader (which always marks). The window is tiny and pre-existing
        # for waited==0 IG cases.
        result = await facade.download(
            url=url,
            chat_id=chat_id,
            config_overrides=config_overrides
        )

        if result.success:
            context.user_data[f"download_status_{correlation_id}"] = "completed"

            # Update progress message to "Sending..." instead of deleting
            # This keeps the user informed during slow uploads
            try:
                await progress_message.edit_text("Enviando archivo...")
            except Exception:
                pass  # Message might be deleted or expired, that's ok

            # Send downloaded file
            # Don't show error for timeout - file might have been sent already
            # Telegram has a 20-second timeout for large file uploads
            try:
                await _send_downloaded_file_with_menu(update, context, result, format_type, correlation_id)
            except Exception as send_error:
                error_name = type(send_error).__name__
                logger.error(f"[{correlation_id}] Failed to send file: {send_error}")
                # Only notify user for non-timeout errors
                # For timeouts, the file likely arrived despite the error
                if error_name not in ('TimedOut', 'TimeoutError'):
                    try:
                        await message.reply_text(
                            "El archivo se descargó pero hubo un problema al enviarlo. "
                            "Intenta de nuevo o usa /download con la misma URL."
                        )
                    except Exception:
                        pass
                else:
                    logger.info(f"[{correlation_id}] Timeout during send - file may have been sent already")

        else:
            context.user_data[f"download_status_{correlation_id}"] = "error"
            await progress_message.edit_text(
                f"Error en la descarga: {getattr(result, 'error_message', 'Error desconocido')}"
            )

    except FileTooLargeError as e:
        logger.warning(f"[{correlation_id}] File too large: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        await message.reply_text(e.to_user_message())
    except URLValidationError as e:
        logger.warning(f"[{correlation_id}] URL validation error: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        await message.reply_text(e.to_user_message())
    except UnsupportedURLError as e:
        logger.warning(f"[{correlation_id}] Unsupported URL: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        await message.reply_text(e.to_user_message())
    except DownloadError as e:
        logger.error(f"[{correlation_id}] Download error: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        error_msg = _get_error_message_for_exception(e, url, correlation_id)
        await message.reply_text(error_msg)
    except Exception as e:
        logger.error(f"[{correlation_id}] Unexpected error: {type(e).__name__}: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        # Don't show error for timeout - file might have been sent already
        # Telegram has a 20-second timeout for large file uploads
        error_name = type(e).__name__
        if error_name in ('TimedOut', 'TimeoutError'):
            logger.info(f"[{correlation_id}] Timeout during send - file may have been sent already")
            # Don't send error message - file likely arrived
        else:
            # Only send error message if we haven't already sent something
            # This prevents duplicate messages when the error is from sending
            try:
                await message.reply_text(
                    f"Error inesperado: {error_name}. Intenta de nuevo mÃ¡s tarde."
                )
            except Exception:
                pass
    finally:
        # Clean up facade reference but keep status for /downloads command
        context.user_data.pop(f"download_facade_{correlation_id}", None)
        try:
            await facade.stop()
        except Exception:
            pass


def _build_caption_from_metadata(metadata: dict, default_title: str = "Descarga") -> str:
    """Build caption from metadata, using original caption if available.

    For Instagram posts, uses the original caption from the post.
    Falls back to default message if no caption is available.
    Truncates to Telegram's caption limit (1024 characters).

    Args:
        metadata: Download metadata dictionary
        default_title: Default title to use if no caption available

    Returns:
        Caption string for Telegram message
    """
    # Debug: log available metadata keys
    logger.debug(f"[caption_builder] Metadata keys: {list(metadata.keys())}")

    # Try to get original caption from metadata
    # gallery-dl uses 'description', yt-dlp uses 'caption'
    caption = (metadata.get("caption") or metadata.get("description") or "").strip()
    username = (metadata.get("username") or "").strip() or (metadata.get("uploader") or "").strip()

    logger.debug(f"[caption_builder] Extracted caption: {caption[:50] if caption else 'None'}...")
    logger.debug(f"[caption_builder] Extractor: {metadata.get('extractor')}, Username: {username}")

    # Build caption with username prefix for Instagram
    if caption:
        # For Instagram, prefix with username if available
        if username and metadata.get("extractor") == "instagram":
            full_caption = f"@{username}:\n{caption}"
        else:
            full_caption = caption
    else:
        # Fall back to default message
        full_caption = f"Descarga completada: {default_title}"

    # Telegram caption limit is 1024 characters
    MAX_CAPTION_LENGTH = 1024
    if len(full_caption) > MAX_CAPTION_LENGTH:
        # Truncate and add ellipsis
        full_caption = full_caption[:MAX_CAPTION_LENGTH - 3].rsplit(" ", 1)[0] + "..."

    return full_caption


def _split_file_if_needed(file_path: str, output_dir: str, correlation_id: str) -> list[str]:
    """Check file size and split if exceeds Telegram limit.

    With local Bot API enabled, files up to 2000MB are sent without splitting.

    Args:
        file_path: Path to the file to check
        output_dir: Directory for output segments
        correlation_id: Unique download ID for logging

    Returns:
        List of file paths (original if small, split parts if large)
    """
    from bot.split_processor import VideoSplitter

    file_size = os.path.getsize(file_path)
    logger.info(f"[{correlation_id}] File size: {file_size / (1024 * 1024):.1f} MB")

    max_file_size = config.telegram_max_upload_bytes
    if file_size <= max_file_size:
        logger.info(f"[{correlation_id}] File within Telegram limit, no splitting needed")
        return [file_path]

    if config.TELEGRAM_LOCAL_MODE:
        logger.info(
            f"[{correlation_id}] Local Bot API enabled — sending file without splitting "
            f"({file_size / (1024 * 1024):.1f} MB)"
        )
        return [file_path]

    # File exceeds cloud Telegram limit - need to split
    max_size_mb = max_file_size / (1024 * 1024)
    logger.info(f"[{correlation_id}] File exceeds {max_size_mb:.0f}MB limit, splitting required")

    # Calculate number of parts needed (add 1 to ensure each part is under limit)
    num_parts = int((file_size / max_file_size) + 1)
    # Limit to prevent too many parts
    num_parts = min(10, max(1, num_parts))

    logger.info(f"[{correlation_id}] Splitting into {num_parts} parts")
    splitter = VideoSplitter(file_path, output_dir)
    return splitter.split_by_parts(num_parts)


async def _send_downloaded_file_with_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    result: Any,
    format_type: str,
    correlation_id: str
) -> None:
    """Send downloaded file and show post-download menu.

    Supports both single files and multiple files (e.g., Instagram carousel).
    Works with both callback queries and regular messages.

    Args:
        update: Telegram update object
        context: Telegram context object
        result: Download result
        format_type: 'video' or 'audio'
        correlation_id: Unique download ID
    """
    from bot.downloaders.download_lifecycle import DownloadResult as LifecycleResult
    from telegram import InputMediaPhoto, InputMediaVideo

    # Get the appropriate message object (from callback or direct message)
    if update.callback_query:
        message = update.callback_query.message
    else:
        message = update.message

    # Extract file paths and metadata from result
    if isinstance(result, LifecycleResult):
        file_paths = result.file_paths if hasattr(result, 'file_paths') and result.file_paths else ([result.file_path] if result.file_path else [])
        metadata = result.metadata or {}
    elif isinstance(result, dict):
        file_paths = result.get('file_paths') or ([result.get('file_path')] if result.get('file_path') else [])
        metadata = result.get('metadata', {})
    else:
        file_paths = [str(result)] if result else []
        metadata = {}

    # Filter out None/empty paths and check existence
    file_paths = [fp for fp in file_paths if fp and os.path.exists(fp)]

    if not file_paths:
        await message.reply_text(
            "Error: No se encontró el archivo descargado."
        )
        return

    title = metadata.get('title', 'Descarga')
    is_carousel = metadata.get('is_carousel', False) or len(file_paths) > 1

    # Build caption from metadata (uses original caption for Instagram)
    main_caption = _build_caption_from_metadata(metadata, title)

    try:
        # Handle multiple files (e.g., Instagram carousel)
        if len(file_paths) > 1:
            logger.info(f"[{correlation_id}] Sending {len(file_paths)} files as media group")

            # Group files by type (images vs videos)
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
            video_extensions = {'.mp4', '.webm', '.mov', '.avi', '.mkv'}
            audio_extensions = {'.mp3', '.aac', '.wav', '.ogg', '.flac', '.m4a', '.opus'}

            # Check each file size and split if needed
            processed_file_paths = []
            for fp in file_paths:
                fp_dir = os.path.dirname(fp)
                split_dir = os.path.join(fp_dir, "split")
                parts = _split_file_if_needed(fp, split_dir, correlation_id)
                processed_file_paths.extend(parts)

            file_paths = processed_file_paths
            logger.info(f"[{correlation_id}] After splitting: {len(file_paths)} files")

            # Send first batch (up to 10) as media group
            first_batch = file_paths[:10]

            def _build_media_group() -> list:
                media_group = []
                for i, file_path in enumerate(first_batch):
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if i == 0:
                        caption = f"{main_caption}\n\n({i+1}/{len(file_paths)})"
                    else:
                        caption = None

                    try:
                        if file_ext in image_extensions:
                            media_group.append(InputMediaPhoto(
                                media=_media_input(file_path),
                                caption=caption if i == 0 else None
                            ))
                        elif file_ext in video_extensions:
                            media_group.append(InputMediaVideo(
                                media=_media_input(file_path),
                                caption=caption if i == 0 else None,
                                supports_streaming=True
                            ))
                        else:
                            media_group.append(InputMediaPhoto(
                                media=_media_input(file_path),
                                caption=caption if i == 0 else None
                            ))
                    except Exception as file_err:
                        logger.warning(f"[{correlation_id}] Failed to add file {file_path}: {file_err}")
                return media_group

            has_video_menu = False
            built_group = _build_media_group()
            if built_group:
                async def _send_group():
                    return await message.reply_media_group(media=_build_media_group())

                sent_messages = await _send_with_retry(
                    _send_group,
                    correlation_id=correlation_id,
                    label="reply_media_group",
                )
                logger.info(f"[{correlation_id}] Sent media group with {len(sent_messages)} items")

                first_video = next((m for m in built_group if isinstance(m, InputMediaVideo)), None)
                if first_video:
                    has_video_menu = True

            # Send remaining files individually (if more than 10)
            remaining = file_paths[10:]
            if remaining:
                logger.info(f"[{correlation_id}] Sending {len(remaining)} remaining files individually")
                for i, file_path in enumerate(remaining):
                    file_ext = os.path.splitext(file_path)[1].lower()
                    item_caption = f"{main_caption}\n\n({len(first_batch) + i + 1}/{len(file_paths)})"

                    try:
                        if file_ext in image_extensions:
                            async def _send_photo(fp=file_path, cap=item_caption):
                                with _open_file_for_send(fp) as photo_file:
                                    return await message.reply_photo(photo=photo_file, caption=cap)

                            await _send_with_retry(
                                _send_photo,
                                correlation_id=correlation_id,
                                label="reply_photo",
                            )
                        elif file_ext in video_extensions:
                            async def _send_video(fp=file_path, cap=item_caption):
                                with _open_file_for_send(fp) as video_file:
                                    return await message.reply_video(
                                        video=video_file,
                                        caption=cap,
                                        supports_streaming=True,
                                    )

                            await _send_with_retry(
                                _send_video,
                                correlation_id=correlation_id,
                                label="reply_video",
                            )
                            has_video_menu = True
                        elif file_ext in audio_extensions:
                            async def _send_audio(fp=file_path, cap=item_caption):
                                with _open_file_for_send(fp) as audio_file:
                                    return await message.reply_audio(
                                        audio=audio_file,
                                        caption=cap,
                                        title=title,
                                        performer=metadata.get('artist') or metadata.get('uploader'),
                                    )

                            await _send_with_retry(
                                _send_audio,
                                correlation_id=correlation_id,
                                label="reply_audio",
                            )
                        else:
                            async def _send_document(fp=file_path, cap=item_caption):
                                with _open_file_for_send(fp) as doc_file:
                                    return await message.reply_document(document=doc_file, caption=cap)

                            await _send_with_retry(
                                _send_document,
                                correlation_id=correlation_id,
                                label="reply_document",
                            )
                    except Exception as file_err:
                        logger.warning(f"[{correlation_id}] Failed to send remaining file {file_path}: {file_err}")

            # Show post-download menu for videos if any video was sent
            if has_video_menu:
                context.user_data["video_menu_correlation_id"] = correlation_id
                reply_markup = _get_video_menu_keyboard()
                await message.reply_text(
                    "¿Qué quieres hacer con estos videos?",
                    reply_markup=reply_markup
                )

        else:
            # Single file - check size and split if needed
            file_path = file_paths[0]

            # Get output directory for split files
            file_dir = os.path.dirname(file_path)
            split_dir = os.path.join(file_dir, "split")

            # Check if file needs splitting
            file_parts = _split_file_if_needed(file_path, split_dir, correlation_id)
            file_ext = os.path.splitext(file_path)[1].lower()
            audio_extensions = {'.mp3', '.aac', '.wav', '.ogg', '.flac', '.m4a', '.opus'}
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

            # Send each part
            num_parts = len(file_parts)
            for i, part_path in enumerate(file_parts):
                # Build caption with part number
                if num_parts > 1:
                    part_caption = f"{main_caption}\n\n(Parte {i+1}/{num_parts})"
                else:
                    part_caption = main_caption

                if format_type == 'audio' or file_ext in audio_extensions:
                    async def _send_audio(pp=part_path, cap=part_caption):
                        with _open_file_for_send(pp) as audio_file:
                            return await message.reply_audio(
                                audio=audio_file,
                                caption=cap,
                                title=title,
                                performer=metadata.get('artist') or metadata.get('uploader'),
                            )

                    await _send_with_retry(
                        _send_audio,
                        correlation_id=correlation_id,
                        label="reply_audio",
                    )
                elif file_ext in image_extensions:
                    async def _send_photo(pp=part_path, cap=part_caption):
                        with _open_file_for_send(pp) as photo_file:
                            return await message.reply_photo(photo=photo_file, caption=cap)

                    await _send_with_retry(
                        _send_photo,
                        correlation_id=correlation_id,
                        label="reply_photo",
                    )
                else:
                    async def _send_video(pp=part_path, cap=part_caption):
                        with _open_file_for_send(pp) as video_file:
                            return await message.reply_video(
                                video=video_file,
                                caption=cap,
                                supports_streaming=True,
                            )

                    sent_message = await _send_with_retry(
                        _send_video,
                        correlation_id=correlation_id,
                        label="reply_video",
                    )

                    # Show post-download menu only for last video part
                    if i == num_parts - 1:
                        context.user_data["video_menu_file_id"] = sent_message.video.file_id
                        context.user_data["video_menu_correlation_id"] = correlation_id
                        reply_markup = _get_video_menu_keyboard()
                        await message.reply_text(
                            "¿Qué quieres hacer con este video?",
                            reply_markup=reply_markup
                        )

            if num_parts > 1:
                await message.reply_text(
                    f"Video dividido en {num_parts} partes y enviado."
                )

        logger.info(f"[{correlation_id}] Downloaded file(s) sent to user {update.effective_user.id}")

        # Clean up temp directory after sending
        if isinstance(result, LifecycleResult) and result.temp_dir:
            import shutil
            try:
                shutil.rmtree(result.temp_dir, ignore_errors=True)
                logger.debug(f"[{correlation_id}] Cleaned up temp directory: {result.temp_dir}")
            except Exception as cleanup_err:
                logger.warning(f"[{correlation_id}] Failed to cleanup temp dir: {cleanup_err}")

    except Exception as e:
        logger.error(f"[{correlation_id}] Failed to send downloaded file(s): {e}")
        # Don't send error message here - let the caller handle it
        # This prevents duplicate error messages
        raise


async def _start_combined_download(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    correlation_id: str,
    url: str,
    format_type: str,
    post_action: str
) -> None:
    """Start combined download and process flow.

    Downloads the file and immediately processes it based on post_action.

    Args:
        update: Telegram update object
        context: Telegram context object
        correlation_id: Unique download ID
        url: URL to download
        format_type: 'video' or 'audio'
        post_action: 'videonote', 'extract', or 'voicenote'
    """
    query = update.callback_query
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Detect platform for display
    platform = _detect_platform_for_display(url)

    # Create facade
    facade = DownloadFacade()

    try:
        await facade.start()

        # Store facade instance for cancellation support
        context.user_data[f"download_facade_{correlation_id}"] = facade
        context.user_data[f"download_url_{correlation_id}"] = url
        context.user_data[f"download_format_{correlation_id}"] = format_type
        context.user_data[f"download_post_action_{correlation_id}"] = post_action
        context.user_data[f"download_status_{correlation_id}"] = "downloading"

        # Initial message with cancel button
        reply_markup = _get_download_cancel_keyboard(correlation_id)

        # Map action to display name
        action_names = {
            "videonote": "Nota de Video",
            "extract": "Extraer Audio",
            "voicenote": "Nota de Voz"
        }
        action_name = action_names.get(post_action, post_action)

        await query.edit_message_text(
            f"Descargando de {platform} para convertir a {action_name}...",
            reply_markup=reply_markup
        )

        # Progress tracking with enhanced state management
        last_message_text = [f"Descargando de {platform}..."]
        last_update_time = [0.0]

        async def progress_callback(progress: dict) -> None:
            """Update download progress message."""
            import time
            from bot.downloaders.progress_tracker import format_progress_message

            status = progress.get('status', 'downloading')
            percent = progress.get('percent', 0)

            # Rate limiting: only update every 1 second minimum
            current_time = time.time()
            if current_time - last_update_time[0] < 1.0 and status == 'downloading':
                return

            # Format message based on status
            if status == 'downloading':
                message = format_progress_message(progress)
                message = f"Descargando de {platform}...\n{message}\nLuego: convertir a {action_name}"

                if message != last_message_text[0]:
                    try:
                        await query.edit_message_text(
                            message,
                            reply_markup=reply_markup
                        )
                        last_message_text[0] = message
                        last_update_time[0] = current_time
                    except Exception as e:
                        logger.debug(f"Failed to update progress message: {e}")

            elif status == 'waiting':
                wait_msg = progress.get('message', 'Aplicando delay...')
                try:
                    await query.edit_message_text(wait_msg, reply_markup=reply_markup)
                except Exception:
                    pass

            elif status == 'completed':
                try:
                    await query.edit_message_text(f"Descarga completada. Convirtiendo a {action_name}...")
                    context.user_data[f"download_status_{correlation_id}"] = "completed"
                except Exception:
                    pass

            elif status == 'error':
                error_msg = progress.get('error', 'Error desconocido')
                try:
                    await query.edit_message_text(f"Error en la descarga: {error_msg}")
                    context.user_data[f"download_status_{correlation_id}"] = "error"
                except Exception:
                    pass

        # Create progress tracker with callback
        from bot.downloaders.progress_tracker import ProgressTracker
        tracker = ProgressTracker(
            min_update_interval=3.0,
            min_percent_change=5.0,
            on_update=lambda p: asyncio.create_task(progress_callback(p))
        )

        # Download with progress callback integration
        # IMPORTANT: cleanup_on_success=False so file remains for sending
        config_overrides = {
            'extract_audio': (format_type == 'audio'),
            'cleanup_on_success': False,
            'max_filesize_mb': _get_download_max_filesize_mb(),
        }

        # Apply Instagram inter-download delay before starting (notify user during wait)
        # (covers /download command + IG button flows for consistent behavior)
        if is_instagram_url(url):
            waited = await _apply_instagram_delay()
            if waited > 0:
                try:
                    await query.edit_message_text(
                        f"Aplicando delay de {waited:.1f}s para evitar detección de Instagram...",
                        reply_markup=reply_markup
                    )
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Failed to show Instagram delay notification: {e}")
                await asyncio.sleep(waited)
                # Restore normal message for seamless transition
                try:
                    await query.edit_message_text(
                        f"Descargando de {platform} para convertir a {action_name}...",
                        reply_markup=reply_markup
                    )
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Failed to restore post-delay message: {e}")

        # Guard: respect cancel that arrived during the (potentially long) delay sleep (Issue 1).
        # Uses the exact download_status user_data pattern from handle_download_cancel_callback.
        if context.user_data.get(f"download_status_{correlation_id}") == "cancelled":
            return

        # Note on Issue 5: any hypothetical exception in this narrow window after _apply
        # (but before facade reaches InstagramDownloader.download's finally) would skip
        # the timestamp _mark. This is best-effort; normal errors are handled inside
        # the downloader (which always marks). The window is tiny and pre-existing
        # for waited==0 IG cases.
        result = await facade.download(
            url=url,
            chat_id=chat_id,
            config_overrides=config_overrides
        )

        if result.success:
            context.user_data[f"download_status_{correlation_id}"] = "completed"

            # Immediately process based on post_action
            await query.edit_message_text(f"Descarga completada. Convirtiendo a {action_name}...")

            try:
                if post_action == "videonote":
                    await _process_to_videonote(update, context, result, correlation_id)
                elif post_action == "extract":
                    await _process_extract_audio(update, context, result, correlation_id)
                elif post_action == "voicenote":
                    await _process_to_voicenote(update, context, result, correlation_id)
                else:
                    logger.warning(f"Unknown post_action: {post_action}")
                    await _send_downloaded_file_with_menu(update, context, result, format_type, correlation_id)
            except Exception as e:
                logger.error(f"[{correlation_id}] Post-download processing failed: {e}")
                await query.edit_message_text(
                    f"Descarga completada pero el procesamiento falló: {e}\n"
                    f"El archivo descargado se enviará sin procesar."
                )
                # Send original file as fallback
                await _send_downloaded_file_with_menu(update, context, result, format_type, correlation_id)
        else:
            context.user_data[f"download_status_{correlation_id}"] = "error"
            await query.edit_message_text(
                f"Error en la descarga: {getattr(result, 'error_message', 'Error desconocido')}"
            )

    except FileTooLargeError as e:
        logger.warning(f"[{correlation_id}] File too large: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        await query.edit_message_text(e.to_user_message())
    except URLValidationError as e:
        logger.warning(f"[{correlation_id}] URL validation error: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        await query.edit_message_text(e.to_user_message())
    except UnsupportedURLError as e:
        logger.warning(f"[{correlation_id}] Unsupported URL: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        await query.edit_message_text(e.to_user_message())
    except DownloadError as e:
        logger.error(f"[{correlation_id}] Download error: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        error_msg = _get_error_message_for_exception(e, url, correlation_id)
        await query.edit_message_text(error_msg)
    except Exception as e:
        logger.error(f"[{correlation_id}] Unexpected error: {type(e).__name__}: {e}")
        context.user_data[f"download_status_{correlation_id}"] = "error"
        error_msg = _get_error_message_for_exception(e, url, correlation_id)
        await query.edit_message_text(error_msg)
    finally:
        # Clean up facade reference but keep status for /downloads command
        context.user_data.pop(f"download_facade_{correlation_id}", None)
        context.user_data.pop(f"download_post_action_{correlation_id}", None)
        try:
            await facade.stop()
        except Exception:
            pass


async def _process_to_videonote(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    result: Any,
    correlation_id: str
) -> None:
    """Process downloaded video to video note.

    Args:
        update: Telegram update object
        context: Telegram context object
        result: Download result with file_path
        correlation_id: Unique download ID
    """
    from bot.downloaders.download_lifecycle import DownloadResult as LifecycleResult

    # Handle both single and multiple files
    if isinstance(result, LifecycleResult):
        file_paths = result.file_paths if hasattr(result, 'file_paths') and result.file_paths else ([result.file_path] if result.file_path else [])
        metadata = result.metadata or {}
    elif isinstance(result, dict):
        file_paths = result.get('file_paths') or ([result.get('file_path')] if result.get('file_path') else [])
        metadata = result.get('metadata', {})
    else:
        file_paths = [str(result)] if result else []
        metadata = {}

    # Filter valid paths and find first video
    file_paths = [fp for fp in file_paths if fp and os.path.exists(fp)]
    video_extensions = {'.mp4', '.webm', '.mov', '.avi', '.mkv'}
    file_path = next((fp for fp in file_paths if os.path.splitext(fp)[1].lower() in video_extensions), file_paths[0] if file_paths else None)

    if not file_path:
        await update.callback_query.message.reply_text(
            "Error: No se encontró un video para procesar."
        )
        return

    temp_mgr = TempManager()
    output_filename = f"videonote_{correlation_id}.mp4"
    output_path = temp_mgr.get_temp_path(output_filename)

    try:
        # Process video to video note format
        success = await asyncio.get_event_loop().run_in_executor(
            None,
            VideoProcessor.process_video,
            str(file_path),
            str(output_path)
        )

        if success and os.path.exists(output_path):
            with open(output_path, 'rb') as video_file:
                await update.callback_query.message.reply_video_note(video_note=video_file)
            logger.info(f"[{correlation_id}] Video note sent successfully")
        else:
            raise FFmpegError("El procesamiento de video falló")

    except Exception as e:
        logger.error(f"[{correlation_id}] Failed to convert to video note: {e}")
        await update.callback_query.message.reply_text(
            f"Error al convertir a nota de video: {e}"
        )
        # Send original as fallback
        await _send_downloaded_file_with_menu(update, context, result, "video", correlation_id)
    finally:
        temp_mgr.cleanup()


async def _process_extract_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    result: Any,
    correlation_id: str
) -> None:
    """Extract audio from downloaded video.

    Args:
        update: Telegram update object
        context: Telegram context object
        result: Download result with file_path
        correlation_id: Unique download ID
    """
    from bot.downloaders.download_lifecycle import DownloadResult as LifecycleResult

    # Handle both single and multiple files
    if isinstance(result, LifecycleResult):
        file_paths = result.file_paths if hasattr(result, 'file_paths') and result.file_paths else ([result.file_path] if result.file_path else [])
        metadata = result.metadata or {}
    elif isinstance(result, dict):
        file_paths = result.get('file_paths') or ([result.get('file_path')] if result.get('file_path') else [])
        metadata = result.get('metadata', {})
    else:
        file_paths = [str(result)] if result else []
        metadata = {}

    # Filter valid paths and find first video
    file_paths = [fp for fp in file_paths if fp and os.path.exists(fp)]
    video_extensions = {'.mp4', '.webm', '.mov', '.avi', '.mkv'}
    file_path = next((fp for fp in file_paths if os.path.splitext(fp)[1].lower() in video_extensions), file_paths[0] if file_paths else None)

    if not file_path:
        await update.callback_query.message.reply_text(
            "Error: No se encontró un video para extraer audio."
        )
        return

    temp_mgr = TempManager()
    output_filename = f"audio_{correlation_id}.mp3"
    output_path = temp_mgr.get_temp_path(output_filename)

    try:
        # Extract audio using AudioExtractor
        extractor = AudioExtractor(str(file_path), str(output_path))
        success = await asyncio.get_event_loop().run_in_executor(
            None,
            extractor.extract
        )

        if success and os.path.exists(output_path):
            title = metadata.get('title', 'Video')
            with open(output_path, 'rb') as audio_file:
                await update.callback_query.message.reply_audio(
                    audio=audio_file,
                    caption=f"Audio extraído: {title}",
                    title=title,
                    performer=metadata.get('artist') or metadata.get('uploader')
                )
            logger.info(f"[{correlation_id}] Audio extracted and sent successfully")
        else:
            raise AudioExtractionError("La extracción de audio falló")

    except Exception as e:
        logger.error(f"[{correlation_id}] Failed to extract audio: {e}")
        await update.callback_query.message.reply_text(
            f"Error al extraer audio: {e}"
        )
        # Send original as fallback
        await _send_downloaded_file_with_menu(update, context, result, "video", correlation_id)
    finally:
        temp_mgr.cleanup()


async def _process_to_voicenote(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    result: Any,
    correlation_id: str
) -> None:
    """Process downloaded audio to voice note.

    Args:
        update: Telegram update object
        context: Telegram context object
        result: Download result with file_path
        correlation_id: Unique download ID
    """
    from bot.downloaders.download_lifecycle import DownloadResult as LifecycleResult

    # Handle both single and multiple files
    if isinstance(result, LifecycleResult):
        file_paths = result.file_paths if hasattr(result, 'file_paths') and result.file_paths else ([result.file_path] if result.file_path else [])
        metadata = result.metadata or {}
    elif isinstance(result, dict):
        file_paths = result.get('file_paths') or ([result.get('file_path')] if result.get('file_path') else [])
        metadata = result.get('metadata', {})
    else:
        file_paths = [str(result)] if result else []
        metadata = {}

    # Filter valid paths and find first audio/video file
    file_paths = [fp for fp in file_paths if fp and os.path.exists(fp)]
    audio_video_extensions = {'.mp4', '.webm', '.mov', '.avi', '.mkv', '.mp3', '.m4a', '.wav', '.ogg'}
    file_path = next((fp for fp in file_paths if os.path.splitext(fp)[1].lower() in audio_video_extensions), file_paths[0] if file_paths else None)

    if not file_path:
        await update.callback_query.message.reply_text(
            "Error: No se encontró un archivo de audio/video para procesar."
        )
        return

    temp_mgr = TempManager()
    output_filename = f"voicenote_{correlation_id}.ogg"
    output_path = temp_mgr.get_temp_path(output_filename)

    try:
        # Convert to voice note format (OGG Opus)
        converter = VoiceNoteConverter(str(file_path), str(output_path))
        success = await asyncio.get_event_loop().run_in_executor(
            None,
            converter.convert
        )

        if success and os.path.exists(output_path):
            with open(output_path, 'rb') as voice_file:
                await update.callback_query.message.reply_voice(voice=voice_file)
            logger.info(f"[{correlation_id}] Voice note sent successfully")
        else:
            raise VoiceConversionError("La conversión a nota de voz falló")

    except Exception as e:
        logger.error(f"[{correlation_id}] Failed to convert to voice note: {e}")
        await update.callback_query.message.reply_text(
            f"Error al convertir a nota de voz: {e}"
        )
        # Send original as fallback
        await _send_downloaded_file_with_menu(update, context, result, "audio", correlation_id)
    finally:
        temp_mgr.cleanup()


async def handle_download_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle download cancellation callback.

    Handles race conditions gracefully:
    - If download completes before cancel is processed, show "already completed"
    - If cancel fails due to already-finished state, show appropriate message
    - Always clean up user_data to prevent stale references

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Parse callback data: download:cancel:correlation_id
    if not callback_data.startswith("download:cancel:"):
        logger.warning(f"Invalid cancel callback data: {callback_data}")
        await query.edit_message_text("Error: callback inválido")
        return

    parts = callback_data.split(":")
    if len(parts) != 3:
        logger.warning(f"Invalid callback data format: {callback_data}")
        await query.edit_message_text("Error: formato de callback inválido")
        return

    correlation_id = parts[2]

    logger.info(f"[{correlation_id}] Download cancel requested by user {user_id}")

    # Get current status to check race conditions
    current_status = context.user_data.get(f"download_status_{correlation_id}", "unknown")

    # Get facade instance
    facade = context.user_data.get(f"download_facade_{correlation_id}")

    cancelled = False
    if facade:
        try:
            # Cancel the download
            cancelled = await facade.cancel_download(correlation_id)
            if cancelled:
                logger.info(f"[{correlation_id}] Download cancelled successfully")
                await query.edit_message_text("Descarga cancelada")
                context.user_data[f"download_status_{correlation_id}"] = "cancelled"
            else:
                # Check if already completed (race condition)
                if current_status == "completed":
                    logger.info(f"[{correlation_id}] Cancel failed - download already completed")
                    await query.edit_message_text("La descarga ya se había completado")
                else:
                    logger.info(f"[{correlation_id}] Cancel failed - download not found or already finished")
                    await query.edit_message_text("No se pudo cancelar (¿ya completada?)")
                    # Mark as cancelled so any in-flight pre-download delay sleep in _start_*
                    # can detect it via the existing user_data status guard (Issue 1).
                    context.user_data[f"download_status_{correlation_id}"] = "cancelled"
        except Exception as e:
            logger.error(f"[{correlation_id}] Error during cancel: {e}")
            await query.edit_message_text("Error al cancelar la descarga")
    else:
        # No facade found - download may have already finished
        if current_status == "completed":
            logger.info(f"[{correlation_id}] No facade found - download already completed")
            await query.edit_message_text("La descarga ya se había completado")
        elif current_status == "error":
            logger.info(f"[{correlation_id}] No facade found - download already failed")
            await query.edit_message_text("La descarga ya había fallado")
        else:
            logger.info(f"[{correlation_id}] No facade found - marking as cancelled")
            await query.edit_message_text("Descarga cancelada")
            context.user_data[f"download_status_{correlation_id}"] = "cancelled"

    # Clean up user_data
    context.user_data.pop(f"download_url_{correlation_id}", None)
    context.user_data.pop(f"download_format_{correlation_id}", None)
    context.user_data.pop(f"download_meta_{correlation_id}", None)
    context.user_data.pop(f"download_facade_{correlation_id}", None)
    # Keep download_status for /downloads command history


async def handle_downloads_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /downloads command to show active and recent downloads.

    Displays a list of active downloads with progress and recent downloads
    with their completion status. Provides cancel buttons for active downloads.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id

    # Collect active downloads from user_data
    active_downloads = []
    recent_downloads = []

    # Scan user_data for download entries
    for key in list(context.user_data.keys()):
        if key.startswith("download_status_"):
            correlation_id = key.replace("download_status_", "")
            status = context.user_data.get(key, "unknown")
            url = context.user_data.get(f"download_url_{correlation_id}", "")
            format_type = context.user_data.get(f"download_format_{correlation_id}", "video")

            # Get platform for display
            platform = _detect_platform_for_display(url) or "Desconocido"

            download_info = {
                "correlation_id": correlation_id,
                "status": status,
                "platform": platform,
                "format": format_type,
                "url": url[:50] + "..." if len(url) > 50 else url
            }

            if status == "downloading":
                active_downloads.append(download_info)
            elif status in ["completed", "error", "cancelled"]:
                recent_downloads.append(download_info)

    # Sort recent downloads by correlation_id (which includes timestamp info)
    recent_downloads = sorted(recent_downloads, key=lambda x: x["correlation_id"], reverse=True)[:5]

    # Build message
    lines = ["Descargas activas:"]

    if active_downloads:
        for d in active_downloads:
            lines.append(f"  {d['correlation_id']}: {d['platform']} ({d['format']})")
    else:
        lines.append("  Ninguna")

    lines.append("\nDescargas recientes:")

    if recent_downloads:
        for d in recent_downloads:
            status_icon = "✅" if d['status'] == "completed" else "❌" if d['status'] == "error" else "🚫"
            lines.append(f"  {status_icon} {d['correlation_id']}: {d['platform']}")
    else:
        lines.append("  Ninguna")

    # Add cancel buttons for active downloads
    keyboard = []
    for d in active_downloads:
        keyboard.append([
            InlineKeyboardButton(
                f"Cancelar {d['correlation_id']}",
                callback_data=f"download:cancel:{d['correlation_id']}"
            )
        ])

    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None

    await update.message.reply_text(
        "\n".join(lines),
        reply_markup=reply_markup
    )


async def send_downloaded_file(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    result: Any
) -> None:
    """Send downloaded file(s) to user (legacy helper).

    Supports both single files and multiple files (e.g., Instagram carousel).

    Args:
        update: Telegram update object
        context: Telegram context object
        result: Download result with file_path and metadata
    """
    from bot.downloaders.download_lifecycle import DownloadResult as LifecycleResult
    from telegram import InputMediaPhoto, InputMediaVideo

    # Extract file paths and metadata
    if isinstance(result, LifecycleResult):
        file_paths = result.file_paths if hasattr(result, 'file_paths') and result.file_paths else ([result.file_path] if result.file_path else [])
        metadata = result.metadata or {}
    elif isinstance(result, dict):
        file_paths = result.get('file_paths') or ([result.get('file_path')] if result.get('file_path') else [])
        metadata = result.get('metadata', {})
    else:
        file_paths = [str(result)] if result else []
        metadata = {}

    # Filter valid paths
    file_paths = [fp for fp in file_paths if fp and os.path.exists(fp)]

    if not file_paths:
        await update.message.reply_text(
            "Error: No se encontró el archivo descargado."
        )
        return

    title = metadata.get('title', 'Descarga')

    # Build caption from metadata (uses original caption for Instagram)
    main_caption = _build_caption_from_metadata(metadata, title)

    try:
        # Handle multiple files
        if len(file_paths) > 1:
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
            video_extensions = {'.mp4', '.webm', '.mov', '.avi', '.mkv'}

            batch = file_paths[:10]

            def _build_legacy_media_group():
                group = []
                for i, file_path in enumerate(batch):
                    file_ext = os.path.splitext(file_path)[1].lower()
                    caption = f"{main_caption}\n\n({i+1}/{len(file_paths)})" if i == 0 else None
                    try:
                        if file_ext in image_extensions:
                            group.append(InputMediaPhoto(media=open(file_path, 'rb'), caption=caption))
                        elif file_ext in video_extensions:
                            group.append(InputMediaVideo(
                                media=open(file_path, 'rb'),
                                caption=caption,
                                supports_streaming=True,
                            ))
                    except Exception as file_err:
                        logger.warning(f"Failed to add file {file_path}: {file_err}")
                return group

            built_group = _build_legacy_media_group()
            if built_group:
                async def _send_group():
                    return await update.message.reply_media_group(media=_build_legacy_media_group())

                await _send_with_retry(_send_group, label="reply_media_group")
                logger.info(f"Downloaded media group sent to user {update.effective_user.id}")
            else:
                await update.message.reply_text("Error: No se pudieron procesar los archivos.")
        else:
            # Single file
            file_path = file_paths[0]
            file_ext = os.path.splitext(file_path)[1].lower()
            audio_extensions = {'.mp3', '.aac', '.wav', '.ogg', '.flac', '.m4a', '.opus'}

            if file_ext in audio_extensions:
                async def _send_audio():
                    with open(file_path, 'rb') as audio_file:
                        return await update.message.reply_audio(
                            audio=audio_file,
                            caption=main_caption,
                            title=title,
                            performer=metadata.get('artist') or metadata.get('uploader'),
                        )

                await _send_with_retry(_send_audio, label="reply_audio")
            else:
                async def _send_video():
                    with open(file_path, 'rb') as video_file:
                        return await update.message.reply_video(
                            video=video_file,
                            caption=main_caption,
                            supports_streaming=True,
                        )

                await _send_with_retry(_send_video, label="reply_video")
            logger.info(f"Downloaded file sent to user {update.effective_user.id}")
    except Exception as e:
        logger.error(f"Failed to send downloaded file(s): {e}")
        await update.message.reply_text(
            "Error al enviar el archivo descargado."
        )


async def handle_url_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle messages containing URLs for download (legacy direct download).

    This handler is kept for backward compatibility.
    New behavior uses handle_url_detection with inline menu.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    # Delegate to the new URL detection handler with menu
    await handle_url_detection(update, context)


# =============================================================================
# Post-Download Integration Handlers
# =============================================================================


def _get_postdownload_video_keyboard(correlation_id: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard for post-download video menu options."""
    keyboard = [
        [
            InlineKeyboardButton("Convertir a Nota de Video", callback_data=f"postdownload:videonote:{correlation_id}"),
            InlineKeyboardButton("Extraer Audio", callback_data=f"postdownload:extract_audio:{correlation_id}"),
        ],
        [
            InlineKeyboardButton("Convertir Formato", callback_data=f"postdownload:convert_video:{correlation_id}"),
            InlineKeyboardButton("Unir Videos", callback_data=f"postdownload:join_video:{correlation_id}"),
        ],
        [
            InlineKeyboardButton("Descargas Recientes", callback_data=f"postdownload:recent:{correlation_id}"),
            InlineKeyboardButton("Nada", callback_data=f"postdownload:nothing:{correlation_id}"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_postdownload_audio_keyboard(correlation_id: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard for post-download audio menu options."""
    keyboard = [
        [
            InlineKeyboardButton("Convertir a Nota de Voz", callback_data=f"postdownload:voicenote:{correlation_id}"),
            InlineKeyboardButton("Convertir Formato", callback_data=f"postdownload:convert_audio:{correlation_id}"),
        ],
        [
            InlineKeyboardButton("Bass Boost", callback_data=f"postdownload:bass:{correlation_id}"),
            InlineKeyboardButton("Reducir Ruido", callback_data=f"postdownload:denoise:{correlation_id}"),
            InlineKeyboardButton("Más Opciones...", callback_data=f"postdownload:more:{correlation_id}"),
        ],
        [
            InlineKeyboardButton("Descargas Recientes", callback_data=f"postdownload:recent:{correlation_id}"),
            InlineKeyboardButton("Nada", callback_data=f"postdownload:nothing:{correlation_id}"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_postdownload_audio_more_keyboard(correlation_id: str) -> InlineKeyboardMarkup:
    """Generate extended inline keyboard for post-download audio options."""
    keyboard = [
        [
            InlineKeyboardButton("Treble Boost", callback_data=f"postdownload:treble:{correlation_id}"),
            InlineKeyboardButton("Ecualizar", callback_data=f"postdownload:equalize:{correlation_id}"),
        ],
        [
            InlineKeyboardButton("Comprimir", callback_data=f"postdownload:compress:{correlation_id}"),
            InlineKeyboardButton("Normalizar", callback_data=f"postdownload:normalize:{correlation_id}"),
        ],
        [
            InlineKeyboardButton("Efecto 3D", callback_data=f"postdownload:stereo_3d:{correlation_id}"),
            InlineKeyboardButton("Cambiar Pitch", callback_data=f"postdownload:pitch_shift:{correlation_id}"),
        ],
        [
            InlineKeyboardButton("Volver", callback_data=f"postdownload:back_audio:{correlation_id}"),
            InlineKeyboardButton("Nada", callback_data=f"postdownload:nothing:{correlation_id}"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_postdownload_pitch_shift_keyboard(correlation_id: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard for pitch shift intensity selection."""
    keyboard = [
        [
            InlineKeyboardButton("Grave", callback_data=f"postdownload:pitch_shift_intensity:{correlation_id}:grave"),
            InlineKeyboardButton("Agudo", callback_data=f"postdownload:pitch_shift_intensity:{correlation_id}:agudo"),
            InlineKeyboardButton("Muy Agudo", callback_data=f"postdownload:pitch_shift_intensity:{correlation_id}:muy_agudo"),
        ],
        [
            InlineKeyboardButton("Volver", callback_data=f"postdownload:back_audio:{correlation_id}"),
            InlineKeyboardButton("Nada", callback_data=f"postdownload:nothing:{correlation_id}"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_postdownload_stereo_3d_keyboard(correlation_id: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard for stereo 3D intensity selection."""
    keyboard = [
        [
            InlineKeyboardButton("Suave", callback_data=f"postdownload:stereo_3d_intensity:{correlation_id}:suave"),
            InlineKeyboardButton("Medio", callback_data=f"postdownload:stereo_3d_intensity:{correlation_id}:medio"),
            InlineKeyboardButton("Intenso", callback_data=f"postdownload:stereo_3d_intensity:{correlation_id}:intenso"),
        ],
        [
            InlineKeyboardButton("Volver", callback_data=f"postdownload:back_audio:{correlation_id}"),
            InlineKeyboardButton("Nada", callback_data=f"postdownload:nothing:{correlation_id}"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_postdownload_intensity_keyboard(correlation_id: str, effect_type: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard for intensity selection (bass/treble)."""
    keyboard = []
    row = []
    for i in range(1, 11):
        row.append(InlineKeyboardButton(str(i), callback_data=f"postdownload:{effect_type}_intensity:{correlation_id}:{i}"))
        if len(row) == 5:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([
        InlineKeyboardButton("Volver", callback_data=f"postdownload:back_audio:{correlation_id}"),
        InlineKeyboardButton("Nada", callback_data=f"postdownload:nothing:{correlation_id}"),
    ])
    return InlineKeyboardMarkup(keyboard)


def _get_postdownload_effect_strength_keyboard(correlation_id: str, effect_type: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard for effect strength selection (denoise/compress)."""
    strengths = [("Leve", "light"), ("Medio", "medium"), ("Fuerte", "strong")]
    keyboard = [
        [InlineKeyboardButton(label, callback_data=f"postdownload:{effect_type}_strength:{correlation_id}:{value}")
         for label, value in strengths]
    ]
    keyboard.append([
        InlineKeyboardButton("Volver", callback_data=f"postdownload:back_audio:{correlation_id}"),
        InlineKeyboardButton("Nada", callback_data=f"postdownload:nothing:{correlation_id}"),
    ])
    return InlineKeyboardMarkup(keyboard)


def _get_postdownload_audio_format_keyboard(correlation_id: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard for audio format conversion."""
    keyboard = [
        [
            InlineKeyboardButton("MP3", callback_data=f"postdownload:audio_format:{correlation_id}:mp3"),
            InlineKeyboardButton("AAC", callback_data=f"postdownload:audio_format:{correlation_id}:aac"),
        ],
        [
            InlineKeyboardButton("WAV", callback_data=f"postdownload:audio_format:{correlation_id}:wav"),
            InlineKeyboardButton("OGG", callback_data=f"postdownload:audio_format:{correlation_id}:ogg"),
        ],
        [
            InlineKeyboardButton("Volver", callback_data=f"postdownload:back_audio:{correlation_id}"),
            InlineKeyboardButton("Nada", callback_data=f"postdownload:nothing:{correlation_id}"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_postdownload_video_format_keyboard(correlation_id: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard for video format conversion."""
    keyboard = [
        [
            InlineKeyboardButton("MP4", callback_data=f"postdownload:video_format:{correlation_id}:mp4"),
            InlineKeyboardButton("AVI", callback_data=f"postdownload:video_format:{correlation_id}:avi"),
            InlineKeyboardButton("MOV", callback_data=f"postdownload:video_format:{correlation_id}:mov"),
        ],
        [
            InlineKeyboardButton("MKV", callback_data=f"postdownload:video_format:{correlation_id}:mkv"),
            InlineKeyboardButton("WEBM", callback_data=f"postdownload:video_format:{correlation_id}:webm"),
        ],
        [
            InlineKeyboardButton("Volver", callback_data=f"postdownload:back_video:{correlation_id}"),
            InlineKeyboardButton("Nada", callback_data=f"postdownload:nothing:{correlation_id}"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_postdownload_video_audio_format_keyboard(correlation_id: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard for audio extraction format selection."""
    keyboard = [
        [
            InlineKeyboardButton("MP3", callback_data=f"postdownload:extract_format:{correlation_id}:mp3"),
            InlineKeyboardButton("AAC", callback_data=f"postdownload:extract_format:{correlation_id}:aac"),
        ],
        [
            InlineKeyboardButton("WAV", callback_data=f"postdownload:extract_format:{correlation_id}:wav"),
            InlineKeyboardButton("OGG", callback_data=f"postdownload:extract_format:{correlation_id}:ogg"),
        ],
        [
            InlineKeyboardButton("Volver", callback_data=f"postdownload:back_video:{correlation_id}"),
            InlineKeyboardButton("Nada", callback_data=f"postdownload:nothing:{correlation_id}"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_recent_downloads_keyboard(session, page: int = 0) -> InlineKeyboardMarkup:
    """Generate inline keyboard for recent downloads list."""
    entries = session.get_recent(5)
    keyboard = []
    for i, entry in enumerate(entries, 1):
        title = entry.get_title()[:20] + "..." if len(entry.get_title()) > 20 else entry.get_title()
        platform = entry.get_platform()
        time_ago = entry.time_ago()
        label = f"{i}. {title} ({platform}) - {time_ago}"
        keyboard.append([
            InlineKeyboardButton(label, callback_data=f"reprocess:{entry.correlation_id}")
        ])
    if entries:
        keyboard.append([
            InlineKeyboardButton("Limpiar Lista", callback_data="postdownload:clear_recent:none"),
        ])
    keyboard.append([
        InlineKeyboardButton("Cerrar", callback_data="cancel"),
    ])
    return InlineKeyboardMarkup(keyboard)


def _get_join_video_keyboard(video_count: int) -> InlineKeyboardMarkup:
    """Generate inline keyboard for video join session."""
    keyboard = [
        [
            InlineKeyboardButton("✅ Unir Videos", callback_data="join_video_action:done"),
            InlineKeyboardButton("❌ Cancelar", callback_data="join_video_action:cancel"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_join_audio_keyboard(audio_count: int) -> InlineKeyboardMarkup:
    """Generate inline keyboard for audio join session."""
    keyboard = [
        [
            InlineKeyboardButton("✅ Unir Audios", callback_data="join_audio_action:done"),
            InlineKeyboardButton("❌ Cancelar", callback_data="join_audio_action:cancel"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


async def handle_postdownload_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle post-download video processing callbacks."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    if not callback_data.startswith("postdownload:"):
        logger.warning(f"Unexpected callback data: {callback_data}")
        return

    parts = callback_data.split(":")
    if len(parts) < 3:
        logger.warning(f"Invalid callback data format: {callback_data}")
        return

    action = parts[1]
    correlation_id = parts[2]

    logger.info(f"[{correlation_id}] Post-download action '{action}' selected by user {user_id}")

    from bot.downloaders import get_user_download_session
    session = get_user_download_session(context)
    entry = session.get(correlation_id)

    if not entry:
        await query.edit_message_text(
            "Error: No se encontró la información de la descarga. El archivo puede haber sido eliminado."
        )
        return

    if not os.path.exists(entry.file_path):
        await query.edit_message_text(
            "Error: El archivo ya no está disponible. Fue eliminado automáticamente."
        )
        return

    if action == "videonote":
        await _handle_postdownload_videonote(update, context, entry, correlation_id)
    elif action == "extract_audio":
        reply_markup = _get_postdownload_video_audio_format_keyboard(correlation_id)
        await query.edit_message_text("Selecciona el formato de audio:", reply_markup=reply_markup)
    elif action == "convert_video":
        reply_markup = _get_postdownload_video_format_keyboard(correlation_id)
        await query.edit_message_text("Selecciona el formato de video:", reply_markup=reply_markup)
    elif action == "join_video":
        await _handle_postdownload_join_video(update, context, entry, correlation_id)
    elif action == "recent":
        await handle_recent_downloads(update, context)
    elif action == "back_video":
        reply_markup = _get_postdownload_video_keyboard(correlation_id)
        await query.edit_message_text("¿Qué quieres hacer con este video?", reply_markup=reply_markup)
    elif action == "nothing":
        await _handle_postdownload_nothing(update, context, entry, correlation_id)


async def _handle_postdownload_videonote(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str
) -> None:
    """Convert downloaded video to video note."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    await query.edit_message_text("Convirtiendo a nota de video...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"videonote_{user_id}_{correlation_id}.mp4"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Processing downloaded video to video note for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, VideoProcessor.process_video, str(file_path), str(output_path)),
                    timeout=config.PROCESSING_TIMEOUT
                )
                if not success:
                    raise FFmpegError("El procesamiento de video falló")
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            logger.info(f"[{correlation_id}] Sending video note to user {user_id}")
            with open(output_path, "rb") as video_file:
                await query.message.reply_video_note(video_note=video_file)

            reply_markup = _get_postdownload_video_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con este video?", reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Video note sent successfully to user {user_id}")
        except (FFmpegError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Video note conversion failed: {e}")
            await query.edit_message_text(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error converting to video note: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def _handle_postdownload_join_video(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str
) -> None:
    """Start a video join session with the downloaded video as the first video."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    logger.info(f"[{correlation_id}] User {user_id} selected 'Join Videos' - starting join session")

    # Check if there's already an active join session
    if context.user_data.get("join_session"):
        await query.edit_message_text(
            "Ya tienes una sesión de unión de videos activa. "
            "Usa /done para unir o /cancel para cancelarla primero."
        )
        return

    # Initialize join session
    temp_mgr = TempManager()
    context.user_data["join_session"] = {
        "videos": [str(file_path)],
        "temp_mgr": temp_mgr,
        "last_activity": asyncio.get_event_loop().time(),
        "correlation_id": correlation_id,
    }

    # Track the file with temp manager
    temp_mgr.track_file(str(file_path))

    await query.edit_message_text(
        "🎬 *Modo unión de videos activado*\n\n"
        "El video descargado es el **primer video** en la lista.\n"
        "Envíame más videos para unir (máximo 10 en total).\n"
        "Los videos se unirán en el orden en que los envíes.\n\n"
        f"Actualmente tienes: *1 video*",
        parse_mode="Markdown",
        reply_markup=_get_join_video_keyboard(1)
    )
    logger.info(f"[{correlation_id}] Join session started for user {user_id} with downloaded video")


async def _handle_postdownload_nothing(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str
) -> None:
    """Clean up and finish - user selected 'Nothing' (Nada).

    Deletes the downloaded file and cleans up user_data state.

    Args:
        update: Telegram update object
        context: Telegram context object
        entry: Download entry with file information
        correlation_id: Download correlation ID
    """
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    logger.info(f"[{correlation_id}] User {user_id} selected 'Nothing' - cleaning up")

    # Delete the downloaded file
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"[{correlation_id}] Deleted file: {file_path}")
        except Exception as e:
            logger.warning(f"[{correlation_id}] Failed to delete file: {e}")

    # Clean up temp directory if it exists
    temp_dir = getattr(entry, 'temp_dir', None)
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"[{correlation_id}] Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"[{correlation_id}] Failed to cleanup temp dir: {e}")

    # Clean up session
    from bot.downloaders import get_user_download_session
    session = get_user_download_session(context)
    session.remove(correlation_id)

    # Clean up user_data
    for key in list(context.user_data.keys()):
        if correlation_id in key:
            context.user_data.pop(key, None)

    await query.edit_message_text(
        "✓ Archivo eliminado.\n\n"
        "Envíame otra URL si quieres descargar algo más."
    )
    logger.info(f"[{correlation_id}] Cleanup completed for user {user_id}")


async def handle_postdownload_audio_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle post-download audio processing callbacks."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    if not callback_data.startswith("postdownload:"):
        logger.warning(f"Unexpected callback data: {callback_data}")
        return

    parts = callback_data.split(":")
    if len(parts) < 3:
        logger.warning(f"Invalid callback data format: {callback_data}")
        return

    action = parts[1]
    correlation_id = parts[2]

    logger.info(f"[{correlation_id}] Post-download audio action '{action}' selected by user {user_id}")

    from bot.downloaders import get_user_download_session
    session = get_user_download_session(context)
    entry = session.get(correlation_id)

    if not entry:
        await query.edit_message_text(
            "Error: No se encontró la información de la descarga. El archivo puede haber sido eliminado."
        )
        return

    if not os.path.exists(entry.file_path):
        await query.edit_message_text("Error: El archivo ya no está disponible. Fue eliminado automáticamente.")
        return

    if action == "voicenote":
        await _handle_postdownload_voicenote(update, context, entry, correlation_id)
    elif action == "convert_audio":
        reply_markup = _get_postdownload_audio_format_keyboard(correlation_id)
        await query.edit_message_text("Selecciona el formato de audio:", reply_markup=reply_markup)
    elif action == "bass":
        reply_markup = _get_postdownload_intensity_keyboard(correlation_id, "bass")
        await query.edit_message_text("Selecciona la intensidad del Bass Boost:", reply_markup=reply_markup)
    elif action == "treble":
        reply_markup = _get_postdownload_intensity_keyboard(correlation_id, "treble")
        await query.edit_message_text("Selecciona la intensidad del Treble Boost:", reply_markup=reply_markup)
    elif action == "denoise":
        reply_markup = _get_postdownload_effect_strength_keyboard(correlation_id, "denoise")
        await query.edit_message_text("Selecciona la intensidad de la reducción de ruido:", reply_markup=reply_markup)
    elif action == "compress":
        reply_markup = _get_postdownload_effect_strength_keyboard(correlation_id, "compress")
        await query.edit_message_text("Selecciona la intensidad de la compresión:", reply_markup=reply_markup)
    elif action == "normalize":
        await _handle_postdownload_normalize(update, context, entry, correlation_id)
    elif action == "equalize":
        await _handle_postdownload_equalize(update, context, entry, correlation_id)
    elif action == "stereo_3d":
        reply_markup = _get_postdownload_stereo_3d_keyboard(correlation_id)
        await query.edit_message_text(
            "Selecciona la intensidad del efecto 3D:\n\n"
            "• Suave - ampliación estéreo ligera\n"
            "• Medio - efecto equilibrado\n"
            "• Intenso - ampliación estéreo marcada",
            reply_markup=reply_markup,
        )
    elif action == "pitch_shift":
        reply_markup = _get_postdownload_pitch_shift_keyboard(correlation_id)
        await query.edit_message_text(
            "Selecciona el cambio de tono:\n\n"
            "• Grave - tono más grave (-3.5 semitonos)\n"
            "• Agudo - tono más agudo (+3.5 semitonos)\n"
            "• Muy Agudo - tono muy agudo (+6.5 semitonos)",
            reply_markup=reply_markup,
        )
    elif action == "more":
        reply_markup = _get_postdownload_audio_more_keyboard(correlation_id)
        await query.edit_message_text("Más opciones de procesamiento de audio:", reply_markup=reply_markup)
    elif action == "back_audio":
        reply_markup = _get_postdownload_audio_keyboard(correlation_id)
        await query.edit_message_text("¿Qué quieres hacer con este audio?", reply_markup=reply_markup)
    elif action == "recent":
        await handle_recent_downloads(update, context)
    elif action == "clear_recent":
        session.clear()
        await query.edit_message_text("Lista de descargas recientes limpiada.")
    elif action == "nothing":
        await _handle_postdownload_nothing(update, context, entry, correlation_id)


async def _handle_postdownload_voicenote(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str
) -> None:
    """Convert downloaded audio to voice note."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    await query.edit_message_text("Convirtiendo a nota de voz...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"voicenote_{user_id}_{correlation_id}.ogg"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Converting audio to voice note for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                converter = VoiceNoteConverter(str(file_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, converter.process), timeout=config.PROCESSING_TIMEOUT
                )
                if not success:
                    raise VoiceConversionError("No pude convertir a nota de voz")
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            logger.info(f"[{correlation_id}] Sending voice note to user {user_id}")
            with open(output_path, "rb") as voice_file:
                await query.message.reply_voice(voice=voice_file)

            reply_markup = _get_postdownload_audio_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con este audio?", reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Voice note sent successfully to user {user_id}")
        except (VoiceConversionError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Voice note conversion failed: {e}")
            await query.edit_message_text(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error converting to voice note: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def _handle_postdownload_normalize(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str
) -> None:
    """Normalize downloaded audio."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    await query.edit_message_text("Normalizando audio...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"normalized_{user_id}_{correlation_id}.mp3"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Normalizing audio for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                effects = AudioEffects(str(file_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, effects.normalize), timeout=config.PROCESSING_TIMEOUT
                )
                if not success:
                    raise AudioEffectsError("No pude normalizar el audio")
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            logger.info(f"[{correlation_id}] Sending normalized audio to user {user_id}")
            with open(output_path, "rb") as audio_file:
                await query.message.reply_audio(
                    audio=audio_file, filename=f"normalized_{correlation_id}.mp3", title="Audio Normalizado"
                )

            reply_markup = _get_postdownload_audio_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con este audio?", reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Normalized audio sent successfully to user {user_id}")
        except (AudioEffectsError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Audio normalization failed: {e}")
            await query.edit_message_text(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error normalizing audio: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def _handle_postdownload_equalize(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str
) -> None:
    """Show equalizer for downloaded audio."""
    query = update.callback_query
    context.user_data["postdownload_eq"] = {"correlation_id": correlation_id, "bass": 0, "mid": 0, "treble": 0}
    reply_markup = _get_equalizer_keyboard(0, 0, 0)
    await query.edit_message_text("Ajusta el ecualizador (Bass/Mid/Treble):", reply_markup=reply_markup)


async def handle_recent_downloads(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show recent downloads list."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    from bot.downloaders import get_user_download_session
    session = get_user_download_session(context)
    entries = session.get_recent(5)

    if not entries:
        await query.edit_message_text(
            "No hay descargas recientes en esta sesión.\n\n"
            "Las descargas se mantienen solo durante la sesión actual "
            "y no se guardan permanentemente por privacidad.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Cerrar", callback_data="cancel")]])
        )
        return

    lines = ["Descargas recientes:"]
    for i, entry in enumerate(entries, 1):
        title = entry.get_title()
        platform = entry.get_platform()
        time_ago = entry.time_ago()
        status_icon = "✅" if entry.status == "completed" else "❌"
        lines.append(f"{status_icon} {i}. {title} ({platform}) - hace {time_ago}")

    reply_markup = _get_recent_downloads_keyboard(session)
    await query.edit_message_text(
        "\n".join(lines) + "\n\nSelecciona una para reprocesar:", reply_markup=reply_markup
    )
    logger.info(f"Displayed recent downloads for user {user_id}: {len(entries)} items")


async def handle_reprocess_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle reprocessing of a recent download."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    if not callback_data.startswith("reprocess:"):
        logger.warning(f"Unexpected callback data: {callback_data}")
        return

    parts = callback_data.split(":")
    if len(parts) != 2:
        logger.warning(f"Invalid callback data format: {callback_data}")
        return

    correlation_id = parts[1]
    logger.info(f"[{correlation_id}] Reprocess requested by user {user_id}")

    from bot.downloaders import get_user_download_session
    session = get_user_download_session(context)
    entry = session.get(correlation_id)

    if not entry:
        await query.edit_message_text(
            "Error: No se encontró la descarga. Puede haber sido eliminada de la sesión."
        )
        return

    if not os.path.exists(entry.file_path):
        await query.edit_message_text(
            "Error: El archivo ya no está disponible. Fue eliminado automáticamente.\n\n"
            "Los archivos temporales se eliminan después de un tiempo. Por favor descarga el contenido nuevamente."
        )
        return

    file_ext = os.path.splitext(entry.file_path)[1].lower()
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    audio_exts = {'.mp3', '.aac', '.wav', '.ogg', '.flac', '.m4a', '.opus'}

    if file_ext in video_exts:
        reply_markup = _get_postdownload_video_keyboard(correlation_id)
        await query.edit_message_text(
            f"Reprocesando: {entry.get_title()}\n\n¿Qué quieres hacer con este video?", reply_markup=reply_markup
        )
    elif file_ext in audio_exts:
        reply_markup = _get_postdownload_audio_keyboard(correlation_id)
        await query.edit_message_text(
            f"Reprocesando: {entry.get_title()}\n\n¿Quieres hacer algo más con este audio?", reply_markup=reply_markup
        )
    else:
        await query.edit_message_text(
            f"Tipo de archivo no reconocido: {file_ext}\n\nSolo se pueden reprocesar videos y archivos de audio."
        )
    logger.info(f"[{correlation_id}] Reprocess menu shown to user {user_id}")


# =============================================================================
# Post-Download Format and Effect Handlers
# =============================================================================

async def handle_postdownload_format_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle post-download format selection callbacks.

    Handles: audio_format, video_format, extract_format callbacks
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Parse: postdownload:ACTION:CORRELATION_ID:FORMAT
    parts = callback_data.split(":")
    if len(parts) != 4:
        logger.warning(f"Invalid callback data format: {callback_data}")
        return

    action = parts[1]
    correlation_id = parts[2]
    format_type = parts[3]

    logger.info(f"[{correlation_id}] Post-download format '{format_type}' selected for {action} by user {user_id}")

    from bot.downloaders import get_user_download_session
    session = get_user_download_session(context)
    entry = session.get(correlation_id)

    if not entry:
        await query.edit_message_text(
            "Error: No se encontró la información de la descarga. El archivo puede haber sido eliminado."
        )
        return

    if not os.path.exists(entry.file_path):
        await query.edit_message_text("Error: El archivo ya no está disponible. Fue eliminado automáticamente.")
        return

    if action == "audio_format":
        await _handle_postdownload_audio_format_conversion(update, context, entry, correlation_id, format_type)
    elif action == "video_format":
        await _handle_postdownload_video_format_conversion(update, context, entry, correlation_id, format_type)
    elif action == "extract_format":
        await _handle_postdownload_extract_audio(update, context, entry, correlation_id, format_type)


async def _handle_postdownload_audio_format_conversion(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str, target_format: str
) -> None:
    """Convert downloaded audio to specified format."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    await query.edit_message_text(f"Convirtiendo a formato {target_format.upper()}...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"converted_{user_id}_{correlation_id}.{target_format}"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Converting audio to {target_format} for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                converter = AudioFormatConverter(str(file_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, converter.convert), timeout=config.PROCESSING_TIMEOUT
                )
                if not success:
                    raise AudioFormatConversionError(f"No pude convertir a {target_format}")
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            logger.info(f"[{correlation_id}] Sending converted audio to user {user_id}")
            with open(output_path, "rb") as audio_file:
                await query.message.reply_audio(
                    audio=audio_file,
                    filename=f"converted_{correlation_id}.{target_format}",
                    title=f"Audio Convertido ({target_format.upper()})"
                )

            reply_markup = _get_postdownload_audio_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con este audio?", reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Converted audio sent successfully to user {user_id}")
        except (AudioFormatConversionError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Audio format conversion failed: {e}")
            await query.edit_message_text(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error converting audio format: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def _handle_postdownload_video_format_conversion(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str, target_format: str
) -> None:
    """Convert downloaded video to specified format."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    await query.edit_message_text(f"Convirtiendo video a formato {target_format.upper()}...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"converted_{user_id}_{correlation_id}.{target_format}"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Converting video to {target_format} for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                converter = FormatConverter(str(file_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, converter.convert), timeout=config.PROCESSING_TIMEOUT
                )
                if not success:
                    raise FormatConversionError(f"No pude convertir a {target_format}")
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            logger.info(f"[{correlation_id}] Sending converted video to user {user_id}")
            with open(output_path, "rb") as video_file:
                await query.message.reply_video(
                    video=video_file,
                    caption=f"Video convertido a {target_format.upper()}",
                    supports_streaming=True
                )

            reply_markup = _get_postdownload_video_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con este video?", reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Converted video sent successfully to user {user_id}")
        except (FormatConversionError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Video format conversion failed: {e}")
            await query.edit_message_text(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error converting video format: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def _handle_postdownload_extract_audio(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str, audio_format: str
) -> None:
    """Extract audio from downloaded video."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    await query.edit_message_text(f"Extrayendo audio en formato {audio_format.upper()}...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"audio_{user_id}_{correlation_id}.{audio_format}"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Extracting audio as {audio_format} for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                extractor = AudioExtractor(str(file_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, extractor.extract), timeout=config.PROCESSING_TIMEOUT
                )
                if not success:
                    raise AudioExtractionError(f"No pude extraer el audio")
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            logger.info(f"[{correlation_id}] Sending extracted audio to user {user_id}")
            with open(output_path, "rb") as audio_file:
                await query.message.reply_audio(
                    audio=audio_file,
                    filename=f"extracted_{correlation_id}.{audio_format}",
                    title=f"Audio Extraído ({audio_format.upper()})"
                )

            reply_markup = _get_postdownload_video_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con este video?", reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Extracted audio sent successfully to user {user_id}")
        except (AudioExtractionError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Audio extraction failed: {e}")
            await query.edit_message_text(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error extracting audio: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def handle_postdownload_stereo_3d_intensity_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle post-download stereo 3D intensity selection callbacks."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Parse: postdownload:stereo_3d_intensity:CORRELATION_ID:INTENSITY
    parts = callback_data.split(":")
    if len(parts) != 4:
        logger.warning(f"Invalid callback data format: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    correlation_id = parts[2]
    intensity = parts[3].lower()
    intensity_labels = {
        "suave": "Suave",
        "medio": "Medio",
        "intenso": "Intenso",
    }

    if intensity not in intensity_labels:
        await query.edit_message_text("Error: intensidad inválida.")
        return

    logger.info(
        f"[{correlation_id}] Post-download stereo 3D intensity '{intensity}' "
        f"selected by user {user_id}"
    )

    from bot.downloaders import get_user_download_session
    session = get_user_download_session(context)
    entry = session.get(correlation_id)

    if not entry:
        await query.edit_message_text(
            "Error: No se encontró la información de la descarga. El archivo puede haber sido eliminado."
        )
        return

    if not os.path.exists(entry.file_path):
        await query.edit_message_text("Error: El archivo ya no está disponible. Fue eliminado automáticamente.")
        return

    await _handle_postdownload_stereo_3d(
        update, context, entry, correlation_id, intensity, intensity_labels[intensity]
    )


async def handle_postdownload_intensity_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle post-download intensity selection callbacks (bass/treble boost).

    Handles: bass_intensity, treble_intensity callbacks
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Parse: postdownload:ACTION:CORRELATION_ID:INTENSITY
    parts = callback_data.split(":")
    if len(parts) != 4:
        logger.warning(f"Invalid callback data format: {callback_data}")
        return

    action = parts[1]
    correlation_id = parts[2]
    intensity = int(parts[3])

    logger.info(f"[{correlation_id}] Post-download {action} intensity {intensity} selected by user {user_id}")

    from bot.downloaders import get_user_download_session
    session = get_user_download_session(context)
    entry = session.get(correlation_id)

    if not entry:
        await query.edit_message_text(
            "Error: No se encontró la información de la descarga. El archivo puede haber sido eliminado."
        )
        return

    if not os.path.exists(entry.file_path):
        await query.edit_message_text("Error: El archivo ya no está disponible. Fue eliminado automáticamente.")
        return

    if action == "bass_intensity":
        await _handle_postdownload_bass_boost(update, context, entry, correlation_id, intensity)
    elif action == "treble_intensity":
        await _handle_postdownload_treble_boost(update, context, entry, correlation_id, intensity)


async def _handle_postdownload_bass_boost(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str, intensity: int
) -> None:
    """Apply bass boost to downloaded audio."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    await query.edit_message_text(f"Aplicando Bass Boost (intensidad {intensity})...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"bass_boosted_{user_id}_{correlation_id}.mp3"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Applying bass boost (intensity {intensity}) for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                enhancer = AudioEnhancer(str(file_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: enhancer.bass_boost(intensity)),
                    timeout=config.PROCESSING_TIMEOUT
                )
                if not success:
                    raise AudioEnhancementError("No pude aplicar el bass boost")
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            logger.info(f"[{correlation_id}] Sending bass boosted audio to user {user_id}")
            with open(output_path, "rb") as audio_file:
                await query.message.reply_audio(
                    audio=audio_file,
                    filename=f"bass_boosted_{correlation_id}.mp3",
                    title=f"Bass Boost (Intensidad {intensity})"
                )

            reply_markup = _get_postdownload_audio_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con este audio?", reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Bass boosted audio sent successfully to user {user_id}")
        except (AudioEnhancementError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Bass boost failed: {e}")
            await query.edit_message_text(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error applying bass boost: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def _handle_postdownload_stereo_3d(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    entry: Any,
    correlation_id: str,
    intensity: str,
    intensity_label: str,
) -> None:
    """Apply stereo 3D effect to downloaded audio."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    await query.edit_message_text(f"Aplicando efecto 3D ({intensity_label})...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"stereo3d_{user_id}_{correlation_id}.mp3"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(
                f"[{correlation_id}] Applying stereo 3D ({intensity}) for user {user_id}"
            )
            try:
                loop = asyncio.get_event_loop()
                effects = AudioEffects(str(file_path), str(output_path))
                await asyncio.wait_for(
                    loop.run_in_executor(None, effects.stereo_3d, intensity),
                    timeout=config.PROCESSING_TIMEOUT,
                )
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            doc_filename = f"stereo_3d_{intensity}_{correlation_id}.mp3"
            document_sent = False
            logger.info(f"[{correlation_id}] Sending stereo 3D audio to user {user_id}")
            with open(output_path, "rb") as audio_file:
                await query.message.reply_audio(
                    audio=audio_file,
                    filename=doc_filename,
                    title=f"Audio con efecto 3D ({intensity_label})",
                )
                try:
                    audio_file.seek(0)
                    await query.message.reply_document(
                        document=audio_file,
                        filename=doc_filename,
                        caption=(
                            f"Archivo MP3 con efecto 3D ({intensity_label}) "
                            "para editores de video"
                        ),
                    )
                    document_sent = True
                except Exception as doc_error:
                    logger.warning(
                        f"[{correlation_id}] Audio sent but document delivery failed: {doc_error}"
                    )

            reply_markup = _get_postdownload_audio_keyboard(correlation_id)
            ready_msg = "¡Listo! ¿Quieres hacer algo más con este audio?"
            if not document_sent:
                ready_msg += (
                    "\n\n(No pude enviar el archivo MP3 como documento; "
                    "usa el audio de arriba.)"
                )
            await query.message.reply_text(ready_msg, reply_markup=reply_markup)
            logger.info(f"[{correlation_id}] Stereo 3D audio sent successfully to user {user_id}")
        except (AudioEffectsError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Stereo 3D processing failed: {e}")
            await query.edit_message_text(f"Error: {get_user_error_message(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error applying stereo 3D: {e}")
            await query.edit_message_text(DEFAULT_ERROR_MESSAGE)


async def handle_postdownload_pitch_shift_intensity_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle post-download pitch shift intensity selection callbacks."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Parse: postdownload:pitch_shift_intensity:CORRELATION_ID:INTENSITY
    parts = callback_data.split(":")
    if len(parts) != 4:
        logger.warning(f"Invalid callback data format: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    correlation_id = parts[2]
    intensity = parts[3].lower()
    intensity_labels = {
        "grave": "Grave",
        "agudo": "Agudo",
        "muy_agudo": "Muy agudo",
    }

    if intensity not in intensity_labels:
        await query.edit_message_text("Error: intensidad inválida.")
        return

    logger.info(
        f"[{correlation_id}] Post-download pitch shift intensity '{intensity}' "
        f"selected by user {user_id}"
    )

    from bot.downloaders import get_user_download_session
    session = get_user_download_session(context)
    entry = session.get(correlation_id)

    if not entry:
        await query.edit_message_text(
            "Error: No se encontró la información de la descarga. El archivo puede haber sido eliminado."
        )
        return

    if not os.path.exists(entry.file_path):
        await query.edit_message_text("Error: El archivo ya no está disponible. Fue eliminado automáticamente.")
        return

    await _handle_postdownload_pitch_shift(
        update, context, entry, correlation_id, intensity, intensity_labels[intensity]
    )


async def _handle_postdownload_pitch_shift(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    entry: Any,
    correlation_id: str,
    intensity: str,
    intensity_label: str,
) -> None:
    """Apply pitch shift effect to downloaded audio."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    await query.edit_message_text(f"Aplicando cambio de tono ({intensity_label})...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"pitch_{user_id}_{correlation_id}.mp3"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(
                f"[{correlation_id}] Applying pitch shift ({intensity}) for user {user_id}"
            )
            try:
                loop = asyncio.get_event_loop()
                effects = AudioEffects(str(file_path), str(output_path))
                await asyncio.wait_for(
                    loop.run_in_executor(None, effects.pitch_shift, intensity),
                    timeout=config.PROCESSING_TIMEOUT,
                )
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            doc_filename = f"pitch_shift_{intensity}_{correlation_id}.mp3"
            document_sent = False
            logger.info(f"[{correlation_id}] Sending pitch shift audio to user {user_id}")
            with open(output_path, "rb") as audio_file:
                await query.message.reply_audio(
                    audio=audio_file,
                    filename=doc_filename,
                    title=f"Audio con cambio de tono ({intensity_label})",
                )
                try:
                    audio_file.seek(0)
                    await query.message.reply_document(
                        document=audio_file,
                        filename=doc_filename,
                        caption=(
                            f"Archivo MP3 con cambio de tono ({intensity_label}) "
                            "para editores de video"
                        ),
                    )
                    document_sent = True
                except Exception as doc_error:
                    logger.warning(
                        f"[{correlation_id}] Audio sent but document delivery failed: {doc_error}"
                    )

            reply_markup = _get_postdownload_audio_keyboard(correlation_id)
            ready_msg = "¡Listo! ¿Quieres hacer algo más con este audio?"
            if not document_sent:
                ready_msg += (
                    "\n\n(No pude enviar el archivo MP3 como documento; "
                    "usa el audio de arriba.)"
                )
            await query.message.reply_text(ready_msg, reply_markup=reply_markup)
            logger.info(f"[{correlation_id}] Pitch shift audio sent successfully to user {user_id}")

        except (AudioEffectsError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Pitch shift processing failed: {e}")
            await query.edit_message_text(f"Error: {get_user_error_message(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error applying pitch shift: {e}")
            await query.edit_message_text(DEFAULT_ERROR_MESSAGE)


async def _handle_postdownload_treble_boost(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str, intensity: int
) -> None:
    """Apply treble boost to downloaded audio."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    await query.edit_message_text(f"Aplicando Treble Boost (intensidad {intensity})...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"treble_boosted_{user_id}_{correlation_id}.mp3"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Applying treble boost (intensity {intensity}) for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                enhancer = AudioEnhancer(str(file_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: enhancer.treble_boost(intensity)),
                    timeout=config.PROCESSING_TIMEOUT
                )
                if not success:
                    raise AudioEnhancementError("No pude aplicar el treble boost")
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            logger.info(f"[{correlation_id}] Sending treble boosted audio to user {user_id}")
            with open(output_path, "rb") as audio_file:
                await query.message.reply_audio(
                    audio=audio_file,
                    filename=f"treble_boosted_{correlation_id}.mp3",
                    title=f"Treble Boost (Intensidad {intensity})"
                )

            reply_markup = _get_postdownload_audio_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con este audio?", reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Treble boosted audio sent successfully to user {user_id}")
        except (AudioEnhancementError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Treble boost failed: {e}")
            await query.edit_message_text(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error applying treble boost: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def handle_postdownload_effect_strength_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle post-download effect strength callbacks (denoise/compress).

    Handles: denoise_strength, compress_strength callbacks
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Parse: postdownload:ACTION:CORRELATION_ID:STRENGTH
    parts = callback_data.split(":")
    if len(parts) != 4:
        logger.warning(f"Invalid callback data format: {callback_data}")
        return

    action = parts[1]
    correlation_id = parts[2]
    strength = parts[3]

    logger.info(f"[{correlation_id}] Post-download {action} strength {strength} selected by user {user_id}")

    from bot.downloaders import get_user_download_session
    session = get_user_download_session(context)
    entry = session.get(correlation_id)

    if not entry:
        await query.edit_message_text(
            "Error: No se encontró la información de la descarga. El archivo puede haber sido eliminado."
        )
        return

    if not os.path.exists(entry.file_path):
        await query.edit_message_text("Error: El archivo ya no está disponible. Fue eliminado automáticamente.")
        return

    if action == "denoise_strength":
        await _handle_postdownload_denoise(update, context, entry, correlation_id, strength)
    elif action == "compress_strength":
        await _handle_postdownload_compress(update, context, entry, correlation_id, strength)


async def _handle_postdownload_denoise(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str, strength: str
) -> None:
    """Apply denoise effect to downloaded audio."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    strength_map = {"light": "leve", "medium": "media", "strong": "fuerte"}
    strength_es = strength_map.get(strength, strength)

    await query.edit_message_text(f"Reduciendo ruido (intensidad {strength_es})...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"denoised_{user_id}_{correlation_id}.mp3"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Applying denoise (strength {strength}) for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                effects = AudioEffects(str(file_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: effects.denoise(strength)),
                    timeout=config.PROCESSING_TIMEOUT
                )
                if not success:
                    raise AudioEffectsError("No pude reducir el ruido")
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            logger.info(f"[{correlation_id}] Sending denoised audio to user {user_id}")
            with open(output_path, "rb") as audio_file:
                await query.message.reply_audio(
                    audio=audio_file,
                    filename=f"denoised_{correlation_id}.mp3",
                    title=f"Audio Sin Ruido ({strength_es.capitalize()})"
                )

            reply_markup = _get_postdownload_audio_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con este audio?", reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Denoised audio sent successfully to user {user_id}")
        except (AudioEffectsError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Denoise failed: {e}")
            await query.edit_message_text(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error denoising audio: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def _handle_postdownload_compress(
    update: Update, context: ContextTypes.DEFAULT_TYPE, entry: Any, correlation_id: str, strength: str
) -> None:
    """Apply compression effect to downloaded audio."""
    query = update.callback_query
    user_id = update.effective_user.id
    file_path = entry.file_path

    strength_map = {"light": "leve", "medium": "media", "strong": "fuerte"}
    strength_es = strength_map.get(strength, strength)

    await query.edit_message_text(f"Comprimiendo audio (intensidad {strength_es})...")

    with TempManager() as temp_mgr:
        try:
            output_filename = f"compressed_{user_id}_{correlation_id}.mp3"
            output_path = temp_mgr.get_temp_path(output_filename)

            logger.info(f"[{correlation_id}] Applying compression (strength {strength}) for user {user_id}")
            try:
                loop = asyncio.get_event_loop()
                effects = AudioEffects(str(file_path), str(output_path))
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: effects.compress(strength)),
                    timeout=config.PROCESSING_TIMEOUT
                )
                if not success:
                    raise AudioEffectsError("No pude comprimir el audio")
            except asyncio.TimeoutError as e:
                raise ProcessingTimeoutError("El procesamiento tardó demasiado") from e

            logger.info(f"[{correlation_id}] Sending compressed audio to user {user_id}")
            with open(output_path, "rb") as audio_file:
                await query.message.reply_audio(
                    audio=audio_file,
                    filename=f"compressed_{correlation_id}.mp3",
                    title=f"Audio Comprimido ({strength_es.capitalize()})"
                )

            reply_markup = _get_postdownload_audio_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con este audio?", reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Compressed audio sent successfully to user {user_id}")
        except (AudioEffectsError, ProcessingTimeoutError) as e:
            logger.error(f"[{correlation_id}] Compression failed: {e}")
            await query.edit_message_text(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error compressing audio: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


# ──────────────────────────────────────────────
# Image Processing Handlers
# ──────────────────────────────────────────────

IMAGE_BATCH_DEBOUNCE_SECONDS = 1.5


def _user_data_key(user_id: int, chat) -> int | tuple[int, int]:
    """Return the PTB user_data key for a user/chat pair."""
    if chat.type == ChatType.PRIVATE:
        return user_id
    return (user_id, chat.id)


def _store_image_menu_context(
    application,
    user_id: int,
    chat,
    file_ids: list[str],
    correlation_id: str,
    truncated: bool = False,
) -> None:
    """Store image menu state for single or batch processing."""
    user_data = application.user_data[_user_data_key(user_id, chat)]
    user_data["image_menu_file_ids"] = file_ids
    user_data["image_menu_file_id"] = file_ids[0]
    user_data["image_menu_correlation_id"] = correlation_id
    user_data["image_menu_truncated"] = truncated


async def _send_image_menu_message(
    application,
    chat,
    user_id: int,
    file_ids: list[str],
    correlation_id: str,
    reply_to_message_id: int | None = None,
    truncated: bool = False,
) -> None:
    """Send the image action menu for one or more images."""
    _store_image_menu_context(
        application, user_id, chat, file_ids, correlation_id, truncated=truncated
    )

    count = len(file_ids)
    if count > 1:
        text = (
            f"{count} imágenes recibidas.\n\n"
            "«Mejorar», «Naturalizar» y «Agrupar» procesan todas las imágenes del álbum. "
            "Selecciona una acción:"
        )
    else:
        text = "Imagen recibida. Selecciona una acción:"

    if truncated:
        text += (
            f"\n\n⚠️ Solo se procesarán las primeras "
            f"{config.MAX_IMAGE_BATCH_SIZE} imágenes."
        )

    reply_markup = _get_image_menu_keyboard(count)
    await application.bot.send_message(
        chat_id=chat.id,
        text=text,
        reply_markup=reply_markup,
        reply_to_message_id=reply_to_message_id,
    )
    logger.info(
        f"[{correlation_id}] Image menu displayed to user {user_id} "
        f"({count} image(s))"
    )


async def _schedule_image_batch_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
) -> None:
    """Accumulate album images and show a single menu after debounce."""
    message = update.message
    media_group_id = message.media_group_id
    user_id = update.effective_user.id
    chat = message.chat

    if not media_group_id:
        correlation_id = str(uuid.uuid4())[:8]
        await _send_image_menu_message(
            context.application,
            chat,
            user_id,
            [file_id],
            correlation_id,
            reply_to_message_id=message.message_id,
        )
        return

    sessions = context.application.bot_data.setdefault("image_batch_sessions", {})
    session_key = f"{chat.id}:{media_group_id}"

    session = sessions.get(session_key)
    if session is None:
        session = {
            "file_ids": [],
            "user_id": user_id,
            "chat": chat,
            "correlation_id": str(uuid.uuid4())[:8],
            "last_message_id": message.message_id,
            "debounce_task": None,
            "truncated": False,
        }
        sessions[session_key] = session

    debounce_task = session.get("debounce_task")
    if debounce_task and not debounce_task.done():
        debounce_task.cancel()

    if len(session["file_ids"]) < config.MAX_IMAGE_BATCH_SIZE:
        session["file_ids"].append(file_id)
    else:
        session["truncated"] = True
    session["last_message_id"] = message.message_id
    session["user_id"] = user_id

    application = context.application

    async def _debounced_show_menu() -> None:
        current_task = asyncio.current_task()
        cancelled = False
        try:
            await asyncio.sleep(IMAGE_BATCH_DEBOUNCE_SECONDS)
            file_ids = list(session["file_ids"])
            await _send_image_menu_message(
                application,
                session["chat"],
                session["user_id"],
                file_ids,
                session["correlation_id"],
                reply_to_message_id=session["last_message_id"],
                truncated=session.get("truncated", False),
            )
        except asyncio.CancelledError:
            cancelled = True
            return
        except Exception as e:
            logger.error(
                f"[{session['correlation_id']}] Failed to send image batch menu: {e}"
            )
        finally:
            if not cancelled and session.get("debounce_task") is current_task:
                sessions.pop(session_key, None)

    task = asyncio.create_task(_debounced_show_menu())
    session["debounce_task"] = task


def _get_image_noise_keyboard() -> InlineKeyboardMarkup:
    """Inline keyboard for subtle noise intensity selection."""
    keyboard = [
        [
            InlineKeyboardButton("1 · Muy sutil", callback_data="image_noise:1"),
            InlineKeyboardButton("2 · Sutil", callback_data="image_noise:2"),
            InlineKeyboardButton("3 · Normal", callback_data="image_noise:3"),
        ],
        [
            InlineKeyboardButton("4 · Notable", callback_data="image_noise:4"),
            InlineKeyboardButton("5 · Marcado", callback_data="image_noise:5"),
        ],
        [
            InlineKeyboardButton("← Volver", callback_data="back:image"),
            InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_image_menu_keyboard(image_count: int = 1) -> InlineKeyboardMarkup:
    """Generate inline keyboard for image processing menu."""
    if image_count > 1:
        keyboard = [
            [
                InlineKeyboardButton("Mejorar", callback_data="image_action:enhance"),
                InlineKeyboardButton("Naturalizar", callback_data="image_action:noise"),
            ],
            [
                InlineKeyboardButton("Agrupar", callback_data="image_action:group"),
            ],
        ]
    else:
        keyboard = [
            [
                InlineKeyboardButton("Comprimir", callback_data="image_action:compress"),
                InlineKeyboardButton("Convertir Formato", callback_data="image_action:convert"),
            ],
            [
                InlineKeyboardButton("Redimensionar", callback_data="image_action:resize"),
                InlineKeyboardButton("Info de Imagen", callback_data="image_action:info"),
            ],
            [
                InlineKeyboardButton("Mejorar", callback_data="image_action:enhance"),
                InlineKeyboardButton("Naturalizar", callback_data="image_action:noise"),
            ],
            [
                InlineKeyboardButton("Agrupar", callback_data="image_action:group"),
            ],
        ]
    return InlineKeyboardMarkup(keyboard)


IMAGE_GROUP_DEBOUNCE_SECONDS = 1.5
TELEGRAM_MAX_CAPTION_LENGTH = 1024


def _truncate_telegram_caption(text: str) -> str:
    """Truncate text to Telegram's media caption limit."""
    if len(text) <= TELEGRAM_MAX_CAPTION_LENGTH:
        return text
    return text[: TELEGRAM_MAX_CAPTION_LENGTH - 3].rsplit(" ", 1)[0] + "..."


def _get_image_group_keyboard(image_count: int) -> InlineKeyboardMarkup:
    """Generate inline keyboard for image group collection session."""
    keyboard = [
        [
            InlineKeyboardButton("✅ Listo", callback_data="image_group_action:done"),
            InlineKeyboardButton("❌ Cancelar", callback_data="image_group_action:cancel"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_image_group_session(context: ContextTypes.DEFAULT_TYPE) -> dict | None:
    """Return the active image group session if it exists and is not expired."""
    session = context.user_data.get("image_group_session")
    if not session:
        return None

    current_time = asyncio.get_event_loop().time()
    if current_time - session["last_activity"] > config.JOIN_SESSION_TIMEOUT:
        context.user_data.pop("image_group_session", None)
        return None

    return session


def _start_image_group_session(
    context: ContextTypes.DEFAULT_TYPE,
    file_ids: list[str],
    correlation_id: str,
) -> dict:
    """Initialize an image group collection session."""
    session = {
        "file_ids": list(file_ids),
        "caption": None,
        "correlation_id": correlation_id,
        "last_activity": asyncio.get_event_loop().time(),
        "debounce_task": None,
        "pending_album_ids": [],
    }
    context.user_data["image_group_session"] = session
    return session


async def _send_album_from_file_ids(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_ids: list[str],
    correlation_id: str,
    caption: str | None = None,
) -> None:
    """Send images as one or more Telegram albums using stored file IDs."""
    from telegram import InputMediaPhoto

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    album_size = min(config.MAX_IMAGE_BATCH_SIZE, 10)
    total = len(file_ids)

    logger.info(
        f"[{correlation_id}] Sending {total} grouped images in albums to user {user_id}"
        + (" with caption" if caption else "")
    )

    effective_message = update.callback_query.message if update.callback_query else update.message

    for i in range(0, total, album_size):
        batch = file_ids[i:i + album_size]
        media_group = []
        for j, file_id in enumerate(batch):
            item_caption = caption if i == 0 and j == 0 and caption else None
            media_group.append(InputMediaPhoto(media=file_id, caption=item_caption))

        if effective_message:
            await effective_message.reply_media_group(media=media_group)
        else:
            await context.bot.send_media_group(chat_id=chat_id, media=media_group)


def _format_image_group_inventory(
    image_count: int,
    has_caption: bool = False,
) -> str:
    """Describe current image group session contents."""
    parts = [f"*{image_count}* imagen(es)"]
    if has_caption:
        parts.append("*caption*")
    inventory = " y ".join(parts)
    return f"Actualmente tienes: {inventory} (máximo {config.MAX_IMAGE_BATCH_SIZE})."


def _format_image_group_footer(image_count: int) -> str:
    """Return guidance text for incomplete image group sessions."""
    if image_count < 2:
        return "\n\nNecesitas al menos 2 imágenes para crear un álbum."
    return ""


async def _notify_image_group_progress(
    update: Update,
    added_count: int,
    total_count: int,
    truncated: bool = False,
    has_caption: bool = False,
) -> None:
    """Send a status message during image group collection."""
    text = (
        f"✓ {added_count} imagen(es) agregada(s).\n\n"
        f"{_format_image_group_inventory(total_count, has_caption=has_caption)}"
    )
    if truncated:
        text += (
            f"\n\n⚠️ Se alcanzó el máximo de {config.MAX_IMAGE_BATCH_SIZE} imágenes. "
            "Las adicionales fueron ignoradas."
        )
    text += _format_image_group_footer(total_count)

    await update.message.reply_text(
        text,
        parse_mode="Markdown",
        reply_markup=_get_image_group_keyboard(total_count),
    )


async def _add_images_to_group_session(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    new_file_ids: list[str],
) -> None:
    """Append images to the active group session and notify the user."""
    session = _get_image_group_session(context)
    if not session:
        return

    added_count = 0
    truncated = False
    for file_id in new_file_ids:
        if len(session["file_ids"]) >= config.MAX_IMAGE_BATCH_SIZE:
            truncated = True
            break
        session["file_ids"].append(file_id)
        added_count += 1

    session["last_activity"] = asyncio.get_event_loop().time()

    if added_count == 0 and truncated:
        await update.message.reply_text(
            f"⚠️ Ya tienes el máximo de {config.MAX_IMAGE_BATCH_SIZE} imágenes.\n"
            "Presiona *Listo* para recibir el álbum o *Cancelar* para salir.",
            parse_mode="Markdown",
            reply_markup=_get_image_group_keyboard(len(session["file_ids"])),
        )
        return

    await _notify_image_group_progress(
        update,
        added_count,
        len(session["file_ids"]),
        truncated=truncated,
        has_caption=bool(session.get("caption")),
    )


async def _schedule_image_group_batch(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
) -> None:
    """Accumulate album images into an active group session after debounce."""
    session = _get_image_group_session(context)
    if not session:
        return

    session["pending_album_ids"].append(file_id)
    session["last_activity"] = asyncio.get_event_loop().time()

    debounce_task = session.get("debounce_task")
    if debounce_task and not debounce_task.done():
        debounce_task.cancel()

    async def _debounced_add() -> None:
        current_task = asyncio.current_task()
        cancelled = False
        try:
            await asyncio.sleep(IMAGE_GROUP_DEBOUNCE_SECONDS)
            pending_ids = list(session["pending_album_ids"])
            session["pending_album_ids"] = []
            await _add_images_to_group_session(update, context, pending_ids)
        except asyncio.CancelledError:
            cancelled = True
            return
        finally:
            if not cancelled and session.get("debounce_task") is current_task:
                session["debounce_task"] = None

    task = asyncio.create_task(_debounced_add())
    session["debounce_task"] = task


async def _try_collect_image_for_group_session(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
) -> bool:
    """Collect an image for an active group session. Returns True if handled."""
    session = _get_image_group_session(context)
    if not session:
        return False

    message = update.message
    if message.media_group_id:
        await _schedule_image_group_batch(update, context, file_id)
        return True

    await _add_images_to_group_session(update, context, [file_id])
    return True


async def _try_collect_caption_for_group_session(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    """Collect album caption text for an active group session. Returns True if handled."""
    session = _get_image_group_session(context)
    if not session:
        return False

    text = (update.message.text or "").strip()
    if not text:
        return False

    session["caption"] = _truncate_telegram_caption(text)
    session["last_activity"] = asyncio.get_event_loop().time()

    image_count = len(session["file_ids"])
    status_text = (
        "✓ Caption guardado para el álbum.\n\n"
        f"{_format_image_group_inventory(image_count, has_caption=True)}"
    )
    status_text += _format_image_group_footer(image_count)

    await update.message.reply_text(
        status_text,
        parse_mode="Markdown",
        reply_markup=_get_image_group_keyboard(image_count),
    )
    return True


async def handle_image_group_s_caption_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle /s captions during image group sessions.

    Telegram treats messages starting with / as commands, so they bypass the
    regular text handler. When grouping images, /s ... is stored verbatim
    (command + caption) for use on the generated album.
    """
    if not await _try_collect_caption_for_group_session(update, context):
        return


async def handle_image_group_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle done/cancel actions for image group collection sessions."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data
    if not callback_data or not callback_data.startswith("image_group_action:"):
        await query.edit_message_text("Error: selección inválida.")
        return

    action = callback_data.split(":")[1]
    session = _get_image_group_session(context)
    if not session:
        await query.edit_message_text("No hay una sesión de agrupación activa.")
        return

    correlation_id = session.get("correlation_id", str(uuid.uuid4())[:8])

    if action == "cancel":
        context.user_data.pop("image_group_session", None)
        await query.edit_message_text("Agrupación cancelada.")
        logger.info(f"[{correlation_id}] Image group session cancelled by user {user_id}")
        return

    if action != "done":
        await query.edit_message_text("Error: selección inválida.")
        return

    file_ids = session["file_ids"]
    if len(file_ids) < 2:
        await query.answer(
            "Necesitas al menos 2 imágenes para crear un álbum.",
            show_alert=True,
        )
        return

    await query.edit_message_text(f"Enviando álbum con {len(file_ids)} imágenes...")

    caption = session.get("caption")

    try:
        await _send_album_from_file_ids(
            update, context, file_ids, correlation_id, caption=caption
        )
        success_text = f"✅ Álbum enviado con {len(file_ids)} imágenes."
        if caption:
            success_text += " Incluye caption."
        await query.edit_message_text(success_text)
        logger.info(
            f"[{correlation_id}] Image group album sent to user {user_id} "
            f"({len(file_ids)} images"
            f"{', with caption' if caption else ''})"
        )
    except Exception as e:
        logger.error(f"[{correlation_id}] Failed to send grouped album: {e}")
        await query.edit_message_text(
            "No pude enviar el álbum. Intenta de nuevo con imágenes más pequeñas."
        )
    finally:
        context.user_data.pop("image_group_session", None)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages by showing an inline menu with image options.

    When a user sends a photo, displays an inline keyboard with options:
    - Comprimir: Reduce file size
    - Convertir Formato: Change image format
    - Redimensionar: Resize image
    - Info: Show image metadata

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Photo received from user {user_id}")

    # Get the largest photo size
    photo = update.message.photo[-1]

    # Validate file size
    if photo.file_size:
        is_valid, error_msg = validate_file_size(photo.file_size, config.max_incoming_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    if await _try_collect_image_for_group_session(update, context, photo.file_id):
        return

    await _schedule_image_batch_menu(update, context, photo.file_id)


async def handle_image_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle image files sent as documents.

    Some users send images as documents to preserve original quality.
    Shows the same image processing inline menu.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Image document received from user {user_id}")

    document = update.message.document

    # Validate file size
    if document.file_size:
        is_valid, error_msg = validate_file_size(document.file_size, config.max_incoming_file_size_mb)
        if not is_valid:
            logger.warning(f"[{correlation_id}] File size validation failed: {error_msg}")
            await update.message.reply_text(error_msg)
            return

    if await _try_collect_image_for_group_session(update, context, document.file_id):
        return

    await _schedule_image_batch_menu(update, context, document.file_id)


async def handle_image_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle image menu selection callbacks from inline keyboard.

    Routes to appropriate action based on user selection:
    - compress: Show compression quality options
    - convert: Show format selection
    - resize: Show resize percentage options
    - info: Show image metadata

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Parse callback data (format: "image_action:<action>")
    callback_data = query.data
    if not callback_data or not callback_data.startswith("image_action:"):
        logger.warning(f"Invalid callback data received: {callback_data}")
        await query.edit_message_text("Error: selección inválida.")
        return

    action = callback_data.split(":")[1]

    # Retrieve file_id from context
    file_ids = context.user_data.get("image_menu_file_ids") or []
    file_id = context.user_data.get("image_menu_file_id")
    correlation_id = context.user_data.get("image_menu_correlation_id", str(uuid.uuid4())[:8])

    if not file_id:
        logger.error(f"[{correlation_id}] No file_id found in context for user {user_id}")
        await query.edit_message_text("Error: no se encontró la imagen. Intenta de nuevo.")
        return

    if len(file_ids) > 1 and action not in ("enhance", "group", "noise"):
        await query.edit_message_text(
            "Solo «Mejorar», «Naturalizar» y «Agrupar» están disponibles para álbumes. "
            "Envía una imagen a la vez para otras acciones."
        )
        return

    logger.info(f"[{correlation_id}] Image menu action '{action}' selected by user {user_id}")

    if action == "group":
        group_file_ids = list(file_ids) if file_ids else [file_id]
        _start_image_group_session(context, group_file_ids, correlation_id)
        count = len(group_file_ids)
        prompt = (
            f"📷 *Modo agrupación activado*\n\n"
            f"Tienes *{count}* imagen(es). Envía más imágenes para agrupar "
            f"(máximo {config.MAX_IMAGE_BATCH_SIZE}).\n"
            "Opcional: envía un texto (o `/s ...`) y se usará como caption del álbum.\n\n"
            "Cuando termines, presiona *Listo* para recibirlas como álbum."
        )
        if count < 2:
            prompt += "\n\nNecesitas al menos 2 imágenes para crear un álbum."
        await query.edit_message_text(
            prompt,
            parse_mode="Markdown",
            reply_markup=_get_image_group_keyboard(count),
        )
        return

    if action == "compress":
        # Show compression quality selection
        keyboard = [
            [
                InlineKeyboardButton("Máxima (90%)", callback_data="image_compress:90"),
                InlineKeyboardButton("Alta (75%)", callback_data="image_compress:75"),
            ],
            [
                InlineKeyboardButton("Media (50%)", callback_data="image_compress:50"),
                InlineKeyboardButton("Baja (25%)", callback_data="image_compress:25"),
            ],
            [
                InlineKeyboardButton("Mínima (10%)", callback_data="image_compress:10"),
            ],
            [
                InlineKeyboardButton("← Volver", callback_data="back:image"),
                InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona el nivel de compresión:\n\n"
            "• Máxima (90%) - calidad casi sin pérdida\n"
            "• Alta (75%) - buen balance calidad/tamaño\n"
            "• Media (50%) - compresión notable\n"
            "• Baja (25%) - archivo pequeño\n"
            "• Mínima (10%) - máxima compresión",
            reply_markup=reply_markup
        )

    elif action == "convert":
        # Show format selection
        keyboard = [
            [
                InlineKeyboardButton("JPEG", callback_data="image_convert:jpeg"),
                InlineKeyboardButton("PNG", callback_data="image_convert:png"),
                InlineKeyboardButton("WebP", callback_data="image_convert:webp"),
            ],
            [
                InlineKeyboardButton("← Volver", callback_data="back:image"),
                InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona el formato de conversión:\n\n"
            "• JPEG - Alto compatible, con pérdida\n"
            "• PNG - Sin pérdida, ideal para gráficos\n"
            "• WebP - Buena compresión, moderno",
            reply_markup=reply_markup
        )

    elif action == "resize":
        # Show resize options
        keyboard = [
            [
                InlineKeyboardButton("25%", callback_data="image_resize:25"),
                InlineKeyboardButton("50%", callback_data="image_resize:50"),
                InlineKeyboardButton("75%", callback_data="image_resize:75"),
            ],
            [
                InlineKeyboardButton("150%", callback_data="image_resize:150"),
                InlineKeyboardButton("200%", callback_data="image_resize:200"),
            ],
            [
                InlineKeyboardButton("← Volver", callback_data="back:image"),
                InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona el porcentaje de redimensionamiento:",
            reply_markup=reply_markup
        )

    elif action == "enhance":
        keyboard = [
            [
                InlineKeyboardButton("Brillo", callback_data="image_enhance:brillo"),
                InlineKeyboardButton("Colores", callback_data="image_enhance:colores"),
            ],
            [
                InlineKeyboardButton("Nitidez", callback_data="image_enhance:nitidez"),
            ],
            [
                InlineKeyboardButton("Equilibrado", callback_data="image_enhance:equilibrado"),
                InlineKeyboardButton("Suave", callback_data="image_enhance:suave"),
            ],
            [
                InlineKeyboardButton("← Volver", callback_data="back:image"),
                InlineKeyboardButton("❌ Cancelar", callback_data="cancel"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Selecciona el perfil de mejora:\n\n"
            "• Brillo - más luminosidad y contraste automático\n"
            "• Colores - colores más vivos\n"
            "• Nitidez - mayor definición\n"
            "• Equilibrado - mejora general balanceada\n"
            "• Suave - mejora sutil",
            reply_markup=reply_markup
        )

    elif action == "noise":
        reply_markup = _get_image_noise_keyboard()
        await query.edit_message_text(
            "Selecciona la intensidad del ruido sutil:\n\n"
            "Añade textura fina para reducir el aspecto artificial de imágenes generadas con IA.\n"
            "• 1-2 - casi imperceptible (recomendado para empezar)\n"
            "• 3 - balance natural\n"
            "• 4-5 - más textura, sin estilo vintage",
            reply_markup=reply_markup
        )

    elif action == "info":
        await _handle_image_menu_info(update, context, file_id, correlation_id)


async def _handle_image_menu_info(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
    correlation_id: str
) -> None:
    """Show image metadata (dimensions, format, file size).

    Downloads the image, extracts metadata using Pillow, and displays it.

    Args:
        update: Telegram update object
        context: Telegram context object
        file_id: Telegram file ID of the image
        correlation_id: Correlation ID for tracing
    """
    query = update.callback_query
    user_id = update.effective_user.id

    await query.edit_message_text("Obteniendo información de la imagen...")

    with TempManager() as temp_mgr:
        try:
            input_filename = f"image_info_{user_id}_{correlation_id}.img"
            input_path = temp_mgr.get_temp_path(input_filename)

            # Download image
            logger.info(f"[{correlation_id}] Downloading image for info")
            try:
                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)
            except Exception as e:
                logger.error(f"[{correlation_id}] Download failed: {e}")
                await query.edit_message_text("No pude descargar la imagen. Intenta de nuevo.")
                return

            # Get image info
            info = ImageProcessor.get_image_info(input_path)
            if info["width"] == 0:
                await query.edit_message_text("No pude leer la información de la imagen.")
                return

            # Format file size
            size_bytes = info["file_size"]
            if size_bytes > 1024 * 1024:
                size_str = f"{size_bytes / (1024*1024):.1f} MB"
            else:
                size_str = f"{size_bytes / 1024:.0f} KB"

            # Format the info message
            format_display = info["format"].upper() if info["format"] != "unknown" else "Desconocido"
            info_text = (
                f"🖼 Información de la imagen:\n\n"
                f"📐 Dimensiones: {info['width']} x {info['height']} px\n"
                f"📦 Formato: {format_display}\n"
                f"📄 Peso: {size_str}\n"
                f"🎨 Modo de color: {info['mode']}"
            )

            await query.edit_message_text(info_text)

            reply_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton("← Volver al menú", callback_data="back:image")],
            ])
            await query.message.reply_text(
                "¿Quieres hacer algo más con esta imagen?",
                reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Image info displayed to user {user_id}")

        except Exception as e:
            logger.exception(f"[{correlation_id}] Error getting image info: {e}")
            await query.edit_message_text("Ocurrió un error al obtener la información.")


async def handle_image_compress_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle image compression quality selection.

    Downloads the image, compresses it with the selected quality,
    and sends the result back.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Parse callback data (format: "image_compress:<quality>")
    callback_data = query.data
    try:
        quality = int(callback_data.split(":")[1])
    except (IndexError, ValueError):
        await query.edit_message_text("Error: calidad inválida.")
        return

    # Retrieve file info
    file_id = context.user_data.get("image_menu_file_id")
    correlation_id = context.user_data.get("image_menu_correlation_id", str(uuid.uuid4())[:8])

    if not file_id:
        await query.edit_message_text("Error: no se encontró la imagen. Intenta de nuevo.")
        return

    logger.info(f"[{correlation_id}] Compressing image at quality {quality} for user {user_id}")
    await query.edit_message_text(f"Comprimiendo imagen (calidad {quality}%)...")

    with TempManager() as temp_mgr:
        try:
            input_filename = f"image_compress_input_{user_id}_{correlation_id}.img"
            output_filename = f"compressed_{user_id}_{correlation_id}.jpg"
            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download image
            logger.info(f"[{correlation_id}] Downloading image for compression")
            file = await context.bot.get_file(file_id)
            await _download_with_retry(file, input_path, correlation_id=correlation_id)

            # Compress
            loop = asyncio.get_event_loop()
            success, error = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: ImageProcessor.compress(str(input_path), str(output_path), quality=quality)
                ),
                timeout=config.PROCESSING_TIMEOUT
            )

            if not success:
                error_msg = error or "No pude comprimir la imagen"
                raise ImageCompressionError(error_msg)

            # Get original and compressed sizes for comparison
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            reduction = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

            # Send compressed image
            with open(output_path, "rb") as img_file:
                await query.message.reply_document(
                    document=img_file,
                    filename=f"comprimida_{correlation_id}.jpg",
                    caption=f"✅ Comprimida al {quality}%\n"
                            f"📦 {_format_size(original_size)} → {_format_size(compressed_size)} "
                            f"({reduction:.0f}% menos)"
                )

            reply_markup = _get_image_post_menu_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con esta imagen?",
                reply_markup=reply_markup
            )
            logger.info(
                f"[{correlation_id}] Image compressed: {original_size} -> {compressed_size} bytes "
                f"({reduction:.1f}% reduction)"
            )

        except ImageCompressionError as e:
            logger.error(f"[{correlation_id}] Compression failed: {e}")
            await query.edit_message_text(f"Error: {e.message}")
        except asyncio.TimeoutError:
            logger.error(f"[{correlation_id}] Compression timed out")
            await query.edit_message_text("La compresión tardó demasiado. Intenta con una imagen más pequeña.")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error compressing image: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def handle_image_convert_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle image format conversion selection.

    Downloads the image, converts it to the selected format,
    and sends the result back.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Parse callback data (format: "image_convert:<format>")
    callback_data = query.data
    try:
        target_format = callback_data.split(":")[1]
    except IndexError:
        await query.edit_message_text("Error: formato inválido.")
        return

    if target_format not in SUPPORTED_IMAGE_FORMATS:
        await query.edit_message_text(f"Formato '{target_format}' no soportado.")
        return

    # Retrieve file info
    file_id = context.user_data.get("image_menu_file_id")
    correlation_id = context.user_data.get("image_menu_correlation_id", str(uuid.uuid4())[:8])

    if not file_id:
        await query.edit_message_text("Error: no se encontró la imagen. Intenta de nuevo.")
        return

    fmt_info = SUPPORTED_IMAGE_FORMATS[target_format]
    logger.info(f"[{correlation_id}] Converting image to {target_format} for user {user_id}")
    await query.edit_message_text(f"Convirtiendo a {target_format.upper()}...")

    with TempManager() as temp_mgr:
        try:
            input_filename = f"image_convert_input_{user_id}_{correlation_id}.img"
            output_filename = f"convertida_{user_id}_{correlation_id}{fmt_info['ext']}"
            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download image
            logger.info(f"[{correlation_id}] Downloading image for conversion")
            file = await context.bot.get_file(file_id)
            await _download_with_retry(file, input_path, correlation_id=correlation_id)

            # Convert format
            loop = asyncio.get_event_loop()
            success, error = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: ImageProcessor.convert_format(str(input_path), str(output_path), target_format)
                ),
                timeout=config.PROCESSING_TIMEOUT
            )

            if not success:
                error_msg = error or "No pude convertir la imagen"
                raise ImageConversionError(error_msg)

            # Determine content type for sending
            caption = f"✅ Convertida a {target_format.upper()}"
            if os.path.exists(input_path):
                original_size = os.path.getsize(input_path)
                new_size = os.path.getsize(output_path)
                caption += f"\n📦 {_format_size(original_size)} → {_format_size(new_size)}"

            # Send converted image
            with open(output_path, "rb") as img_file:
                await query.message.reply_document(
                    document=img_file,
                    filename=f"convertida_{correlation_id}{fmt_info['ext']}",
                    caption=caption
                )

            reply_markup = _get_image_post_menu_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con esta imagen?",
                reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Image converted to {target_format} successfully")

        except ImageConversionError as e:
            logger.error(f"[{correlation_id}] Conversion failed: {e}")
            await query.edit_message_text(f"Error: {e.message}")
        except asyncio.TimeoutError:
            logger.error(f"[{correlation_id}] Conversion timed out")
            await query.edit_message_text("La conversión tardó demasiado. Intenta con una imagen más pequeña.")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error converting image: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def handle_image_resize_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle image resize percentage selection.

    Downloads the image, resizes it by the selected percentage,
    and sends the result back.

    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Parse callback data (format: "image_resize:<percentage>")
    callback_data = query.data
    try:
        percentage = int(callback_data.split(":")[1])
    except (IndexError, ValueError):
        await query.edit_message_text("Error: porcentaje inválido.")
        return

    if percentage < 1 or percentage > 1000:
        await query.edit_message_text("El porcentaje debe estar entre 1 y 1000.")
        return

    # Retrieve file info
    file_id = context.user_data.get("image_menu_file_id")
    correlation_id = context.user_data.get("image_menu_correlation_id", str(uuid.uuid4())[:8])

    if not file_id:
        await query.edit_message_text("Error: no se encontró la imagen. Intenta de nuevo.")
        return

    logger.info(f"[{correlation_id}] Resizing image to {percentage}% for user {user_id}")
    await query.edit_message_text(f"Redimensionando imagen al {percentage}%...")

    with TempManager() as temp_mgr:
        try:
            input_filename = f"image_resize_input_{user_id}_{correlation_id}.img"
            output_filename = f"redimensionada_{user_id}_{correlation_id}.jpg"
            input_path = temp_mgr.get_temp_path(input_filename)
            output_path = temp_mgr.get_temp_path(output_filename)

            # Download image
            logger.info(f"[{correlation_id}] Downloading image for resize")
            file = await context.bot.get_file(file_id)
            await _download_with_retry(file, input_path, correlation_id=correlation_id)

            # Get original dimensions for the caption
            orig_info = ImageProcessor.get_image_info(str(input_path))

            # Resize
            loop = asyncio.get_event_loop()
            success, error = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: ImageProcessor.resize(str(input_path), str(output_path), percentage=percentage)
                ),
                timeout=config.PROCESSING_TIMEOUT
            )

            if not success:
                error_msg = error or "No pude redimensionar la imagen"
                raise ImageResizeError(error_msg)

            # Get new dimensions
            new_info = ImageProcessor.get_image_info(str(output_path))

            caption = (
                f"✅ Redimensionada al {percentage}%\n"
                f"📐 {orig_info['width']}x{orig_info['height']} → "
                f"{new_info['width']}x{new_info['height']} px"
            )

            # Send resized image
            with open(output_path, "rb") as img_file:
                await query.message.reply_document(
                    document=img_file,
                    filename=f"redimensionada_{correlation_id}.jpg",
                    caption=caption
                )

            reply_markup = _get_image_post_menu_keyboard(correlation_id)
            await query.message.reply_text(
                "¡Listo! ¿Quieres hacer algo más con esta imagen?",
                reply_markup=reply_markup
            )
            logger.info(f"[{correlation_id}] Image resized to {percentage}% successfully")

        except ImageResizeError as e:
            logger.error(f"[{correlation_id}] Resize failed: {e}")
            await query.edit_message_text(f"Error: {e.message}")
        except asyncio.TimeoutError:
            logger.error(f"[{correlation_id}] Resize timed out")
            await query.edit_message_text("El redimensionamiento tardó demasiado. Intenta con una imagen más pequeña.")
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error resizing image: {e}")
            await query.edit_message_text("Ocurrió un error inesperado. Por favor intenta de nuevo.")


async def handle_image_enhance_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle image enhancement profile selection (supports batch albums)."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    try:
        profile = callback_data.split(":")[1]
    except IndexError:
        await query.edit_message_text("Error: perfil inválido.")
        return

    if profile not in ENHANCEMENT_PROFILES:
        await query.edit_message_text(f"Perfil '{profile}' no soportado.")
        return

    file_ids = context.user_data.get("image_menu_file_ids")
    if not file_ids:
        single_id = context.user_data.get("image_menu_file_id")
        file_ids = [single_id] if single_id else []

    correlation_id = context.user_data.get("image_menu_correlation_id", str(uuid.uuid4())[:8])

    if not file_ids:
        await query.edit_message_text("Error: no se encontraron imágenes. Intenta de nuevo.")
        return

    profile_label = ENHANCEMENT_PROFILES[profile]
    count = len(file_ids)
    batch_timeout = min(180, 30 + 15 * count)
    batch_deadline = time.monotonic() + batch_timeout

    required_space_mb = count * config.max_incoming_file_size_mb * 2
    has_space, space_error = check_disk_space(required_space_mb)
    if not has_space:
        await query.edit_message_text(space_error)
        return

    logger.info(
        f"[{correlation_id}] Enhancing {count} image(s) with profile "
        f"'{profile}' for user {user_id}"
    )
    await query.edit_message_text(
        f"Mejorando {count} imagen(es) con perfil {profile_label}..."
    )

    with TempManager() as temp_mgr:
        try:
            enhanced_paths = []
            loop = asyncio.get_event_loop()

            for idx, file_id in enumerate(file_ids, start=1):
                if count > 1:
                    try:
                        await query.edit_message_text(
                            f"Mejorando imagen {idx}/{count} ({profile_label})..."
                        )
                    except Exception:
                        pass

                input_filename = f"image_enhance_input_{user_id}_{correlation_id}_{idx}.img"
                output_filename = f"mejorada_{user_id}_{correlation_id}_{idx}.jpg"
                input_path = temp_mgr.get_temp_path(input_filename)
                output_path = temp_mgr.get_temp_path(output_filename)

                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)

                remaining_timeout = batch_deadline - time.monotonic()
                if remaining_timeout <= 0:
                    raise ProcessingTimeoutError(
                        "La mejora tardó demasiado. Intenta con menos imágenes o más pequeñas."
                    )
                remaining_timeout = max(5, remaining_timeout)

                success, error = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda inp=str(input_path), out=str(output_path), prof=profile: (
                            ImageProcessor.enhance(inp, out, prof)
                        ),
                    ),
                    timeout=remaining_timeout,
                )

                if not success:
                    error_msg = error or "No pude mejorar la imagen"
                    raise ImageEnhancementError(error_msg)

                enhanced_paths.append(str(output_path))

            caption = f"✅ Mejorada ({profile_label})"
            if count > 1:
                await _send_images_in_albums(
                    update,
                    context,
                    enhanced_paths,
                    correlation_id,
                    caption_prefix=f"Mejorada ({profile_label})",
                )
            else:
                with open(enhanced_paths[0], "rb") as img_file:
                    await query.message.reply_document(
                        document=img_file,
                        filename=f"mejorada_{correlation_id}.jpg",
                        caption=caption,
                    )

            await query.edit_message_text(
                f"¡Listo! {count} imagen(es) mejorada(s) con perfil {profile_label}."
            )

            reply_markup = _get_image_post_menu_keyboard(correlation_id)
            await query.message.reply_text(
                "¿Quieres hacer algo más con estas imágenes?",
                reply_markup=reply_markup,
            )
            logger.info(
                f"[{correlation_id}] Image enhancement completed for user {user_id}"
            )

        except ImageEnhancementError as e:
            logger.error(f"[{correlation_id}] Enhancement failed: {e}")
            await query.edit_message_text(f"Error: {get_user_error_message(e)}")
        except ProcessingTimeoutError as e:
            logger.error(f"[{correlation_id}] Enhancement timed out")
            await query.edit_message_text(f"Error: {get_user_error_message(e)}")
        except asyncio.TimeoutError:
            logger.error(f"[{correlation_id}] Enhancement timed out")
            await query.edit_message_text(
                "Error: La mejora tardó demasiado. Intenta con menos imágenes o más pequeñas."
            )
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error enhancing images: {e}")
            await query.edit_message_text(DEFAULT_ERROR_MESSAGE)


async def handle_image_noise_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle subtle noise intensity selection (supports batch albums)."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    try:
        strength = int(callback_data.split(":")[1])
    except (IndexError, ValueError):
        await query.edit_message_text("Error: intensidad inválida.")
        return

    if strength not in NOISE_STRENGTH_LEVELS:
        await query.edit_message_text(f"Intensidad '{strength}' no soportada.")
        return

    file_ids = context.user_data.get("image_menu_file_ids")
    if not file_ids:
        single_id = context.user_data.get("image_menu_file_id")
        file_ids = [single_id] if single_id else []

    correlation_id = context.user_data.get("image_menu_correlation_id", str(uuid.uuid4())[:8])

    if not file_ids:
        await query.edit_message_text("Error: no se encontraron imágenes. Intenta de nuevo.")
        return

    strength_label = NOISE_STRENGTH_LEVELS[strength]["label"]
    count = len(file_ids)
    batch_timeout = min(180, 30 + 15 * count)
    batch_deadline = time.monotonic() + batch_timeout

    required_space_mb = count * config.max_incoming_file_size_mb * 2
    has_space, space_error = check_disk_space(required_space_mb)
    if not has_space:
        await query.edit_message_text(space_error)
        return

    logger.info(
        f"[{correlation_id}] Applying subtle noise to {count} image(s) "
        f"(strength={strength}) for user {user_id}"
    )
    await query.edit_message_text(
        f"Naturalizando {count} imagen(es) ({strength_label})..."
    )

    with TempManager() as temp_mgr:
        try:
            processed_paths = []
            loop = asyncio.get_event_loop()

            for idx, file_id in enumerate(file_ids, start=1):
                if count > 1:
                    try:
                        await query.edit_message_text(
                            f"Naturalizando imagen {idx}/{count} ({strength_label})..."
                        )
                    except Exception:
                        pass

                input_filename = f"image_noise_input_{user_id}_{correlation_id}_{idx}.img"
                output_filename = f"naturalizada_{user_id}_{correlation_id}_{idx}.jpg"
                input_path = temp_mgr.get_temp_path(input_filename)
                output_path = temp_mgr.get_temp_path(output_filename)

                file = await context.bot.get_file(file_id)
                await _download_with_retry(file, input_path, correlation_id=correlation_id)

                remaining_timeout = batch_deadline - time.monotonic()
                if remaining_timeout <= 0:
                    raise ProcessingTimeoutError(
                        "El procesamiento tardó demasiado. Intenta con menos imágenes o más pequeñas."
                    )
                remaining_timeout = max(5, remaining_timeout)

                success, error = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda inp=str(input_path), out=str(output_path), lvl=strength: (
                            ImageProcessor.add_noise(inp, out, lvl)
                        ),
                    ),
                    timeout=remaining_timeout,
                )

                if not success:
                    error_msg = error or "No pude aplicar el ruido sutil"
                    raise ImageNoiseError(error_msg)

                processed_paths.append(str(output_path))

            caption = f"✅ Naturalizada ({strength_label})"
            if count > 1:
                await _send_images_in_albums(
                    update,
                    context,
                    processed_paths,
                    correlation_id,
                    caption_prefix=f"Naturalizada ({strength_label})",
                )
            else:
                with open(processed_paths[0], "rb") as img_file:
                    await query.message.reply_document(
                        document=img_file,
                        filename=f"naturalizada_{correlation_id}.jpg",
                        caption=caption,
                    )

            await query.edit_message_text(
                f"¡Listo! {count} imagen(es) naturalizada(s) con intensidad {strength_label}."
            )

            reply_markup = _get_image_post_menu_keyboard(correlation_id)
            await query.message.reply_text(
                "¿Quieres hacer algo más con estas imágenes?",
                reply_markup=reply_markup,
            )
            logger.info(
                f"[{correlation_id}] Subtle noise applied for user {user_id}"
            )

        except ImageNoiseError as e:
            logger.error(f"[{correlation_id}] Noise effect failed: {e}")
            await query.edit_message_text(f"Error: {get_user_error_message(e)}")
        except ProcessingTimeoutError as e:
            logger.error(f"[{correlation_id}] Noise effect timed out")
            await query.edit_message_text(f"Error: {get_user_error_message(e)}")
        except asyncio.TimeoutError:
            logger.error(f"[{correlation_id}] Noise effect timed out")
            await query.edit_message_text(
                "Error: El procesamiento tardó demasiado. Intenta con menos imágenes o más pequeñas."
            )
        except Exception as e:
            logger.exception(f"[{correlation_id}] Unexpected error applying noise: {e}")
            await query.edit_message_text(DEFAULT_ERROR_MESSAGE)


def _get_image_post_menu_keyboard(correlation_id: str) -> InlineKeyboardMarkup:
    """Generate inline keyboard for post-image-processing options.

    Args:
        correlation_id: Correlation ID for tracing

    Returns:
        InlineKeyboardMarkup with post-processing options
    """
    keyboard = [
        [
            InlineKeyboardButton("← Volver al menú", callback_data="back:image"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted string like "1.5 MB" or "500 KB"
    """
    if size_bytes > 1024 * 1024:
        return f"{size_bytes / (1024*1024):.1f} MB"
    elif size_bytes > 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes} B"
