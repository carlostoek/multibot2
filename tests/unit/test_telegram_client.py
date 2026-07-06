"""Tests for Telegram Application builder."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bot.telegram_client import create_application, derive_file_base_url


def _mock_config(*, local_mode: bool):
    return SimpleNamespace(
        BOT_TOKEN="test-token",
        TELEGRAM_LOCAL_MODE=local_mode,
        TELEGRAM_API_BASE_URL="http://127.0.0.1:8081/bot",
        TELEGRAM_API_FILE_BASE_URL=None,
        TELEGRAM_API_TIMEOUT=45.0,
        TELEGRAM_MAX_UPLOAD_SIZE_MB=2000,
    )


class TestDeriveFileBaseUrl:
    def test_derives_file_url_from_bot_base_url(self):
        assert (
            derive_file_base_url("http://telegram-bot-api.railway.internal:8081/bot")
            == "http://telegram-bot-api.railway.internal:8081/file/bot"
        )


class TestTelegramClient:
    """Validate Application builder configuration."""

    @patch("bot.telegram_client.ApplicationBuilder")
    def test_cloud_mode_configures_timeouts(self, mock_builder_cls):
        mock_builder = MagicMock()
        mock_builder.token.return_value = mock_builder
        mock_builder.connect_timeout.return_value = mock_builder
        mock_builder.read_timeout.return_value = mock_builder
        mock_builder.write_timeout.return_value = mock_builder
        mock_builder.pool_timeout.return_value = mock_builder
        mock_builder.media_write_timeout.return_value = mock_builder
        mock_builder.build.return_value = MagicMock()
        mock_builder_cls.return_value = mock_builder

        with patch("bot.telegram_client.config", _mock_config(local_mode=False)):
            create_application()

        mock_builder.token.assert_called_once_with("test-token")
        mock_builder.base_url.assert_not_called()
        mock_builder.local_mode.assert_not_called()
        mock_builder.connect_timeout.assert_called_once_with(45.0)
        mock_builder.read_timeout.assert_called_once_with(45.0)
        mock_builder.write_timeout.assert_called_once_with(45.0)
        mock_builder.pool_timeout.assert_called_once_with(45.0)
        mock_builder.media_write_timeout.assert_called_once_with(120.0)

    @patch("bot.telegram_client.ApplicationBuilder")
    def test_local_mode_configures_base_url_and_timeouts(self, mock_builder_cls):
        mock_builder = MagicMock()
        mock_builder.token.return_value = mock_builder
        mock_builder.base_url.return_value = mock_builder
        mock_builder.base_file_url.return_value = mock_builder
        mock_builder.local_mode.return_value = mock_builder
        mock_builder.connect_timeout.return_value = mock_builder
        mock_builder.read_timeout.return_value = mock_builder
        mock_builder.write_timeout.return_value = mock_builder
        mock_builder.pool_timeout.return_value = mock_builder
        mock_builder.media_write_timeout.return_value = mock_builder
        mock_builder.build.return_value = MagicMock()
        mock_builder_cls.return_value = mock_builder

        with patch("bot.telegram_client.config", _mock_config(local_mode=True)):
            create_application()

        mock_builder.base_url.assert_called_once_with("http://127.0.0.1:8081/bot")
        mock_builder.base_file_url.assert_called_once_with("http://127.0.0.1:8081/file/bot")
        mock_builder.local_mode.assert_called_once_with(True)
        mock_builder.connect_timeout.assert_called_once_with(45.0)
        mock_builder.read_timeout.assert_called_once_with(45.0)
        mock_builder.write_timeout.assert_called_once_with(45.0)
        mock_builder.pool_timeout.assert_called_once_with(45.0)
        mock_builder.media_write_timeout.assert_called_once_with(120.0)