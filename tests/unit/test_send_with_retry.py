"""Tests for Telegram send retry helper."""
import pytest
from unittest.mock import AsyncMock, patch

from telegram.error import TimedOut, NetworkError

from bot.handlers import _send_with_retry


@pytest.mark.asyncio
async def test_send_with_retry_succeeds_on_first_attempt():
    send_fn = AsyncMock(return_value="ok")

    result = await _send_with_retry(send_fn, correlation_id="abc123", label="reply_video")

    assert result == "ok"
    send_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_with_retry_retries_on_timed_out_then_succeeds():
    send_fn = AsyncMock(side_effect=[TimedOut("Timed out"), TimedOut("Timed out"), "ok"])

    with patch("bot.handlers.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
        result = await _send_with_retry(send_fn, correlation_id="abc123", label="reply_video")

    assert result == "ok"
    assert send_fn.await_count == 3
    assert sleep_mock.await_count == 2
    sleep_mock.assert_any_await(2)
    sleep_mock.assert_any_await(4)


@pytest.mark.asyncio
async def test_send_with_retry_raises_after_max_attempts():
    send_fn = AsyncMock(side_effect=TimedOut("Timed out"))

    with patch("bot.handlers.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(TimedOut):
            await _send_with_retry(send_fn, max_retries=3, label="reply_video")

    assert send_fn.await_count == 3


@pytest.mark.asyncio
async def test_send_with_retry_does_not_retry_network_error():
    send_fn = AsyncMock(side_effect=NetworkError("network down"))

    with pytest.raises(NetworkError):
        await _send_with_retry(send_fn, label="reply_video")

    send_fn.assert_awaited_once()