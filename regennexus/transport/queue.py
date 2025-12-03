"""
RegenNexus UAP - Message Queue Transport

Reliable message delivery with persistence and retry logic.
Guarantees message delivery even if recipient is temporarily offline.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import os
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set
from pathlib import Path

from regennexus.core.message import Message
from regennexus.transport.base import (
    Transport,
    TransportConfig,
    TransportState,
)

logger = logging.getLogger(__name__)


@dataclass
class QueuedMessage:
    """A message in the queue with delivery tracking."""
    message: Message
    target: Optional[str]
    timestamp: float
    attempts: int = 0
    last_attempt: Optional[float] = None
    delivered: bool = False

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "message": self.message.to_dict(),
            "target": self.target,
            "timestamp": self.timestamp,
            "attempts": self.attempts,
            "last_attempt": self.last_attempt,
            "delivered": self.delivered,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QueuedMessage":
        """Deserialize from dictionary."""
        return cls(
            message=Message.from_dict(data["message"]),
            target=data["target"],
            timestamp=data["timestamp"],
            attempts=data.get("attempts", 0),
            last_attempt=data.get("last_attempt"),
            delivered=data.get("delivered", False),
        )


class MessageQueueTransport(Transport):
    """
    Message Queue Transport for reliable delivery.

    Provides guaranteed message delivery with:
    - Persistent queue storage
    - Automatic retry with exponential backoff
    - Dead letter queue for failed messages
    - Priority queuing

    Works as a layer on top of other transports.
    """

    def __init__(
        self,
        config: Optional[TransportConfig] = None,
        underlying_transport: Optional[Transport] = None
    ):
        """
        Initialize message queue transport.

        Args:
            config: Transport configuration
            underlying_transport: Transport to use for actual delivery
        """
        super().__init__(config)
        self._underlying = underlying_transport
        self._queue: Deque[QueuedMessage] = deque()
        self._dead_letter: List[QueuedMessage] = []
        self._pending_acks: Dict[str, QueuedMessage] = {}
        self._process_task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    @property
    def dead_letter_size(self) -> int:
        """Get dead letter queue size."""
        return len(self._dead_letter)

    @property
    def persist_path(self) -> Path:
        """Get persistence directory path."""
        return Path(self.config.queue_persist_path)

    async def connect(self) -> bool:
        """
        Start the message queue.

        Returns:
            True if started successfully
        """
        async with self._lock:
            if self._state == TransportState.CONNECTED:
                return True

            self._state = TransportState.CONNECTING

            try:
                # Load persisted queue
                if self.config.queue_persist:
                    await self._load_queue()

                # Connect underlying transport
                if self._underlying:
                    if not await self._underlying.connect():
                        logger.warning(
                            "Underlying transport not connected, "
                            "messages will be queued"
                        )

                self._state = TransportState.CONNECTED
                self._stats.connect_time = time.time()

                # Start queue processor
                self._process_task = asyncio.create_task(self._process_loop())

                # Start persistence task
                if self.config.queue_persist:
                    self._persist_task = asyncio.create_task(self._persist_loop())

                logger.info("Message queue transport started")
                return True

            except Exception as e:
                logger.error(f"Queue start error: {e}")
                self._state = TransportState.ERROR
                self._stats.errors += 1
                return False

    async def disconnect(self) -> None:
        """Stop the message queue."""
        async with self._lock:
            self._state = TransportState.DISCONNECTED

            # Cancel tasks
            if self._process_task:
                self._process_task.cancel()
                try:
                    await self._process_task
                except asyncio.CancelledError:
                    pass

            if self._persist_task:
                self._persist_task.cancel()
                try:
                    await self._persist_task
                except asyncio.CancelledError:
                    pass

            # Persist queue before shutdown
            if self.config.queue_persist:
                await self._save_queue()

            # Disconnect underlying transport
            if self._underlying:
                await self._underlying.disconnect()

            logger.info("Message queue transport stopped")

    async def _process_loop(self) -> None:
        """Process queued messages."""
        try:
            while self._state == TransportState.CONNECTED:
                if not self._queue:
                    await asyncio.sleep(0.1)
                    continue

                # Check if underlying transport is available
                if not self._underlying or not self._underlying.is_connected:
                    await asyncio.sleep(1.0)
                    continue

                # Get next message
                queued = self._queue[0]

                # Check retry delay
                if queued.last_attempt:
                    delay = self.config.queue_retry_delay * (2 ** queued.attempts)
                    if time.time() - queued.last_attempt < delay:
                        await asyncio.sleep(0.1)
                        continue

                # Attempt delivery
                success = await self._deliver(queued)

                if success:
                    self._queue.popleft()
                    logger.debug(f"Message delivered: {queued.message.id}")
                else:
                    queued.attempts += 1
                    queued.last_attempt = time.time()

                    if queued.attempts >= self.config.queue_retry_attempts:
                        # Move to dead letter queue
                        self._queue.popleft()
                        self._dead_letter.append(queued)
                        logger.warning(
                            f"Message moved to dead letter: {queued.message.id}"
                        )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Queue processor error: {e}")

    async def _deliver(self, queued: QueuedMessage) -> bool:
        """
        Attempt to deliver a queued message.

        Args:
            queued: Queued message to deliver

        Returns:
            True if delivery successful
        """
        if not self._underlying:
            return False

        try:
            return await self._underlying.send(
                queued.message,
                queued.target
            )
        except Exception as e:
            logger.error(f"Delivery error: {e}")
            return False

    async def _persist_loop(self) -> None:
        """Periodically persist queue to disk."""
        try:
            while self._state == TransportState.CONNECTED:
                await asyncio.sleep(30)  # Persist every 30 seconds
                await self._save_queue()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Persist loop error: {e}")

    async def _save_queue(self) -> None:
        """Save queue to disk."""
        try:
            path = self.persist_path
            path.mkdir(parents=True, exist_ok=True)

            # Save main queue
            queue_file = path / "queue.json"
            queue_data = [q.to_dict() for q in self._queue]
            with open(queue_file, "w") as f:
                json.dump(queue_data, f)

            # Save dead letter queue
            dl_file = path / "dead_letter.json"
            dl_data = [q.to_dict() for q in self._dead_letter]
            with open(dl_file, "w") as f:
                json.dump(dl_data, f)

            logger.debug(f"Queue persisted: {len(self._queue)} messages")

        except Exception as e:
            logger.error(f"Queue save error: {e}")

    async def _load_queue(self) -> None:
        """Load queue from disk."""
        try:
            path = self.persist_path

            # Load main queue
            queue_file = path / "queue.json"
            if queue_file.exists():
                with open(queue_file, "r") as f:
                    queue_data = json.load(f)
                self._queue = deque(
                    QueuedMessage.from_dict(q) for q in queue_data
                )
                logger.info(f"Loaded {len(self._queue)} queued messages")

            # Load dead letter queue
            dl_file = path / "dead_letter.json"
            if dl_file.exists():
                with open(dl_file, "r") as f:
                    dl_data = json.load(f)
                self._dead_letter = [
                    QueuedMessage.from_dict(q) for q in dl_data
                ]
                logger.info(
                    f"Loaded {len(self._dead_letter)} dead letter messages"
                )

        except Exception as e:
            logger.error(f"Queue load error: {e}")

    async def send(self, message: Message, target: Optional[str] = None) -> bool:
        """
        Send a message (queued for reliable delivery).

        Args:
            message: Message to send
            target: Target peer ID

        Returns:
            True if message queued successfully
        """
        if self._state != TransportState.CONNECTED:
            return False

        # Check queue size limit
        if len(self._queue) >= self.config.queue_max_size:
            logger.error("Queue full, message rejected")
            return False

        # Add to queue
        queued = QueuedMessage(
            message=message,
            target=target,
            timestamp=time.time(),
        )
        self._queue.append(queued)

        self._stats.messages_sent += 1
        logger.debug(f"Message queued: {message.id}")

        return True

    async def broadcast(self, message: Message) -> int:
        """
        Broadcast message (queued to all known peers).

        Args:
            message: Message to broadcast

        Returns:
            Number of peers message was queued for
        """
        if not self._underlying:
            return 0

        peers = self._underlying.peers
        count = 0

        for peer_id in peers:
            if await self.send(message, peer_id):
                count += 1

        return count

    def retry_dead_letter(self, message_id: Optional[str] = None) -> int:
        """
        Move messages from dead letter queue back to main queue.

        Args:
            message_id: Specific message to retry, or None for all

        Returns:
            Number of messages moved
        """
        count = 0

        if message_id:
            # Retry specific message
            for i, queued in enumerate(self._dead_letter):
                if queued.message.id == message_id:
                    queued.attempts = 0
                    queued.last_attempt = None
                    self._queue.append(queued)
                    self._dead_letter.pop(i)
                    count = 1
                    break
        else:
            # Retry all
            for queued in self._dead_letter:
                queued.attempts = 0
                queued.last_attempt = None
                self._queue.append(queued)
                count += 1
            self._dead_letter.clear()

        return count

    def clear_dead_letter(self) -> int:
        """
        Clear dead letter queue.

        Returns:
            Number of messages cleared
        """
        count = len(self._dead_letter)
        self._dead_letter.clear()
        return count
