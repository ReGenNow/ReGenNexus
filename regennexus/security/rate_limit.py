"""
RegenNexus UAP - Rate Limiting Module

Provides request rate limiting to prevent abuse and ensure
fair usage of resources.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RateLimitResult(Enum):
    """Rate limit check result."""
    ALLOWED = auto()
    RATE_LIMITED = auto()
    BURST_LIMITED = auto()


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    # Requests per minute
    requests_per_minute: int = 100

    # Burst limit (max requests in short window)
    burst_limit: int = 20
    burst_window: float = 1.0  # seconds

    # Cooldown after hitting limit
    cooldown_seconds: float = 60.0

    # Per-entity limits (more granular control)
    per_entity_rpm: Optional[int] = None
    per_ip_rpm: Optional[int] = None

    # Whitelist/blacklist
    whitelist: List[str] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)


@dataclass
class RateLimitState:
    """State for tracking rate limits."""
    requests: List[float] = field(default_factory=list)
    burst_requests: List[float] = field(default_factory=list)
    limited_until: Optional[float] = None
    total_requests: int = 0
    total_limited: int = 0


class RateLimiter:
    """
    Rate limiter for request throttling.

    Implements a sliding window rate limiter with burst protection.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limiting configuration
        """
        self.config = config or RateLimitConfig()
        self._states: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._global_state = RateLimitState()
        self._lock = asyncio.Lock()

    def check(
        self,
        identifier: str,
        check_global: bool = True
    ) -> Tuple[RateLimitResult, Optional[float]]:
        """
        Check if a request is allowed.

        Args:
            identifier: Entity/IP identifier
            check_global: Also check global rate limit

        Returns:
            Tuple of (result, retry_after_seconds)
        """
        now = time.time()

        # Check whitelist
        if identifier in self.config.whitelist:
            return RateLimitResult.ALLOWED, None

        # Check blacklist
        if identifier in self.config.blacklist:
            return RateLimitResult.RATE_LIMITED, self.config.cooldown_seconds

        # Get state for this identifier
        state = self._states[identifier]

        # Check if in cooldown
        if state.limited_until and now < state.limited_until:
            retry_after = state.limited_until - now
            return RateLimitResult.RATE_LIMITED, retry_after

        # Clean old requests
        self._clean_old_requests(state, now)

        # Check burst limit
        burst_count = len(state.burst_requests)
        if burst_count >= self.config.burst_limit:
            state.total_limited += 1
            state.limited_until = now + self.config.cooldown_seconds
            logger.warning(f"Burst limit hit for {identifier}")
            return RateLimitResult.BURST_LIMITED, self.config.cooldown_seconds

        # Check rate limit
        rpm = self.config.per_entity_rpm or self.config.requests_per_minute
        if len(state.requests) >= rpm:
            state.total_limited += 1
            state.limited_until = now + self.config.cooldown_seconds
            logger.warning(f"Rate limit hit for {identifier}")
            return RateLimitResult.RATE_LIMITED, self.config.cooldown_seconds

        # Check global limit
        if check_global:
            self._clean_old_requests(self._global_state, now)
            if len(self._global_state.requests) >= self.config.requests_per_minute * 10:
                return RateLimitResult.RATE_LIMITED, 1.0

        return RateLimitResult.ALLOWED, None

    def record(self, identifier: str) -> None:
        """
        Record a request.

        Args:
            identifier: Entity/IP identifier
        """
        now = time.time()
        state = self._states[identifier]

        state.requests.append(now)
        state.burst_requests.append(now)
        state.total_requests += 1

        # Also record globally
        self._global_state.requests.append(now)
        self._global_state.total_requests += 1

    async def check_and_record(
        self,
        identifier: str
    ) -> Tuple[RateLimitResult, Optional[float]]:
        """
        Check rate limit and record if allowed.

        Args:
            identifier: Entity/IP identifier

        Returns:
            Tuple of (result, retry_after_seconds)
        """
        async with self._lock:
            result, retry_after = self.check(identifier)

            if result == RateLimitResult.ALLOWED:
                self.record(identifier)

            return result, retry_after

    def _clean_old_requests(self, state: RateLimitState, now: float) -> None:
        """
        Remove expired requests from tracking.

        Args:
            state: Rate limit state to clean
            now: Current timestamp
        """
        # Clean minute window
        minute_ago = now - 60
        state.requests = [t for t in state.requests if t > minute_ago]

        # Clean burst window
        burst_ago = now - self.config.burst_window
        state.burst_requests = [t for t in state.burst_requests if t > burst_ago]

    def get_stats(self, identifier: str) -> Dict:
        """
        Get rate limit statistics for an identifier.

        Args:
            identifier: Entity/IP identifier

        Returns:
            Statistics dictionary
        """
        state = self._states.get(identifier, RateLimitState())
        now = time.time()

        # Calculate current usage
        self._clean_old_requests(state, now)

        return {
            "identifier": identifier,
            "requests_in_window": len(state.requests),
            "burst_requests": len(state.burst_requests),
            "total_requests": state.total_requests,
            "total_limited": state.total_limited,
            "is_limited": state.limited_until is not None and now < state.limited_until,
            "limited_until": state.limited_until,
            "limit": self.config.per_entity_rpm or self.config.requests_per_minute,
            "burst_limit": self.config.burst_limit,
        }

    def get_global_stats(self) -> Dict:
        """
        Get global rate limit statistics.

        Returns:
            Statistics dictionary
        """
        now = time.time()
        self._clean_old_requests(self._global_state, now)

        return {
            "requests_in_window": len(self._global_state.requests),
            "total_requests": self._global_state.total_requests,
            "total_limited": self._global_state.total_limited,
            "unique_identifiers": len(self._states),
            "limit": self.config.requests_per_minute,
        }

    def reset(self, identifier: Optional[str] = None) -> None:
        """
        Reset rate limit state.

        Args:
            identifier: Specific identifier to reset, or None for all
        """
        if identifier:
            if identifier in self._states:
                self._states[identifier] = RateLimitState()
        else:
            self._states.clear()
            self._global_state = RateLimitState()

    def add_to_whitelist(self, identifier: str) -> None:
        """Add an identifier to the whitelist."""
        if identifier not in self.config.whitelist:
            self.config.whitelist.append(identifier)

    def remove_from_whitelist(self, identifier: str) -> None:
        """Remove an identifier from the whitelist."""
        if identifier in self.config.whitelist:
            self.config.whitelist.remove(identifier)

    def add_to_blacklist(self, identifier: str) -> None:
        """Add an identifier to the blacklist."""
        if identifier not in self.config.blacklist:
            self.config.blacklist.append(identifier)

    def remove_from_blacklist(self, identifier: str) -> None:
        """Remove an identifier from the blacklist."""
        if identifier in self.config.blacklist:
            self.config.blacklist.remove(identifier)


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts limits based on system load.

    Automatically reduces limits when system is under high load
    and increases them when load is low.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        min_multiplier: float = 0.1,
        max_multiplier: float = 2.0
    ):
        """
        Initialize adaptive rate limiter.

        Args:
            config: Rate limiting configuration
            min_multiplier: Minimum limit multiplier
            max_multiplier: Maximum limit multiplier
        """
        super().__init__(config)
        self._min_multiplier = min_multiplier
        self._max_multiplier = max_multiplier
        self._current_multiplier = 1.0
        self._load_samples: List[float] = []

    def update_load(self, load: float) -> None:
        """
        Update system load for adaptive limiting.

        Args:
            load: Current system load (0.0 to 1.0+)
        """
        self._load_samples.append(load)

        # Keep only recent samples
        if len(self._load_samples) > 60:
            self._load_samples = self._load_samples[-60:]

        # Calculate average load
        avg_load = sum(self._load_samples) / len(self._load_samples)

        # Adjust multiplier based on load
        if avg_load > 0.9:
            # High load - reduce limits
            self._current_multiplier = max(
                self._min_multiplier,
                self._current_multiplier * 0.9
            )
        elif avg_load < 0.5:
            # Low load - increase limits
            self._current_multiplier = min(
                self._max_multiplier,
                self._current_multiplier * 1.1
            )

    def check(
        self,
        identifier: str,
        check_global: bool = True
    ) -> Tuple[RateLimitResult, Optional[float]]:
        """Check rate limit with adaptive multiplier."""
        # Temporarily adjust limits
        original_rpm = self.config.requests_per_minute
        original_burst = self.config.burst_limit

        self.config.requests_per_minute = int(original_rpm * self._current_multiplier)
        self.config.burst_limit = int(original_burst * self._current_multiplier)

        try:
            return super().check(identifier, check_global)
        finally:
            # Restore original limits
            self.config.requests_per_minute = original_rpm
            self.config.burst_limit = original_burst

    @property
    def current_multiplier(self) -> float:
        """Get current limit multiplier."""
        return self._current_multiplier
