import time
from collections import defaultdict, deque
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Dict, Deque

class SimpleRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.
    
    For production, consider using Redis-based rate limiting for:
    - Distributed systems
    - Persistent rate limit state
    - More sophisticated algorithms
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        
        # In-memory storage (use Redis in production)
        self.requests: Dict[str, Deque[float]] = defaultdict(deque)
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # In production, consider using API keys or user IDs
        client_ip = getattr(request.client, 'host', 'unknown') if request.client else "unknown"
        return client_ip
    
    def _cleanup_old_requests(self, request_times: Deque[float], current_time: float):
        """Remove requests outside the current window"""
        while request_times and current_time - request_times[0] > self.window_seconds:
            request_times.popleft()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        client_id = self._get_client_identifier(request)
        current_time = time.time()
        
        # Get client's request history
        client_requests = self.requests[client_id]
        
        # Clean up old requests
        self._cleanup_old_requests(client_requests, current_time)
        
        # Check rate limit
        if len(client_requests) >= self.requests_per_minute:
            # Calculate time until rate limit resets
            oldest_request = client_requests[0] if client_requests else current_time
            reset_time = oldest_request + self.window_seconds
            retry_after = max(1, int(reset_time - current_time))
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                    "retry_after": retry_after
                }
            )
        
        # Record this request
        client_requests.append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self.requests_per_minute - len(client_requests))
        
        # Calculate reset time more accurately
        if client_requests:
            oldest_request = client_requests[0]
            reset_time = oldest_request + self.window_seconds
        else:
            reset_time = current_time + self.window_seconds
            
        response.headers["X-RateLimit-Reset"] = str(int(reset_time))
        
        return response