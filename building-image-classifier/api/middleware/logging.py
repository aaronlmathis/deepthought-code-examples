import time
import uuid
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive request logging middleware for production monitoring.
    
    Logs every request with timing, status codes, and unique identifiers.
    Essential for debugging, performance monitoring, and security auditing.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger("api.requests")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())[:8]
        
        # Extract client information safely
        client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Start timing
        start_time = time.time()
        
        # Log incoming request
        self.logger.info(
            f"REQUEST {request_id} | {request.method} {request.url.path} | "
            f"Client: {client_ip} | User-Agent: {user_agent[:50]}{'...' if len(user_agent) > 50 else ''}"
        )
        
        # Add request ID to request state for use in endpoints
        request.state.request_id = request_id
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            log_level = logging.INFO if response.status_code < 400 else logging.WARNING
            self.logger.log(
                log_level,
                f"RESPONSE {request_id} | {response.status_code} | "
                f"{process_time*1000:.2f}ms | {request.method} {request.url.path}"
            )
            
            # Add custom headers for monitoring
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # Log exceptions
            process_time = time.time() - start_time
            self.logger.error(
                f"ERROR {request_id} | {str(e)} | "
                f"{process_time*1000:.2f}ms | {request.method} {request.url.path}",
                exc_info=True
            )
            raise