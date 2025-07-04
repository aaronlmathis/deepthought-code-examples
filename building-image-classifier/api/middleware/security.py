from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security headers middleware for production deployment.
    
    Adds essential security headers to protect against common web vulnerabilities:
    - XSS attacks
    - Clickjacking
    - MIME type sniffing
    - Information leakage
    - HTTPS enforcement
    """
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers for production
        security_headers = {
            # Prevent XSS attacks
            "X-Content-Type-Options": "nosniff",
            
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            
            # XSS protection (deprecated but still useful for older browsers)
            "X-XSS-Protection": "1; mode=block",
            
            # Hide server information
            "Server": "PyramidNet-API",
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Content Security Policy (more permissive for API with docs)
            "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'",
            
            # HTTPS enforcement (only in production)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            
            # Permissions policy (restrict browser features)
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        # Apply security headers
        for header, value in security_headers.items():
            if value:  # Only set headers with non-empty values
                response.headers[header] = value
        
        # Remove potentially sensitive headers if they exist
        sensitive_headers = ["X-Powered-By", "Server"]
        for header in sensitive_headers:
            if header in response.headers and header != "Server":  # Keep our custom Server header
                del response.headers[header]
        
        return response