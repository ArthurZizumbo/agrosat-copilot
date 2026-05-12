---
name: agrosat-security
description: Implement authentication (Clerk OAuth), JWT validation, rate limiting, CSP headers, RLS policies per session_id, audit logging, and input validation for AgroSatCopilot. Use when adding security controls, role guards, or access control.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Security Skill

## Auth Model

- **Provider**: Clerk OAuth (Google + Microsoft + email/password)
- **Roles**: `super_admin`, `user` (per-AOI tenancy via `session_id`)
- **JWT**: validated en cada request via `Depends(get_current_user)`
- **Multi-tenant key**: `session_id` (no `wedding_id`)

## get_current_user

```python
# backend/app/core/auth.py
from fastapi import Depends, HTTPException, Request
from clerk_sdk import authenticate_request
from app.models.user import UserPublic

async def get_current_user(request: Request) -> UserPublic:
    auth_state = authenticate_request(
        request,
        secret_key=settings.CLERK_SECRET_KEY,
        authorized_parties=[settings.FRONTEND_URL],
    )
    if not auth_state.is_signed_in:
        raise HTTPException(401, "Not authenticated")
    return UserPublic(
        id=auth_state.payload["sub"],
        email=auth_state.payload["email"],
        role=auth_state.payload.get("public_metadata", {}).get("role", "user"),
        session_ids=auth_state.payload.get("public_metadata", {}).get("session_ids", []),
    )
```

## Session Owner Check

```python
def _check_session_owner(session_id: str, user: UserPublic) -> None:
    if user.role == "super_admin":
        return
    if session_id not in user.session_ids:
        raise HTTPException(403, "Access denied")
```

## Rate Limit (per user_id)

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

def get_user_key(request: Request) -> str:
    # Usa user_id si está autenticado, sino IP
    return getattr(request.state, "user_id", get_remote_address(request))

limiter = Limiter(key_func=get_user_key)

@router.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, ...): ...

@router.post("/llm/switch")
@limiter.limit("5/minute")
async def switch(request: Request, ...): ...
```

## CSP + Security Headers

```python
# backend/app/middleware/security_headers.py
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://*.clerk.accounts.dev; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https://storage.googleapis.com; "
            "connect-src 'self' https://*.run.app https://*.clerk.accounts.dev wss:; "
            "frame-ancestors 'none';"
        )
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response
```

## RLS per session_id

```sql
ALTER TABLE aois ENABLE ROW LEVEL SECURITY;
CREATE POLICY aois_session_iso ON aois
    USING (session_id::text = current_setting('app.current_session_id', true));
```

```python
async def get_session(request: Request) -> AsyncSession:
    async with AsyncSessionLocal() as session:
        sid = getattr(request.state, "session_id", None)
        if sid:
            await session.execute(text("SET LOCAL app.current_session_id = :s"), {"s": str(sid)})
        yield session
```

## Pydantic Input Validation

```python
from pydantic import BaseModel, field_validator
import re

class ChatRequest(BaseModel):
    query: str
    aoi_geojson: dict
    llm_variant: Literal["gemini", "qwen35"]
    session_id: str

    @field_validator("query")
    @classmethod
    def no_html(cls, v: str) -> str:
        if re.search(r"<[^>]+>", v):
            raise ValueError("HTML not allowed in query")
        if len(v) > 4000:
            raise ValueError("Query too long")
        return v.strip()

    @field_validator("aoi_geojson")
    @classmethod
    def validate_geojson(cls, v: dict) -> dict:
        if v.get("type") not in {"Feature", "FeatureCollection"}:
            raise ValueError("Invalid GeoJSON type")
        return v
```

## Audit Logging

```python
import structlog
logger = structlog.get_logger()

logger.info(
    "resource_modified",
    action="classify",
    resource="aoi",
    resource_id=aoi.id,
    session_id=session_id,
    user_id=current_user.id,
    ip=request.client.host,
    llm_variant=llm_variant,
)
```

## File Upload Validation

```python
import magic

ALLOWED_MIME = {"image/jpeg", "image/png", "application/geo+json", "application/json"}
MAX_SIZE = 20 * 1024 * 1024  # 20 MB

async def validate_upload(file: UploadFile) -> bytes:
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(400, "File too large")
    mime = magic.from_buffer(content[:2048], mime=True)
    if mime not in ALLOWED_MIME:
        raise HTTPException(400, f"Invalid file type: {mime}")
    return content
```

## Frontend Role Guard

```typescript
// frontend/middleware/role-guard.ts
export default defineNuxtRouteMiddleware((to) => {
  const { user } = useUser()
  const requiredRoles = to.meta.roles as string[] | undefined
  if (!requiredRoles) return
  if (!user.value || !requiredRoles.includes(user.value.publicMetadata?.role)) {
    return navigateTo('/unauthorized')
  }
})
```
