"""Generic async repository (DRY base for every concrete repository).

Global user rule: every concrete repository MUST extend :class:`BaseRepository`
to reuse the read/write helpers already implemented here instead of
re-implementing CRUD against the :class:`AsyncSession`.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel


class BaseRepository[ModelT: SQLModel]:
    """Reusable CRUD helpers bound to one SQLModel table and an AsyncSession.

    Concrete repositories pass the model class and the live session, then call
    the inherited helpers. Session-scoped filtering is layered on top by the
    concrete repositories (multi-tenant NON-NEGOTIABLE).
    """

    def __init__(self, model: type[ModelT], session: AsyncSession) -> None:
        self.model = model
        self.session = session

    async def get(self, obj_id: object) -> ModelT | None:
        """Return the row with the given primary key, or ``None``."""
        return await self.session.get(self.model, obj_id)

    async def list(self, *, limit: int | None = None) -> Sequence[ModelT]:
        """Return all rows of the table (optionally capped by ``limit``)."""
        stmt = select(self.model)
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    def add(self, obj: ModelT) -> ModelT:
        """Stage a new row for insertion (flush/commit happen explicitly)."""
        self.session.add(obj)
        return obj

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()

    async def refresh(self, obj: ModelT) -> ModelT:
        """Reload server-generated columns (PK, defaults) into ``obj``."""
        await self.session.refresh(obj)
        return obj

    async def add_commit_refresh(self, obj: ModelT) -> ModelT:
        """Convenience: add, commit and refresh in one call."""
        self.add(obj)
        await self.commit()
        return await self.refresh(obj)
