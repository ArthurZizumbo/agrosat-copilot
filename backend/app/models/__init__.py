"""SQLModel + GeoAlchemy2 models mirroring the dbmate-managed schema.

The schema is owned by ``db/migrations/*.sql`` (dbmate). These models reflect it
exactly for the ORM/repository layer; they must never drive DDL
(``create_all()`` is forbidden — root rule 11).
"""

from backend.app.models.aoi import Aoi
from backend.app.models.chat_message import ChatMessage
from backend.app.models.feature import FeatureParcel
from backend.app.models.parcel import Parcel
from backend.app.models.session import ChatSession

__all__ = [
    "Aoi",
    "ChatMessage",
    "ChatSession",
    "FeatureParcel",
    "Parcel",
]
