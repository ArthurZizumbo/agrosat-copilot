"""Background inference workers (US-056, deferred to Full -- ADR-009 / ADR-012).

The MVP runs inference synchronously in the backend (see
``backend.app.services.jobs_service.JobsService`` in ``sync`` mode). The Pub/Sub
subscriber here is the Full/futuro counterpart -- its subscription loop does NOT
run today (no GCP Pub/Sub topic, no Cloud Run GPU service).
"""
