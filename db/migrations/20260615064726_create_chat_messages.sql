-- migrate:up
-- US-EPIC7 (Be My Eyes conversational): persisted chat memory per session.
-- One row per conversation turn. Session-scoped (multi-tenant NON-NEGOTIABLE):
-- every read filters by session_id. CASCADE so deleting a session reaps history.
-- role limited to user|assistant via TEXT CHECK (no native PG enum, repo convention).
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_messages_session_id_idx ON chat_messages(session_id);

-- migrate:down
DROP TABLE IF EXISTS chat_messages CASCADE;
