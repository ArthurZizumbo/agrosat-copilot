-- migrate:up
-- US-080: list a user's chat sessions despite per-session RLS.
--
-- The chat_sessions RLS policy (US-051) only exposes the row whose id equals
-- app.current_session, so the NOBYPASSRLS app role cannot SELECT "all sessions
-- of user X". A SECURITY DEFINER function owned by the (BYPASSRLS) migration
-- role performs that controlled, parameterized list -- returning ONLY the rows
-- matching the given user_id -- and is the single sanctioned RLS bypass for the
-- session switcher. EXECUTE is granted to the app role; direct table access
-- stays RLS-enforced.

CREATE OR REPLACE FUNCTION list_chat_sessions(p_user_id text)
RETURNS TABLE (id uuid, title text, llm_model text, created_at timestamptz)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT id, title, llm_model, created_at
    FROM chat_sessions
    WHERE user_id = p_user_id
    ORDER BY created_at DESC
    LIMIT 200;
$$;

REVOKE ALL ON FUNCTION list_chat_sessions(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION list_chat_sessions(text) TO agrosat_app;

-- migrate:down
DROP FUNCTION IF EXISTS list_chat_sessions(text);
