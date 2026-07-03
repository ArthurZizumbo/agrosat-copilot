SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: tiger; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA tiger;


--
-- Name: tiger_data; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA tiger_data;


--
-- Name: topology; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA topology;


--
-- Name: SCHEMA topology; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA topology IS 'PostGIS Topology schema';


--
-- Name: fuzzystrmatch; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS fuzzystrmatch WITH SCHEMA public;


--
-- Name: EXTENSION fuzzystrmatch; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION fuzzystrmatch IS 'determine similarities and distance between strings';


--
-- Name: pg_stat_statements; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pg_stat_statements WITH SCHEMA public;


--
-- Name: EXTENSION pg_stat_statements; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION pg_stat_statements IS 'track planning and execution statistics of all SQL statements executed';


--
-- Name: postgis; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;


--
-- Name: EXTENSION postgis; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION postgis IS 'PostGIS geometry and geography spatial types and functions';


--
-- Name: postgis_tiger_geocoder; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder WITH SCHEMA tiger;


--
-- Name: EXTENSION postgis_tiger_geocoder; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION postgis_tiger_geocoder IS 'PostGIS tiger geocoder and reverse geocoder';


--
-- Name: postgis_topology; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis_topology WITH SCHEMA topology;


--
-- Name: EXTENSION postgis_topology; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION postgis_topology IS 'PostGIS topology spatial types and functions';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: list_chat_sessions(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.list_chat_sessions(p_user_id text) RETURNS TABLE(id uuid, title text, llm_model text, created_at timestamp with time zone)
    LANGUAGE sql STABLE SECURITY DEFINER
    SET search_path TO 'public'
    AS $$
    SELECT id, title, llm_model, created_at
    FROM chat_sessions
    WHERE user_id = p_user_id
    ORDER BY created_at DESC
    LIMIT 200;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


--
-- Name: aois; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.aois (
    id bigint NOT NULL,
    session_id uuid NOT NULL,
    geom public.geometry(Polygon,4326) NOT NULL,
    label text,
    area_ha real,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);

ALTER TABLE ONLY public.aois FORCE ROW LEVEL SECURITY;


--
-- Name: aois_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.aois_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: aois_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.aois_id_seq OWNED BY public.aois.id;


--
-- Name: asset_check_executions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.asset_check_executions (
    id bigint NOT NULL,
    asset_key text,
    check_name text,
    partition text,
    run_id character varying(255),
    execution_status character varying(255),
    evaluation_event text,
    evaluation_event_timestamp timestamp without time zone,
    evaluation_event_storage_id bigint,
    materialization_event_storage_id bigint,
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: asset_check_executions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.asset_check_executions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: asset_check_executions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.asset_check_executions_id_seq OWNED BY public.asset_check_executions.id;


--
-- Name: asset_daemon_asset_evaluations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.asset_daemon_asset_evaluations (
    id bigint NOT NULL,
    evaluation_id bigint,
    asset_key text,
    asset_evaluation_body text,
    num_requested integer,
    num_skipped integer,
    num_discarded integer,
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: asset_daemon_asset_evaluations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.asset_daemon_asset_evaluations_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: asset_daemon_asset_evaluations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.asset_daemon_asset_evaluations_id_seq OWNED BY public.asset_daemon_asset_evaluations.id;


--
-- Name: asset_event_tags; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.asset_event_tags (
    id bigint NOT NULL,
    event_id bigint,
    asset_key text NOT NULL,
    key text NOT NULL,
    value text,
    event_timestamp timestamp without time zone
);


--
-- Name: asset_event_tags_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.asset_event_tags_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: asset_event_tags_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.asset_event_tags_id_seq OWNED BY public.asset_event_tags.id;


--
-- Name: asset_keys; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.asset_keys (
    id bigint NOT NULL,
    asset_key character varying(512),
    last_materialization text,
    last_run_id character varying(255),
    asset_details text,
    wipe_timestamp timestamp without time zone,
    last_materialization_timestamp timestamp without time zone,
    tags text,
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    cached_status_data text
);


--
-- Name: asset_keys_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.asset_keys_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: asset_keys_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.asset_keys_id_seq OWNED BY public.asset_keys.id;


--
-- Name: backfill_tags; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.backfill_tags (
    id bigint NOT NULL,
    backfill_id character varying(255),
    key text,
    value text
);


--
-- Name: backfill_tags_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.backfill_tags_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: backfill_tags_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.backfill_tags_id_seq OWNED BY public.backfill_tags.id;


--
-- Name: bulk_actions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bulk_actions (
    id bigint NOT NULL,
    key character varying(32) NOT NULL,
    status character varying(255) NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    body text,
    action_type character varying(32),
    selector_id text,
    job_name text
);


--
-- Name: bulk_actions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.bulk_actions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: bulk_actions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.bulk_actions_id_seq OWNED BY public.bulk_actions.id;


--
-- Name: chat_messages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chat_messages (
    id bigint NOT NULL,
    session_id uuid NOT NULL,
    role text NOT NULL,
    content text NOT NULL,
    extra jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chat_messages_role_check CHECK ((role = ANY (ARRAY['user'::text, 'assistant'::text, 'system'::text])))
);

ALTER TABLE ONLY public.chat_messages FORCE ROW LEVEL SECURITY;


--
-- Name: chat_messages_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.chat_messages_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: chat_messages_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.chat_messages_id_seq OWNED BY public.chat_messages.id;


--
-- Name: chat_sessions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chat_sessions (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id text NOT NULL,
    llm_model text DEFAULT 'gemini'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    title text,
    CONSTRAINT chat_sessions_llm_model_check CHECK ((llm_model = ANY (ARRAY['gemini'::text, 'qwen-api'::text, 'qwen-onprem'::text, 'gemma'::text, 'qwen-vl'::text])))
);

ALTER TABLE ONLY public.chat_sessions FORCE ROW LEVEL SECURITY;


--
-- Name: concurrency_limits; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.concurrency_limits (
    id bigint NOT NULL,
    concurrency_key character varying(512) NOT NULL,
    "limit" integer NOT NULL,
    using_default_limit boolean DEFAULT false NOT NULL,
    update_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: concurrency_limits_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.concurrency_limits_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: concurrency_limits_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.concurrency_limits_id_seq OWNED BY public.concurrency_limits.id;


--
-- Name: concurrency_slots; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.concurrency_slots (
    id bigint NOT NULL,
    concurrency_key text NOT NULL,
    run_id text,
    step_key text,
    deleted boolean NOT NULL,
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: concurrency_slots_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.concurrency_slots_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: concurrency_slots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.concurrency_slots_id_seq OWNED BY public.concurrency_slots.id;


--
-- Name: daemon_heartbeats; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.daemon_heartbeats (
    id bigint NOT NULL,
    daemon_type character varying(255) NOT NULL,
    daemon_id character varying(255),
    "timestamp" timestamp without time zone NOT NULL,
    body text
);


--
-- Name: daemon_heartbeats_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.daemon_heartbeats_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: daemon_heartbeats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.daemon_heartbeats_id_seq OWNED BY public.daemon_heartbeats.id;


--
-- Name: dynamic_partitions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.dynamic_partitions (
    id bigint NOT NULL,
    partitions_def_name text NOT NULL,
    partition text NOT NULL,
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: dynamic_partitions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.dynamic_partitions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: dynamic_partitions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.dynamic_partitions_id_seq OWNED BY public.dynamic_partitions.id;


--
-- Name: event_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.event_logs (
    id bigint NOT NULL,
    run_id character varying(255),
    event text NOT NULL,
    dagster_event_type text,
    "timestamp" timestamp without time zone,
    step_key text,
    asset_key text,
    partition text
);


--
-- Name: event_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.event_logs_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: event_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.event_logs_id_seq OWNED BY public.event_logs.id;


--
-- Name: features_parcels; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.features_parcels (
    id bigint NOT NULL,
    parcel_id bigint NOT NULL,
    year smallint NOT NULL,
    alphaearth_embedding public.vector(64),
    ndvi_stats jsonb DEFAULT '{}'::jsonb NOT NULL,
    phenology jsonb DEFAULT '{}'::jsonb NOT NULL,
    sog_doy smallint,
    peak_doy smallint,
    peak_value real,
    senescence_doy smallint,
    ndvi_auc real,
    ndvi_slope_pre_peak real,
    ndvi_slope_post_peak real,
    maturity_duration_days smallint,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

ALTER TABLE ONLY public.features_parcels FORCE ROW LEVEL SECURITY;


--
-- Name: features_parcels_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.features_parcels_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: features_parcels_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.features_parcels_id_seq OWNED BY public.features_parcels.id;


--
-- Name: instance_info; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.instance_info (
    id bigint NOT NULL,
    run_storage_id text
);


--
-- Name: instance_info_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.instance_info_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: instance_info_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.instance_info_id_seq OWNED BY public.instance_info.id;


--
-- Name: instigators; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.instigators (
    id bigint NOT NULL,
    selector_id character varying(255),
    repository_selector_id character varying(255),
    status character varying(63),
    instigator_type character varying(63),
    instigator_body text,
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: instigators_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.instigators_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: instigators_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.instigators_id_seq OWNED BY public.instigators.id;


--
-- Name: job_ticks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.job_ticks (
    id bigint NOT NULL,
    job_origin_id character varying(255),
    selector_id character varying(255),
    status character varying(63),
    type character varying(63),
    "timestamp" timestamp without time zone,
    tick_body text,
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: job_ticks_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.job_ticks_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: job_ticks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.job_ticks_id_seq OWNED BY public.job_ticks.id;


--
-- Name: jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.jobs (
    id bigint NOT NULL,
    job_origin_id character varying(255),
    selector_id character varying(255),
    repository_origin_id character varying(255),
    status character varying(63),
    job_type character varying(63),
    job_body text,
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: jobs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.jobs_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: jobs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.jobs_id_seq OWNED BY public.jobs.id;


--
-- Name: kvs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.kvs (
    id bigint NOT NULL,
    key text NOT NULL,
    value text
);


--
-- Name: kvs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.kvs_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: kvs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.kvs_id_seq OWNED BY public.kvs.id;


--
-- Name: parcels; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.parcels (
    id bigint NOT NULL,
    session_id uuid,
    aoi_id bigint,
    geom public.geometry(Polygon,4326) NOT NULL,
    crop_class text,
    confidence real,
    area_ha real,
    year smallint NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    canonical_parcel_id text
);

ALTER TABLE ONLY public.parcels FORCE ROW LEVEL SECURITY;


--
-- Name: COLUMN parcels.canonical_parcel_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.parcels.canonical_parcel_id IS 'Canonical PASTIS-R parcel id ("{patch}_{local}") matching the model OOF parquets; NULL for parcels not backed by a PASTIS-R OOF row.';


--
-- Name: parcels_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.parcels_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: parcels_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.parcels_id_seq OWNED BY public.parcels.id;


--
-- Name: pending_steps; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pending_steps (
    id bigint NOT NULL,
    concurrency_key text NOT NULL,
    run_id text,
    step_key text,
    priority integer,
    assigned_timestamp timestamp without time zone,
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: pending_steps_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.pending_steps_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: pending_steps_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.pending_steps_id_seq OWNED BY public.pending_steps.id;


--
-- Name: rag_documents; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.rag_documents (
    id bigint NOT NULL,
    parcel_id text,
    geom public.geometry(Geometry,4326),
    content text NOT NULL,
    source text NOT NULL,
    embedding public.vector(64),
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: rag_documents_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.rag_documents_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: rag_documents_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.rag_documents_id_seq OWNED BY public.rag_documents.id;


--
-- Name: run_tags; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.run_tags (
    id bigint NOT NULL,
    run_id character varying(255),
    key text,
    value text
);


--
-- Name: run_tags_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.run_tags_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: run_tags_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.run_tags_id_seq OWNED BY public.run_tags.id;


--
-- Name: runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.runs (
    id bigint NOT NULL,
    run_id character varying(255),
    snapshot_id character varying(255),
    pipeline_name text,
    mode text,
    status character varying(63),
    run_body text,
    partition text,
    partition_set text,
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    start_time double precision,
    end_time double precision,
    backfill_id character varying(255)
);


--
-- Name: runs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.runs_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: runs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.runs_id_seq OWNED BY public.runs.id;


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version character varying NOT NULL
);


--
-- Name: secondary_indexes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.secondary_indexes (
    id bigint NOT NULL,
    name character varying(512),
    create_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    migration_completed timestamp without time zone
);


--
-- Name: secondary_indexes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.secondary_indexes_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: secondary_indexes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.secondary_indexes_id_seq OWNED BY public.secondary_indexes.id;


--
-- Name: snapshots; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.snapshots (
    id bigint NOT NULL,
    snapshot_id character varying(255) NOT NULL,
    snapshot_body bytea NOT NULL,
    snapshot_type character varying(63) NOT NULL
);


--
-- Name: snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.snapshots_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.snapshots_id_seq OWNED BY public.snapshots.id;


--
-- Name: aois id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.aois ALTER COLUMN id SET DEFAULT nextval('public.aois_id_seq'::regclass);


--
-- Name: asset_check_executions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.asset_check_executions ALTER COLUMN id SET DEFAULT nextval('public.asset_check_executions_id_seq'::regclass);


--
-- Name: asset_daemon_asset_evaluations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.asset_daemon_asset_evaluations ALTER COLUMN id SET DEFAULT nextval('public.asset_daemon_asset_evaluations_id_seq'::regclass);


--
-- Name: asset_event_tags id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.asset_event_tags ALTER COLUMN id SET DEFAULT nextval('public.asset_event_tags_id_seq'::regclass);


--
-- Name: asset_keys id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.asset_keys ALTER COLUMN id SET DEFAULT nextval('public.asset_keys_id_seq'::regclass);


--
-- Name: backfill_tags id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.backfill_tags ALTER COLUMN id SET DEFAULT nextval('public.backfill_tags_id_seq'::regclass);


--
-- Name: bulk_actions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bulk_actions ALTER COLUMN id SET DEFAULT nextval('public.bulk_actions_id_seq'::regclass);


--
-- Name: chat_messages id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_messages ALTER COLUMN id SET DEFAULT nextval('public.chat_messages_id_seq'::regclass);


--
-- Name: concurrency_limits id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.concurrency_limits ALTER COLUMN id SET DEFAULT nextval('public.concurrency_limits_id_seq'::regclass);


--
-- Name: concurrency_slots id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.concurrency_slots ALTER COLUMN id SET DEFAULT nextval('public.concurrency_slots_id_seq'::regclass);


--
-- Name: daemon_heartbeats id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.daemon_heartbeats ALTER COLUMN id SET DEFAULT nextval('public.daemon_heartbeats_id_seq'::regclass);


--
-- Name: dynamic_partitions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dynamic_partitions ALTER COLUMN id SET DEFAULT nextval('public.dynamic_partitions_id_seq'::regclass);


--
-- Name: event_logs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.event_logs ALTER COLUMN id SET DEFAULT nextval('public.event_logs_id_seq'::regclass);


--
-- Name: features_parcels id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.features_parcels ALTER COLUMN id SET DEFAULT nextval('public.features_parcels_id_seq'::regclass);


--
-- Name: instance_info id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.instance_info ALTER COLUMN id SET DEFAULT nextval('public.instance_info_id_seq'::regclass);


--
-- Name: instigators id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.instigators ALTER COLUMN id SET DEFAULT nextval('public.instigators_id_seq'::regclass);


--
-- Name: job_ticks id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_ticks ALTER COLUMN id SET DEFAULT nextval('public.job_ticks_id_seq'::regclass);


--
-- Name: jobs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs ALTER COLUMN id SET DEFAULT nextval('public.jobs_id_seq'::regclass);


--
-- Name: kvs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.kvs ALTER COLUMN id SET DEFAULT nextval('public.kvs_id_seq'::regclass);


--
-- Name: parcels id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.parcels ALTER COLUMN id SET DEFAULT nextval('public.parcels_id_seq'::regclass);


--
-- Name: pending_steps id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pending_steps ALTER COLUMN id SET DEFAULT nextval('public.pending_steps_id_seq'::regclass);


--
-- Name: rag_documents id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rag_documents ALTER COLUMN id SET DEFAULT nextval('public.rag_documents_id_seq'::regclass);


--
-- Name: run_tags id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_tags ALTER COLUMN id SET DEFAULT nextval('public.run_tags_id_seq'::regclass);


--
-- Name: runs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.runs ALTER COLUMN id SET DEFAULT nextval('public.runs_id_seq'::regclass);


--
-- Name: secondary_indexes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.secondary_indexes ALTER COLUMN id SET DEFAULT nextval('public.secondary_indexes_id_seq'::regclass);


--
-- Name: snapshots id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.snapshots ALTER COLUMN id SET DEFAULT nextval('public.snapshots_id_seq'::regclass);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: aois aois_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.aois
    ADD CONSTRAINT aois_pkey PRIMARY KEY (id);


--
-- Name: asset_check_executions asset_check_executions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.asset_check_executions
    ADD CONSTRAINT asset_check_executions_pkey PRIMARY KEY (id);


--
-- Name: asset_daemon_asset_evaluations asset_daemon_asset_evaluations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.asset_daemon_asset_evaluations
    ADD CONSTRAINT asset_daemon_asset_evaluations_pkey PRIMARY KEY (id);


--
-- Name: asset_event_tags asset_event_tags_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.asset_event_tags
    ADD CONSTRAINT asset_event_tags_pkey PRIMARY KEY (id);


--
-- Name: asset_keys asset_keys_asset_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.asset_keys
    ADD CONSTRAINT asset_keys_asset_key_key UNIQUE (asset_key);


--
-- Name: asset_keys asset_keys_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.asset_keys
    ADD CONSTRAINT asset_keys_pkey PRIMARY KEY (id);


--
-- Name: backfill_tags backfill_tags_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.backfill_tags
    ADD CONSTRAINT backfill_tags_pkey PRIMARY KEY (id);


--
-- Name: bulk_actions bulk_actions_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bulk_actions
    ADD CONSTRAINT bulk_actions_key_key UNIQUE (key);


--
-- Name: bulk_actions bulk_actions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bulk_actions
    ADD CONSTRAINT bulk_actions_pkey PRIMARY KEY (id);


--
-- Name: chat_messages chat_messages_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_messages
    ADD CONSTRAINT chat_messages_pkey PRIMARY KEY (id);


--
-- Name: chat_sessions chat_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_sessions
    ADD CONSTRAINT chat_sessions_pkey PRIMARY KEY (id);


--
-- Name: concurrency_limits concurrency_limits_concurrency_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.concurrency_limits
    ADD CONSTRAINT concurrency_limits_concurrency_key_key UNIQUE (concurrency_key);


--
-- Name: concurrency_limits concurrency_limits_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.concurrency_limits
    ADD CONSTRAINT concurrency_limits_pkey PRIMARY KEY (id);


--
-- Name: concurrency_slots concurrency_slots_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.concurrency_slots
    ADD CONSTRAINT concurrency_slots_pkey PRIMARY KEY (id);


--
-- Name: daemon_heartbeats daemon_heartbeats_daemon_type_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.daemon_heartbeats
    ADD CONSTRAINT daemon_heartbeats_daemon_type_key UNIQUE (daemon_type);


--
-- Name: daemon_heartbeats daemon_heartbeats_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.daemon_heartbeats
    ADD CONSTRAINT daemon_heartbeats_pkey PRIMARY KEY (id);


--
-- Name: dynamic_partitions dynamic_partitions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dynamic_partitions
    ADD CONSTRAINT dynamic_partitions_pkey PRIMARY KEY (id);


--
-- Name: event_logs event_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.event_logs
    ADD CONSTRAINT event_logs_pkey PRIMARY KEY (id);


--
-- Name: features_parcels features_parcels_parcel_year_uniq; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.features_parcels
    ADD CONSTRAINT features_parcels_parcel_year_uniq UNIQUE (parcel_id, year);


--
-- Name: features_parcels features_parcels_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.features_parcels
    ADD CONSTRAINT features_parcels_pkey PRIMARY KEY (id);


--
-- Name: instance_info instance_info_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.instance_info
    ADD CONSTRAINT instance_info_pkey PRIMARY KEY (id);


--
-- Name: instigators instigators_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.instigators
    ADD CONSTRAINT instigators_pkey PRIMARY KEY (id);


--
-- Name: instigators instigators_selector_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.instigators
    ADD CONSTRAINT instigators_selector_id_key UNIQUE (selector_id);


--
-- Name: job_ticks job_ticks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_ticks
    ADD CONSTRAINT job_ticks_pkey PRIMARY KEY (id);


--
-- Name: jobs jobs_job_origin_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT jobs_job_origin_id_key UNIQUE (job_origin_id);


--
-- Name: jobs jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT jobs_pkey PRIMARY KEY (id);


--
-- Name: kvs kvs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.kvs
    ADD CONSTRAINT kvs_pkey PRIMARY KEY (id);


--
-- Name: parcels parcels_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.parcels
    ADD CONSTRAINT parcels_pkey PRIMARY KEY (id);


--
-- Name: pending_steps pending_steps_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pending_steps
    ADD CONSTRAINT pending_steps_pkey PRIMARY KEY (id);


--
-- Name: rag_documents rag_documents_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rag_documents
    ADD CONSTRAINT rag_documents_pkey PRIMARY KEY (id);


--
-- Name: run_tags run_tags_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_tags
    ADD CONSTRAINT run_tags_pkey PRIMARY KEY (id);


--
-- Name: runs runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.runs
    ADD CONSTRAINT runs_pkey PRIMARY KEY (id);


--
-- Name: runs runs_run_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.runs
    ADD CONSTRAINT runs_run_id_key UNIQUE (run_id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: secondary_indexes secondary_indexes_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.secondary_indexes
    ADD CONSTRAINT secondary_indexes_name_key UNIQUE (name);


--
-- Name: secondary_indexes secondary_indexes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.secondary_indexes
    ADD CONSTRAINT secondary_indexes_pkey PRIMARY KEY (id);


--
-- Name: snapshots snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.snapshots
    ADD CONSTRAINT snapshots_pkey PRIMARY KEY (id);


--
-- Name: snapshots snapshots_snapshot_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.snapshots
    ADD CONSTRAINT snapshots_snapshot_id_key UNIQUE (snapshot_id);


--
-- Name: aois_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX aois_geom_idx ON public.aois USING gist (geom);


--
-- Name: aois_session_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX aois_session_id_idx ON public.aois USING btree (session_id);


--
-- Name: chat_messages_session_id_created_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX chat_messages_session_id_created_idx ON public.chat_messages USING btree (session_id, created_at);


--
-- Name: chat_sessions_user_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX chat_sessions_user_id_idx ON public.chat_sessions USING btree (user_id);


--
-- Name: features_parcels_parcel_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX features_parcels_parcel_id_idx ON public.features_parcels USING btree (parcel_id);


--
-- Name: features_parcels_year_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX features_parcels_year_idx ON public.features_parcels USING btree (year);


--
-- Name: idx_asset_check_executions; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_asset_check_executions ON public.asset_check_executions USING btree (asset_key, check_name, materialization_event_storage_id, partition);


--
-- Name: idx_asset_check_executions_unique; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_asset_check_executions_unique ON public.asset_check_executions USING btree (asset_key, check_name, run_id, partition);


--
-- Name: idx_asset_daemon_asset_evaluations_asset_key_evaluation_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_asset_daemon_asset_evaluations_asset_key_evaluation_id ON public.asset_daemon_asset_evaluations USING btree (asset_key, evaluation_id);


--
-- Name: idx_asset_event_tags; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_asset_event_tags ON public.asset_event_tags USING btree (asset_key, key, value);


--
-- Name: idx_asset_event_tags_event_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_asset_event_tags_event_id ON public.asset_event_tags USING btree (event_id);


--
-- Name: idx_backfill_tags_backfill_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_backfill_tags_backfill_id ON public.backfill_tags USING btree (backfill_id, id);


--
-- Name: idx_bulk_actions; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_bulk_actions ON public.bulk_actions USING btree (key);


--
-- Name: idx_bulk_actions_action_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_bulk_actions_action_type ON public.bulk_actions USING btree (action_type);


--
-- Name: idx_bulk_actions_selector_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_bulk_actions_selector_id ON public.bulk_actions USING btree (selector_id);


--
-- Name: idx_bulk_actions_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_bulk_actions_status ON public.bulk_actions USING btree (status);


--
-- Name: idx_dynamic_partitions; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_dynamic_partitions ON public.dynamic_partitions USING btree (partitions_def_name, partition);


--
-- Name: idx_event_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_event_type ON public.event_logs USING btree (dagster_event_type, id);


--
-- Name: idx_events_by_asset; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_events_by_asset ON public.event_logs USING btree (asset_key, dagster_event_type, id) WHERE (asset_key IS NOT NULL);


--
-- Name: idx_events_by_asset_partition; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_events_by_asset_partition ON public.event_logs USING btree (asset_key, dagster_event_type, partition, id) WHERE ((asset_key IS NOT NULL) AND (partition IS NOT NULL));


--
-- Name: idx_events_by_run_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_events_by_run_id ON public.event_logs USING btree (run_id, id);


--
-- Name: idx_job_tick_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_job_tick_status ON public.job_ticks USING btree (job_origin_id, status);


--
-- Name: idx_job_tick_timestamp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_job_tick_timestamp ON public.job_ticks USING btree (job_origin_id, "timestamp");


--
-- Name: idx_kvs_keys_unique; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_kvs_keys_unique ON public.kvs USING btree (key);


--
-- Name: idx_pending_steps; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_pending_steps ON public.pending_steps USING btree (concurrency_key, run_id, step_key);


--
-- Name: idx_run_partitions; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_run_partitions ON public.runs USING btree (partition_set, partition);


--
-- Name: idx_run_range; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_run_range ON public.runs USING btree (status, update_timestamp, create_timestamp);


--
-- Name: idx_run_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_run_status ON public.runs USING btree (status);


--
-- Name: idx_run_tags; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_run_tags ON public.run_tags USING btree (key, value);


--
-- Name: idx_run_tags_run_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_run_tags_run_idx ON public.run_tags USING btree (run_id, id);


--
-- Name: idx_runs_by_backfill_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_runs_by_backfill_id ON public.runs USING btree (backfill_id, id);


--
-- Name: idx_runs_by_job; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_runs_by_job ON public.runs USING btree (pipeline_name, id);


--
-- Name: idx_step_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_step_key ON public.event_logs USING btree (step_key);


--
-- Name: idx_tick_selector_timestamp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_tick_selector_timestamp ON public.job_ticks USING btree (selector_id, "timestamp");


--
-- Name: ix_asset_daemon_asset_evaluations_evaluation_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_asset_daemon_asset_evaluations_evaluation_id ON public.asset_daemon_asset_evaluations USING btree (evaluation_id);


--
-- Name: ix_instigators_instigator_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_instigators_instigator_type ON public.instigators USING btree (instigator_type);


--
-- Name: ix_job_ticks_job_origin_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_job_ticks_job_origin_id ON public.job_ticks USING btree (job_origin_id);


--
-- Name: ix_jobs_job_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_jobs_job_type ON public.jobs USING btree (job_type);


--
-- Name: parcels_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX parcels_geom_idx ON public.parcels USING gist (geom);


--
-- Name: parcels_session_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX parcels_session_id_idx ON public.parcels USING btree (session_id);


--
-- Name: parcels_year_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX parcels_year_idx ON public.parcels USING btree (year);


--
-- Name: rag_documents_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX rag_documents_geom_idx ON public.rag_documents USING gist (geom);


--
-- Name: rag_documents_source_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX rag_documents_source_idx ON public.rag_documents USING btree (source);


--
-- Name: aois aois_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.aois
    ADD CONSTRAINT aois_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.chat_sessions(id) ON DELETE CASCADE;


--
-- Name: chat_messages chat_messages_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_messages
    ADD CONSTRAINT chat_messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.chat_sessions(id) ON DELETE CASCADE;


--
-- Name: features_parcels features_parcels_parcel_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.features_parcels
    ADD CONSTRAINT features_parcels_parcel_id_fkey FOREIGN KEY (parcel_id) REFERENCES public.parcels(id) ON DELETE CASCADE;


--
-- Name: runs fk_runs_snapshot_id_snapshots_snapshot_id; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.runs
    ADD CONSTRAINT fk_runs_snapshot_id_snapshots_snapshot_id FOREIGN KEY (snapshot_id) REFERENCES public.snapshots(snapshot_id);


--
-- Name: parcels parcels_aoi_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.parcels
    ADD CONSTRAINT parcels_aoi_id_fkey FOREIGN KEY (aoi_id) REFERENCES public.aois(id) ON DELETE SET NULL;


--
-- Name: parcels parcels_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.parcels
    ADD CONSTRAINT parcels_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.chat_sessions(id) ON DELETE CASCADE;


--
-- Name: run_tags run_tags_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.run_tags
    ADD CONSTRAINT run_tags_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.runs(run_id) ON DELETE CASCADE;


--
-- Name: aois; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.aois ENABLE ROW LEVEL SECURITY;

--
-- Name: chat_messages; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

--
-- Name: chat_sessions; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;

--
-- Name: features_parcels; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.features_parcels ENABLE ROW LEVEL SECURITY;

--
-- Name: parcels; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.parcels ENABLE ROW LEVEL SECURITY;

--
-- Name: aois tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.aois USING ((session_id = (current_setting('app.current_session'::text, true))::uuid)) WITH CHECK ((session_id = (current_setting('app.current_session'::text, true))::uuid));


--
-- Name: chat_messages tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.chat_messages USING ((session_id = (current_setting('app.current_session'::text, true))::uuid)) WITH CHECK ((session_id = (current_setting('app.current_session'::text, true))::uuid));


--
-- Name: chat_sessions tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.chat_sessions USING ((id = (current_setting('app.current_session'::text, true))::uuid)) WITH CHECK ((id = (current_setting('app.current_session'::text, true))::uuid));


--
-- Name: features_parcels tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.features_parcels USING ((EXISTS ( SELECT 1
   FROM public.parcels p
  WHERE ((p.id = features_parcels.parcel_id) AND (p.session_id = (current_setting('app.current_session'::text, true))::uuid))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.parcels p
  WHERE ((p.id = features_parcels.parcel_id) AND (p.session_id = (current_setting('app.current_session'::text, true))::uuid)))));


--
-- Name: parcels tenant_isolation; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY tenant_isolation ON public.parcels USING ((session_id = (current_setting('app.current_session'::text, true))::uuid)) WITH CHECK ((session_id = (current_setting('app.current_session'::text, true))::uuid));


--
-- PostgreSQL database dump complete
--


--
-- Dbmate schema migrations
--

INSERT INTO public.schema_migrations (version) VALUES
    ('20260511213942'),
    ('20260516210000'),
    ('20260516210100'),
    ('20260615082041'),
    ('20260620000418'),
    ('20260620002624'),
    ('20260628120000'),
    ('20260628130000'),
    ('20260628233613'),
    ('20260630120000');
