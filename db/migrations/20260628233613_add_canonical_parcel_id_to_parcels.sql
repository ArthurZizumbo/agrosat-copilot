-- migrate:up
-- US-079 / Avance 6 copilot: bridge a stored parcel to its canonical PASTIS-R
-- parcel id ("{patch}_{local}") so the OOF-backed tools (the Voting-3 perceiver
-- and compare_models) can resolve a demo parcel to the real fold-5 OOF instead of
-- the numeric cast of parcels.id, which never matches a "patch_local" OOF key.
-- Nullable: parcels not backed by a PASTIS-R OOF row (e.g. a freshly drawn AOI)
-- keep it NULL and the tools degrade honestly as before. Looked up by parcels.id
-- (the primary key), so no extra index is needed.
ALTER TABLE parcels ADD COLUMN IF NOT EXISTS canonical_parcel_id text;

COMMENT ON COLUMN parcels.canonical_parcel_id IS
    'Canonical PASTIS-R parcel id ("{patch}_{local}") matching the model OOF parquets; '
    'NULL for parcels not backed by a PASTIS-R OOF row.';

-- migrate:down
ALTER TABLE parcels DROP COLUMN IF EXISTS canonical_parcel_id;
