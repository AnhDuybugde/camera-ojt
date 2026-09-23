-- Additive migration. Does not alter or delete legacy tables.
BEGIN;
CREATE TABLE IF NOT EXISTS public.camera_memberships (
    user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    employee_id text,
    role text NOT NULL CHECK (role IN ('admin','employee')),
    active boolean NOT NULL DEFAULT true,
    updated_at timestamptz NOT NULL DEFAULT now()
);
-- CREATE TABLE IF NOT EXISTS does not evolve an already-deployed membership
-- table. Keep this baseline migration safe for projects created by older builds.
ALTER TABLE public.camera_memberships
    ADD COLUMN IF NOT EXISTS active boolean NOT NULL DEFAULT true;
ALTER TABLE public.camera_memberships
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();
CREATE TABLE IF NOT EXISTS public.camera_events (
    event_id uuid PRIMARY KEY,
    schema_version integer NOT NULL DEFAULT 1,
    employee_id text,
    camera_id text NOT NULL,
    session_id text NOT NULL,
    kind text NOT NULL,
    observed_at timestamptz NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}',
    needs_review boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS camera_events_observed ON public.camera_events(observed_at);
CREATE SEQUENCE IF NOT EXISTS public.camera_changes;
CREATE TABLE IF NOT EXISTS public.camera_records (
    kind text NOT NULL CHECK (kind IN ('employees','work_schedules','attendance','audit_logs','enrollment_samples')),
    record_key text NOT NULL,
    payload jsonb NOT NULL,
    deleted boolean NOT NULL DEFAULT false,
    revision bigint NOT NULL DEFAULT 1,
    change_seq bigint NOT NULL DEFAULT nextval('public.camera_changes'),
    PRIMARY KEY(kind, record_key)
);
CREATE INDEX IF NOT EXISTS camera_records_changes ON public.camera_records(change_seq);
ALTER TABLE public.camera_memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.camera_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.camera_records ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS camera_membership_self ON public.camera_memberships;
CREATE POLICY camera_membership_self ON public.camera_memberships FOR SELECT TO authenticated
    USING(user_id=auth.uid() AND active);
DROP POLICY IF EXISTS camera_events_read ON public.camera_events;
CREATE POLICY camera_events_read ON public.camera_events FOR SELECT TO authenticated USING (
    EXISTS(SELECT 1 FROM public.camera_memberships m WHERE m.user_id=auth.uid()
        AND m.active AND (m.role='admin' OR m.employee_id=camera_events.employee_id))
);
DROP POLICY IF EXISTS camera_records_read ON public.camera_records;
CREATE POLICY camera_records_read ON public.camera_records FOR SELECT TO authenticated USING (
    EXISTS(SELECT 1 FROM public.camera_memberships m WHERE m.user_id=auth.uid()
        AND m.active AND (m.role='admin' OR (kind IN ('employees','work_schedules','attendance')
            AND m.employee_id=payload->>'employee_id')))
);
REVOKE ALL ON public.camera_memberships,public.camera_events,public.camera_records FROM anon;
GRANT SELECT ON public.camera_memberships,public.camera_events,public.camera_records TO authenticated;
GRANT ALL ON public.camera_memberships,public.camera_events,public.camera_records TO service_role;
GRANT USAGE,SELECT ON SEQUENCE public.camera_changes TO service_role;

CREATE OR REPLACE FUNCTION public.camera_record_cas(
    p_kind text, p_key text, p_payload jsonb, p_deleted boolean, p_expected bigint
) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=public,pg_temp AS $$
DECLARE current_row public.camera_records;
BEGIN
    -- A global transaction lock keeps change_seq order equal to commit order,
    -- so incremental clients cannot skip a transaction that commits later.
    PERFORM pg_advisory_xact_lock(782031900);
    SELECT * INTO current_row FROM public.camera_records
        WHERE kind=p_kind AND record_key=p_key FOR UPDATE;
    IF FOUND THEN
        IF current_row.payload=p_payload AND current_row.deleted=p_deleted THEN
            RETURN jsonb_build_object('conflict',false,'record',to_jsonb(current_row));
        END IF;
        IF current_row.revision<>p_expected THEN
            RETURN jsonb_build_object('conflict',true,'record',to_jsonb(current_row));
        END IF;
        UPDATE public.camera_records SET payload=p_payload,deleted=p_deleted,
            revision=revision+1,change_seq=nextval('public.camera_changes')
            WHERE kind=p_kind AND record_key=p_key RETURNING * INTO current_row;
    ELSE
        IF p_expected<>0 THEN
            RAISE EXCEPTION 'Missing base revision';
        END IF;
        INSERT INTO public.camera_records(kind,record_key,payload,deleted)
            VALUES(p_kind,p_key,p_payload,p_deleted) RETURNING * INTO current_row;
    END IF;
    RETURN jsonb_build_object('conflict',false,'record',to_jsonb(current_row));
END $$;
REVOKE ALL ON FUNCTION public.camera_record_cas(text,text,jsonb,boolean,bigint) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.camera_record_cas(text,text,jsonb,boolean,bigint) TO service_role;
CREATE OR REPLACE FUNCTION public.camera_records_batch(items jsonb)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=public,pg_temp AS $$
DECLARE item jsonb; results jsonb := '[]';
BEGIN
    IF jsonb_array_length(items)>100 THEN RAISE EXCEPTION 'Batch exceeds 100 records'; END IF;
    FOR item IN SELECT value FROM jsonb_array_elements(items) LOOP
        results := results || jsonb_build_array(public.camera_record_cas(
            item->>'kind',item->>'key',item->'payload',
            (item->>'deleted')::boolean,(item->>'expected')::bigint));
    END LOOP;
    RETURN results;
END $$;
REVOKE ALL ON FUNCTION public.camera_records_batch(jsonb) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.camera_records_batch(jsonb) TO service_role;
COMMIT;
