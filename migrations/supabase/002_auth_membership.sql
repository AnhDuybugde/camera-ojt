-- Supabase Auth account lifecycle and role revocation.
BEGIN;
ALTER TABLE public.camera_memberships
    ADD COLUMN IF NOT EXISTS active boolean NOT NULL DEFAULT true;
ALTER TABLE public.camera_memberships
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();
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
COMMIT;
