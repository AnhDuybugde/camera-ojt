"""Load the authoritative, versioned samples without re-running inference."""
from pathlib import Path
import sqlite3

from camera_tracking.face.gallery import EnrolledPerson


def refresh_enrollment(gallery, store, model_version, employee_db):
    if not Path(employee_db).is_file():
        return
    with sqlite3.connect(f"file:{Path(employee_db).resolve()}?mode=ro", uri=True) as conn:
        names = dict(conn.execute("SELECT employee_id, full_name FROM employees"))
    grouped = {}
    for employee, vector, _ in store.samples(model_version):
        if employee in names:
            grouped.setdefault(employee, []).append(vector)
    # The employee database controls existence. Remove deleted employees from
    # the gallery, including cached source-image entries with canonical IDs.
    gallery.people[:] = [p for p in gallery.people if not p.employee_id or p.employee_id in names]
    by_employee = {p.employee_id or p.person_id: p for p in gallery.people}
    for employee, vectors in grouped.items():
        person = by_employee.get(employee)
        if person is None:
            person = EnrolledPerson(employee, names[employee], vectors[0], employee_id=employee)
            gallery.people.append(person)
        person.display_name = names[employee]
        person.embedding, person.prototypes = vectors[0], vectors[1:]
