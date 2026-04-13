"""
=============================================================================
TIMETABLE GENERATION SYSTEM - Department of Computer Science, University of Yaoundé I
=============================================================================

MATHEMATICAL MODEL
==================

SETS:
  - C  : set of classes (groups of students per level/semester)
  - S  : set of subjects (courses)
  - T  : set of teachers (Course Lecturers)
  - R  : set of rooms (classrooms)
  - D  : set of days     D = {0..5}  (Monday=0 to Saturday=5)
  - P  : set of periods  P = {0..4}
        period 0: 07:00-09:55
        period 1: 10:05-12:55
        period 2: 13:05-15:55
        period 3: 16:05-18:55
        period 4: 19:05-21:55

PARAMETERS:
  - curriculum[c][s]  : 1 if course s belongs to the curriculum of class c, else 0
  - teacher[c][s]     : teacher assigned to course s of class c
  - w[p]              : weight of period p  (w[0]=5, w[1]=4, w[2]=3, w[3]=2, w[4]=1)
                        earlier periods have higher weight => optimizer favors mornings

DECISION VARIABLE:
  x[c, s, r, d, p] in {0, 1}
    = 1  if class c takes course s in room r on day d at period p
    = 0  otherwise

OBJECTIVE FUNCTION:
  Maximize  Z = SUM_{c,s,r,d,p}  w[p] * x[c, s, r, d, p]
  (maximizes sessions before noon since P0 and P1 carry weights 5 and 4)

CONSTRAINTS:
  C1 - No class double-booked at the same slot:
       for all c,d,p :  SUM_{s,r} x[c,s,r,d,p] <= 1

  C2 - Each course of a class scheduled exactly once per week:
       for all c, s in curriculum(c) :  SUM_{r,d,p} x[c,s,r,d,p] = 1

  C3 - Class only takes courses from its curriculum (implicit):
       variables x[c,s,...] created only when s in curriculum(c)

  C4 - No room double-booked at the same slot:
       for all r,d,p :  SUM_{c,s} x[c,s,r,d,p] <= 1

  C5 - No teacher double-booked at the same slot (same semester only):
       Teachers can teach in different semesters (S1 vs S2) concurrently
       since those semesters do not run simultaneously.
       for all t, for all pairs (c1,s1),(c2,s2) where teacher(c1,s1)=teacher(c2,s2)
                  AND same_semester(c1,c2), for all d,p :
           SUM_{r} x[c1,s1,r,d,p] + SUM_{r} x[c2,s2,r,d,p] <= 1

=============================================================================
"""

import json
import os
import sys
from ortools.sat.python import cp_model
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA FROM JSON FILES
# ─────────────────────────────────────────────────────────────────────────────

def find_file(filename):
    """Search for a file next to this script or in a 'data/' subfolder."""
    candidates = [
        filename,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"\n[ERROR] '{filename}' not found.\n"
        f"Place rooms.json and subjects.json in the same folder as this script.\n"
        f"Searched:\n" + "\n".join(f"  {p}" for p in candidates)
    )

print("=" * 65)
print("  TIMETABLE GENERATOR")
print("  Universite de Yaounde I  -  Departement Informatique")
print("=" * 65)

# ── rooms.json ────────────────────────────────────────────────────
rooms_path = find_file("rooms.json")
with open(rooms_path, encoding="utf-8") as f:
    rooms_raw = json.load(f)

rooms_list = rooms_raw.get("Informatique", [])
if not rooms_list:
    sys.exit("[ERROR] Key 'Informatique' not found or empty in rooms.json")

rooms = [r["num"] for r in rooms_list]
room_capacity = {r["num"]: int(r["capacite"]) for r in rooms_list}

print(f"\n[rooms.json]    {len(rooms)} rooms loaded from '{rooms_path}'")
for r in rooms_list:
    print(f"   {r['num']:<8}  capacity={r['capacite']:<5}  building={r['batiment']}")

# ── subjects.json ─────────────────────────────────────────────────
subjects_path = find_file("subjects.json")
with open(subjects_path, encoding="utf-8") as f:
    subjects_raw = json.load(f)

niveaux = subjects_raw.get("niveau", {})
if not niveaux:
    sys.exit("[ERROR] Key 'niveau' not found in subjects.json")

# Build all_classes[class_id] = list of {code, name, teacher}
# Key fix: courses with no lecturer get a UNIQUE teacher ID so they
# never conflict with each other under C5.
all_classes  = {}
unknown_count = 0

for level_key, semesters in niveaux.items():
    for sem_key, sem_data in semesters.items():
        raw_subjects = sem_data.get("subjects", [])
        valid = []
        for subj in raw_subjects:
            code = subj.get("code", "").strip()
            if not code:
                continue

            name = subj.get("name", "")
            if isinstance(name, list):
                name = " ".join(name).strip()
            name = name.strip() or code

            lecturers = subj.get("Course Lecturer", ["", ""])
            teacher_label = "_".join(p.strip() for p in lecturers if p.strip())

            # If no teacher info, assign a unique placeholder so the course
            # never blocks another course under C5
            if not teacher_label:
                unknown_count += 1
                teacher_label = f"__UNKNOWN_{unknown_count}__"

            valid.append({
                "code":    code,
                "name":    name,
                "teacher": teacher_label,
                "credit":  subj.get("credit", 3),
            })

        if valid:
            sem_num = "1" if sem_key.lower() in ("s1",) else "2"
            class_id = f"L{level_key}_S{sem_num}"
            all_classes[class_id] = valid

if not all_classes:
    sys.exit("[ERROR] No valid classes found in subjects.json")

print(f"\n[subjects.json] {len(all_classes)} classes loaded from '{subjects_path}'")
for cid in sorted(all_classes.keys()):
    print(f"   {cid:<10}  {len(all_classes[cid])} courses")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DAYS    = list(range(6))   # 0=Mon .. 5=Sat
PERIODS = list(range(5))   # 0=07h .. 4=19h

PERIOD_NAMES = {
    0: "07:00-09:55",
    1: "10:05-12:55",
    2: "13:05-15:55",
    3: "16:05-18:55",
    4: "19:05-21:55",
}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Higher weight = earlier period => optimizer fills mornings first
WEIGHTS = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1}

# ─────────────────────────────────────────────────────────────────────────────
# 3. INDEX STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

classes = sorted(all_classes.keys())

# subjects_of[class_id] = [code, ...]
subjects_of = {c: [s["code"] for s in all_classes[c]] for c in classes}

# teacher_of[(class_id, code)] = teacher_label
teacher_of = {}
for c in classes:
    for s in all_classes[c]:
        teacher_of[(c, s["code"])] = s["teacher"]

# semester of a class: "S1" or "S2"
def get_semester(class_id):
    return class_id.split("_")[1]   # e.g. "L1_S1" -> "S1"

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD THE CP-SAT MODEL
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-" * 65)
print("  Building CP-SAT model ...")

model = cp_model.CpModel()

# ── Decision variables x[(c, s, r, d, p)] in {0,1} ──────────────
# Variables created ONLY for (c,s) pairs where s is in c's curriculum => C3 implicit
x = {}
for c in classes:
    for s in subjects_of[c]:
        for r in rooms:
            for d in DAYS:
                for p in PERIODS:
                    x[(c, s, r, d, p)] = model.NewBoolVar(f"x|{c}|{s}|{r}|{d}|{p}")

print(f"  Variables created : {len(x):,}")

# ── C1 : A class has at most 1 session per (day, period) ─────────
for c in classes:
    for d in DAYS:
        for p in PERIODS:
            model.Add(
                sum(x[(c, s, r, d, p)]
                    for s in subjects_of[c]
                    for r in rooms) <= 1
            )

# ── C2 : Each course scheduled exactly once per week ─────────────
for c in classes:
    for s in subjects_of[c]:
        model.Add(
            sum(x[(c, s, r, d, p)]
                for r in rooms
                for d in DAYS
                for p in PERIODS) == 1
        )

# ── C4 : Each room used by at most 1 class per (day, period) ─────
for r in rooms:
    for d in DAYS:
        for p in PERIODS:
            model.Add(
                sum(x[(c, s, r, d, p)]
                    for c in classes
                    for s in subjects_of[c]) <= 1
            )

# ── C5 : Teacher conflict — same semester only ───────────────────
# Rationale: S1 and S2 run at different times of the year, so a teacher
# can teach INF121 (L1_S1) and INF122 (L1_S2) without conflict.
# Within the same semester however, a teacher cannot teach two classes
# at the same (day, period).
# Also skip placeholder teachers (__UNKNOWN_N__) since they have no
# real identity — each one represents a distinct, unknown lecturer.

# Group (class, subject) by (teacher, semester)
teacher_sem_slots = defaultdict(list)  # (teacher, semester) -> [(c, s), ...]
for (c, s), teacher in teacher_of.items():
    if teacher.startswith("__UNKNOWN_"):
        continue  # unique placeholder: no conflict possible
    sem = get_semester(c)
    teacher_sem_slots[(teacher, sem)].append((c, s))

c5_constraints = 0
for (teacher, sem), pairs in teacher_sem_slots.items():
    if len(pairs) <= 1:
        continue  # only one session this semester: no conflict possible
    for d in DAYS:
        for p in PERIODS:
            model.Add(
                sum(x[(c, s, r, d, p)]
                    for (c, s) in pairs
                    for r in rooms) <= 1
            )
            c5_constraints += 1

print(f"  Constraints added : C1, C2, C3 (implicit), C4, C5 ({c5_constraints} C5 constraints)")

# ── Objective : Maximize weighted morning sessions ────────────────
model.Maximize(
    sum(
        WEIGHTS[p] * x[(c, s, r, d, p)]
        for c in classes
        for s in subjects_of[c]
        for r in rooms
        for d in DAYS
        for p in PERIODS
    )
)
print("  Objective         : Maximize sum(w[p] * x[c,s,r,d,p])  (morning bias)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SOLVE
# ─────────────────────────────────────────────────────────────────────────────

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 120.0
solver.parameters.log_search_progress = False

print("\n  Solving... (timeout: 120s)\n")
status = solver.Solve(model)

STATUS_NAMES = {
    cp_model.OPTIMAL:    "OPTIMAL",
    cp_model.FEASIBLE:   "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.UNKNOWN:    "UNKNOWN",
}
print(f"  Status          : {STATUS_NAMES.get(status, status)}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f"  Objective value : {solver.ObjectiveValue():.0f}")
    print(f"  Wall time       : {solver.WallTime():.2f}s")

    # Collect solution
    timetable = defaultdict(list)
    for c in classes:
        for s in subjects_of[c]:
            for r in rooms:
                for d in DAYS:
                    for p in PERIODS:
                        if solver.Value(x[(c, s, r, d, p)]) == 1:
                            teacher = teacher_of[(c, s)]
                            # Display UNKNOWN teachers as "TBD"
                            display_teacher = "TBD" if teacher.startswith("__UNKNOWN_") else teacher
                            timetable[c].append({
                                "code":        s,
                                "room":        r,
                                "day":         DAY_NAMES[d],
                                "day_idx":     d,
                                "period":      PERIOD_NAMES[p],
                                "period_idx":  p,
                                "teacher":     display_teacher,
                                "before_noon": p < 2,
                            })

    SEP  = "-" * 80
    SEP2 = "=" * 80

    for c in sorted(timetable.keys()):
        entries = sorted(timetable[c], key=lambda e: (e["day_idx"], e["period_idx"]))
        print(f"\n{SEP2}")
        print(f"  TIMETABLE  -  {c}   ({len(entries)} sessions)")
        print(SEP2)
        print(f"  {'DAY':<12} {'PERIOD':<18} {'COURSE':<12} {'ROOM':<8} {'TEACHER'}")
        print(SEP)
        for e in entries:
            tag = "  [before noon]" if e["before_noon"] else ""
            print(f"  {e['day']:<12} {e['period']:<18} {e['code']:<12} {e['room']:<8} {e['teacher']}{tag}")
        print(SEP)

    # Statistics
    total        = sum(len(v) for v in timetable.values())
    before_noon  = sum(1 for v in timetable.values() for e in v if e["before_noon"])
    pct          = (100 * before_noon // total) if total else 0

    print(f"\n{SEP2}")
    print("  STATISTICS")
    print(SEP2)
    print(f"  Total sessions scheduled : {total}")
    print(f"  Sessions before noon     : {before_noon} / {total}  ({pct}%)")
    print(f"  Classes with timetable   : {len(timetable)}")
    print(f"  Rooms available          : {len(rooms)}")
    print(f"{SEP2}\n")

else:
    print("\n  [!] No feasible solution found.")
    print("  Diagnostics:")
    total_courses = sum(len(v) for v in subjects_of.values())
    max_slots = len(DAYS) * len(PERIODS) * len(rooms)
    print(f"   - Total courses to schedule : {total_courses}")
    print(f"   - Total available slots     : {max_slots}  ({len(DAYS)} days x {len(PERIODS)} periods x {len(rooms)} rooms)")
    if total_courses > max_slots:
        print("   => Not enough slots! Add more rooms or days.")
    print("   - Try increasing max_time_in_seconds\n")
