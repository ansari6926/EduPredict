import pandas as pd
import numpy as np

def generate_student_data():
    np.random.seed(42)
    n = 200

    names = [f"Student_{i+1}" for i in range(n)]
    departments = np.random.choice(['CSE', 'ECE', 'MECH', 'CIVIL', 'IT'], n)
    semesters = np.random.choice([1, 2, 3, 4, 5, 6], n)

    attendance = np.random.randint(50, 100, n)
    internal1 = np.random.randint(10, 50, n)
    internal2 = np.random.randint(10, 50, n)
    assignment = np.random.randint(5, 20, n)
    participation = np.random.randint(1, 10, n)

    performance = (
        attendance * 0.3 +
        internal1 * 0.25 +
        internal2 * 0.25 +
        assignment * 0.1 +
        participation * 0.1
    )

    grades = []
    for p in performance:
        if p >= 70: grades.append('A')
        elif p >= 55: grades.append('B')
        elif p >= 40: grades.append('C')
        else: grades.append('F')

    at_risk = ((attendance < 75) | (performance < 40)).astype(int)

    df = pd.DataFrame({
        'Name': names,
        'Department': departments,
        'Semester': semesters,
        'Attendance (%)': attendance,
        'Internal 1': internal1,
        'Internal 2': internal2,
        'Assignment Score': assignment,
        'Participation': participation,
        'Performance Score': performance.round(2),
        'Grade': grades,
        'At Risk': at_risk
    })

    return df