import pandas as pd

def get_basic_stats(df):
    """
    Returns basic statistics of the student dataset.
    """
    stats = {
        'total_students': len(df),
        'avg_attendance': df['Attendance'].mean().round(1),
        'avg_performance': df['Performance'].mean().round(1),
        'at_risk_count': int(df['AtRisk'].sum()),
        'pass_rate': round((df['Grade'] != 'F').mean() * 100, 1)
    }
    return stats

def get_department_stats(df):
    """
    Returns department-wise statistics.
    """
    dept_attendance = df.groupby('Department')['Attendance'].mean().round(1).to_dict()
    dept_atrisk = df[df['AtRisk'] == 1]['Department'].value_counts().to_dict()
    
    dept_grades = {}
    for dept in df['Department'].unique():
        dept_grades[dept] = df[df['Department'] == dept]['Grade'].value_counts().to_dict()
        
    return {
        'attendance': dept_attendance,
        'at_risk': dept_atrisk,
        'grades': dept_grades
    }

if __name__ == "__main__":
    try:
        df = pd.read_csv("../dataset/raw_data/student_data.csv")
        print("Basic Stats:", get_basic_stats(df))
    except FileNotFoundError:
        print("Data not found. Please run preprocessing.py first.")
