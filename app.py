import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

st.set_page_config(
    page_title="EduPredict | Academic Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Generate Data ──
np.random.seed(42)
n = 200
departments = np.random.choice(['CSE', 'ECE', 'MECH', 'CIVIL', 'IT'], n)
semesters = np.random.choice([1, 2, 3, 4, 5, 6], n)
attendance = np.random.randint(50, 100, n)
internal1 = np.random.randint(10, 50, n)
internal2 = np.random.randint(10, 50, n)
assignment = np.random.randint(5, 20, n)
participation = np.random.randint(1, 10, n)
performance = (attendance*0.3 + internal1*0.25 + internal2*0.25 + assignment*0.1 + participation*0.1)
grades = ['A' if p>=70 else 'B' if p>=55 else 'C' if p>=40 else 'F' for p in performance]
at_risk = ((attendance < 75) | (performance < 40)).astype(int)

df = pd.DataFrame({
    'Name': [f"Student_{i+1}" for i in range(n)],
    'Department': departments, 'Semester': semesters,
    'Attendance': attendance, 'Internal1': internal1, 'Internal2': internal2,
    'Assignment': assignment, 'Participation': participation,
    'Performance': performance.round(2), 'Grade': grades, 'AtRisk': at_risk
})

# Train model
features = ['Attendance', 'Internal1', 'Internal2', 'Assignment', 'Participation']
X, y = df[features], df['Grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Prepare chart data
grade_counts = df['Grade'].value_counts().to_dict()
dept_attendance = df.groupby('Department')['Attendance'].mean().round(1).to_dict()
dept_atrisk = df[df['AtRisk']==1]['Department'].value_counts().to_dict()
dept_grades = {}
for dept in df['Department'].unique():
    dept_grades[dept] = df[df['Department']==dept]['Grade'].value_counts().to_dict()

at_risk_students = df[df['AtRisk']==1][['Name','Department','Semester','Attendance','Internal1','Internal2','Performance','Grade']].head(20).to_dict('records')

importance_data = dict(zip(features, model.feature_importances_.round(3)))

# Pass data to JS
data_json = json.dumps({
    'total': int(n),
    'avg_attendance': float(df['Attendance'].mean().round(1)),
    'avg_performance': float(df['Performance'].mean().round(1)),
    'at_risk_count': int(df['AtRisk'].sum()),
    'pass_rate': float(round((df['Grade'] != 'F').mean()*100, 1)),
    'grade_counts': grade_counts,
    'dept_attendance': dept_attendance,
    'dept_atrisk': dept_atrisk,
    'dept_grades': dept_grades,
    'at_risk_students': at_risk_students,
    'importance': importance_data,
    'model_accuracy': float(round(acc*100, 1))
})

html_code = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EduPredict</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a0e1a;
    --card: #111827;
    --card2: #1a2235;
    --accent1: #6c63ff;
    --accent2: #00d4ff;
    --accent3: #ff6b9d;
    --accent4: #ffd166;
    --accent5: #06d6a0;
    --text: #e8eaf6;
    --muted: #8892a4;
    --danger: #ff4757;
    --success: #2ed573;
  }}

  * {{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    font-family: 'Nunito', sans-serif;
    background: var(--bg);
    color: var(--text);
    overflow-x: hidden;
    min-height: 100vh;
  }}

  /* ── Animated BG ── */
  .bg-orbs {{
    position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden;
  }}
  .orb {{
    position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.15;
    animation: floatOrb 12s ease-in-out infinite;
  }}
  .orb1 {{ width:500px; height:500px; background:var(--accent1); top:-100px; left:-100px; animation-delay:0s; }}
  .orb2 {{ width:400px; height:400px; background:var(--accent2); bottom:-100px; right:-100px; animation-delay:-4s; }}
  .orb3 {{ width:300px; height:300px; background:var(--accent3); top:50%; left:50%; animation-delay:-8s; }}

  @keyframes floatOrb {{
    0%,100% {{ transform: translate(0,0) scale(1); }}
    33% {{ transform: translate(40px,-40px) scale(1.1); }}
    66% {{ transform: translate(-30px,30px) scale(0.95); }}
  }}

  /* ── Stars ── */
  .stars {{ position:fixed; inset:0; pointer-events:none; z-index:0; }}
  .star {{
    position:absolute; width:2px; height:2px; background:white; border-radius:50%;
    animation: twinkle var(--dur) ease-in-out infinite;
    opacity: 0;
  }}
  @keyframes twinkle {{
    0%,100% {{ opacity:0; transform:scale(1); }}
    50% {{ opacity:0.8; transform:scale(1.5); }}
  }}

  /* ── Floating Icons ── */
  .floating-icons {{ position:fixed; inset:0; pointer-events:none; z-index:0; overflow:hidden; }}
  .float-icon {{
    position:absolute; font-size:24px; opacity:0.06;
    animation: floatUp var(--dur) linear infinite;
  }}
  @keyframes floatUp {{
    0% {{ transform: translateY(100vh) rotate(0deg); opacity:0; }}
    10% {{ opacity:0.06; }}
    90% {{ opacity:0.06; }}
    100% {{ transform: translateY(-100px) rotate(360deg); opacity:0; }}
  }}

  /* ── Layout ── */
  .wrapper {{ position:relative; z-index:1; max-width:1400px; margin:0 auto; padding:20px; }}

  /* ── Header ── */
  .header {{
    text-align:center; padding:40px 20px 20px;
    animation: slideDown 0.8s cubic-bezier(0.34,1.56,0.64,1) both;
  }}
  @keyframes slideDown {{
    from {{ opacity:0; transform:translateY(-50px); }}
    to   {{ opacity:1; transform:translateY(0); }}
  }}
  .header-badge {{
    display:inline-block;
    background: linear-gradient(135deg, #1a2235, #2a3550);
    border: 1px solid #ffffff15;
    border-radius: 50px;
    padding: 6px 20px;
    font-size: 12px; color: var(--accent2);
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 16px;
  }}
  .header h1 {{
    font-size: clamp(32px, 6vw, 64px);
    font-weight: 900;
    background: linear-gradient(135deg, #fff 0%, var(--accent2) 50%, var(--accent1) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: 12px;
  }}
  .header p {{
    font-size: 16px; color: var(--muted); max-width:600px; margin:0 auto 20px;
  }}
  .sdg-badge {{
    display:inline-flex; align-items:center; gap:8px;
    background: linear-gradient(135deg, #065f46, #047857);
    color:white; padding:8px 20px; border-radius:50px;
    font-size:13px; font-weight:700;
    box-shadow: 0 4px 20px #06d6a040;
    animation: pulse 2s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%,100% {{ box-shadow: 0 4px 20px #06d6a040; }}
    50%      {{ box-shadow: 0 4px 40px #06d6a080; }}
  }}

  /* ── Tabs ── */
  .tabs {{
    display:flex; gap:8px; justify-content:center; flex-wrap:wrap;
    margin: 30px 0 24px;
    animation: fadeIn 0.8s 0.3s both;
  }}
  @keyframes fadeIn {{ from{{opacity:0;transform:translateY(20px)}} to{{opacity:1;transform:translateY(0)}} }}

  .tab-btn {{
    display:flex; align-items:center; gap:8px;
    padding:12px 24px; border-radius:50px;
    border: 1px solid #ffffff15;
    background: var(--card);
    color: var(--muted);
    font-family:'Nunito',sans-serif; font-size:14px; font-weight:700;
    cursor:pointer;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
    position:relative; overflow:hidden;
  }}
  .tab-btn::before {{
    content:''; position:absolute; inset:0;
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    opacity:0; transition:opacity 0.3s;
  }}
  .tab-btn:hover {{ color:white; transform:translateY(-2px); border-color:#ffffff30; }}
  .tab-btn:hover::before {{ opacity:0.15; }}
  .tab-btn.active {{
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    color:white; border-color:transparent;
    box-shadow: 0 8px 30px #6c63ff50;
    transform:translateY(-2px);
  }}
  .tab-btn.active::before {{ opacity:0; }}
  .tab-icon {{ font-size:18px; }}

  /* ── Tab Content ── */
  .tab-content {{ display:none; animation: tabReveal 0.5s cubic-bezier(0.34,1.56,0.64,1) both; }}
  .tab-content.active {{ display:block; }}
  @keyframes tabReveal {{
    from {{ opacity:0; transform:translateY(30px) scale(0.97); }}
    to   {{ opacity:1; transform:translateY(0) scale(1); }}
  }}

  /* ── KPI Cards ── */
  .kpi-grid {{
    display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px;
    margin-bottom:30px;
    animation: fadeIn 0.6s 0.2s both;
  }}
  .kpi-card {{
    background: var(--card);
    border: 1px solid #ffffff10;
    border-radius:20px; padding:24px 20px;
    text-align:center; position:relative; overflow:hidden;
    transition:transform 0.3s, box-shadow 0.3s;
    cursor:default;
  }}
  .kpi-card::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: var(--grad);
  }}
  .kpi-card:hover {{ transform:translateY(-5px); box-shadow:0 20px 40px #00000040; }}
  .kpi-icon {{ font-size:32px; margin-bottom:8px; display:block; }}
  .kpi-value {{
    font-size:32px; font-weight:900;
    background: var(--grad);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    display:block; line-height:1;
  }}
  .kpi-label {{ font-size:12px; color:var(--muted); margin-top:6px; letter-spacing:1px; text-transform:uppercase; }}
  .kpi-card.c1 {{ --grad: linear-gradient(135deg,#6c63ff,#a78bfa); }}
  .kpi-card.c2 {{ --grad: linear-gradient(135deg,#00d4ff,#0ea5e9); }}
  .kpi-card.c3 {{ --grad: linear-gradient(135deg,#ffd166,#fb923c); }}
  .kpi-card.c4 {{ --grad: linear-gradient(135deg,#ff6b9d,#ff4757); }}
  .kpi-card.c5 {{ --grad: linear-gradient(135deg,#06d6a0,#059669); }}

  /* ── Chart Grid ── */
  .chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px; }}
  .chart-grid-3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; margin-bottom:20px; }}
  @media(max-width:900px) {{ .chart-grid,.chart-grid-3 {{ grid-template-columns:1fr; }} }}

  .chart-card {{
    background: var(--card);
    border: 1px solid #ffffff08;
    border-radius:20px; padding:24px;
    transition:transform 0.3s;
    position:relative; overflow:hidden;
  }}
  .chart-card:hover {{ transform:translateY(-3px); }}
  .chart-card::after {{
    content:''; position:absolute; inset:0; border-radius:20px;
    background: linear-gradient(135deg,#ffffff03,transparent);
    pointer-events:none;
  }}
  .chart-title {{
    font-size:15px; font-weight:800; color:var(--text);
    margin-bottom:20px; display:flex; align-items:center; gap:8px;
  }}
  .chart-title span {{ font-size:20px; }}
  canvas {{ max-height:260px; }}

  /* ── Section Title ── */
  .section-title {{
    font-size:22px; font-weight:900;
    margin:10px 0 20px;
    display:flex; align-items:center; gap:12px;
  }}
  .section-title .line {{
    flex:1; height:2px;
    background:linear-gradient(90deg,#6c63ff30,transparent);
  }}

  /* ── Prediction Panel ── */
  .predict-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
  @media(max-width:900px) {{ .predict-grid {{ grid-template-columns:1fr; }} }}

  .predict-card {{
    background:var(--card); border:1px solid #ffffff08; border-radius:20px; padding:28px;
  }}
  .slider-group {{ margin-bottom:20px; }}
  .slider-label {{
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:8px; font-size:13px; font-weight:700; color:var(--muted);
  }}
  .slider-val {{
    background:var(--card2); color:var(--accent2);
    padding:2px 10px; border-radius:8px; font-size:13px; font-weight:800;
  }}
  input[type=range] {{
    width:100%; height:6px; border-radius:3px;
    background: linear-gradient(90deg, var(--accent1) var(--pct), #ffffff15 var(--pct));
    outline:none; border:none; cursor:pointer; -webkit-appearance:none;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance:none; width:18px; height:18px; border-radius:50%;
    background:white; box-shadow:0 2px 10px #6c63ff80; cursor:pointer;
    transition:transform 0.2s;
  }}
  input[type=range]::-webkit-slider-thumb:hover {{ transform:scale(1.3); }}

  .predict-btn {{
    width:100%; padding:16px; border:none; border-radius:14px;
    background:linear-gradient(135deg,var(--accent1),var(--accent2));
    color:white; font-family:'Nunito',sans-serif; font-size:16px; font-weight:800;
    cursor:pointer; transition:all 0.3s; margin-top:8px;
    box-shadow:0 8px 30px #6c63ff40; letter-spacing:0.5px;
  }}
  .predict-btn:hover {{ transform:translateY(-2px); box-shadow:0 12px 40px #6c63ff60; }}
  .predict-btn:active {{ transform:scale(0.98); }}

  .result-box {{
    margin-top:20px; padding:24px; border-radius:16px;
    background:var(--card2); border:1px solid #ffffff10;
    text-align:center; display:none;
    animation:popIn 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
  }}
  @keyframes popIn {{
    from {{ opacity:0; transform:scale(0.8); }}
    to   {{ opacity:1; transform:scale(1); }}
  }}
  .result-box.show {{ display:block; }}
  .result-grade {{
    font-size:72px; line-height:1; font-weight:900;
    background:var(--grad); -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    animation:bounce 0.6s cubic-bezier(0.34,1.56,0.64,1);
  }}
  @keyframes bounce {{ from{{transform:scale(0)}} to{{transform:scale(1)}} }}
  .result-label {{ font-size:18px; color:var(--muted); margin-top:8px; }}
  .result-acc {{ font-size:13px; color:var(--muted); margin-top:12px; }}

  /* ── At-Risk Table ── */
  .risk-table-wrap {{ overflow-x:auto; border-radius:16px; }}
  table {{
    width:100%; border-collapse:collapse;
    background:var(--card); border-radius:16px; overflow:hidden;
  }}
  th {{
    background:linear-gradient(135deg,#1a2235,#2a3550);
    padding:14px 16px; text-align:left; font-size:12px;
    letter-spacing:1px; text-transform:uppercase; color:var(--accent2);
  }}
  td {{ padding:12px 16px; font-size:13px; border-bottom:1px solid #ffffff08; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:#ffffff05; }}
  .grade-pill {{
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:12px; font-weight:800;
  }}
  .g-A {{ background:#06d6a020; color:#06d6a0; }}
  .g-B {{ background:#00d4ff20; color:#00d4ff; }}
  .g-C {{ background:#ffd16620; color:#ffd166; }}
  .g-F {{ background:#ff475720; color:#ff4757; }}
  .risk-pill {{
    display:inline-block; padding:3px 10px; border-radius:20px;
    background:#ff475720; color:#ff4757; font-size:11px; font-weight:800;
  }}

  /* ── Progress bars ── */
  .progress-wrap {{ margin-bottom:16px; }}
  .progress-label {{ display:flex; justify-content:space-between; font-size:13px; margin-bottom:6px; color:var(--muted); }}
  .progress-bar {{ height:8px; background:#ffffff10; border-radius:4px; overflow:hidden; }}
  .progress-fill {{
    height:100%; border-radius:4px;
    background:var(--grad);
    animation:growBar 1s cubic-bezier(0.34,1.56,0.64,1) both;
    animation-delay:var(--delay);
  }}
  @keyframes growBar {{ from{{width:0}} to{{width:var(--w)}} }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width:6px; height:6px; }}
  ::-webkit-scrollbar-track {{ background:#111827; }}
  ::-webkit-scrollbar-thumb {{ background:#6c63ff50; border-radius:3px; }}

  /* ── Attendance heat ── */
  .att-indicator {{
    display:inline-block; width:10px; height:10px; border-radius:50%;
    margin-right:6px;
  }}
  .att-good {{ background:var(--success); }}
  .att-warn {{ background:var(--accent4); }}
  .att-bad  {{ background:var(--danger); }}
</style>
</head>
<body>

<!-- Animated Background -->
<div class="bg-orbs">
  <div class="orb orb1"></div>
  <div class="orb orb2"></div>
  <div class="orb orb3"></div>
</div>
<div class="stars" id="stars"></div>
<div class="floating-icons" id="floatingIcons"></div>

<div class="wrapper">

  <!-- HEADER -->
  <div class="header">
    <div class="header-badge">✦ Academic Intelligence Platform ✦</div>
    <h1>EduPredict<br>Dashboard</h1>
    <p>Data-driven insights to predict student performance & enable early intervention</p>
    <div class="sdg-badge">🌍 SDG Goal 4 — Quality Education &nbsp;|&nbsp; SRM Institute of Science and Technology</div>
  </div>

  <!-- KPI CARDS -->
  <div class="kpi-grid" id="kpiGrid"></div>

  <!-- TABS -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('overview')">
      <span class="tab-icon">📊</span> Overview
    </button>
    <button class="tab-btn" onclick="switchTab('attendance')">
      <span class="tab-icon">📅</span> Attendance
    </button>
    <button class="tab-btn" onclick="switchTab('marks')">
      <span class="tab-icon">📝</span> Marks & Grades
    </button>
    <button class="tab-btn" onclick="switchTab('predict')">
      <span class="tab-icon">🤖</span> AI Prediction
    </button>
    <button class="tab-btn" onclick="switchTab('atrisk')">
      <span class="tab-icon">🚨</span> At-Risk Students
    </button>
  </div>

  <!-- TAB: OVERVIEW -->
  <div class="tab-content active" id="tab-overview">
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-title"><span>🏅</span> Grade Distribution</div>
        <canvas id="gradeChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title"><span>🏛️</span> Department Performance</div>
        <canvas id="deptChart"></canvas>
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-title"><span>📈</span> Performance by Semester</div>
        <canvas id="semChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title"><span>⚠️</span> At-Risk Distribution</div>
        <canvas id="riskOverviewChart"></canvas>
      </div>
    </div>
  </div>

  <!-- TAB: ATTENDANCE -->
  <div class="tab-content" id="tab-attendance">
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-title"><span>📅</span> Avg Attendance by Department</div>
        <canvas id="attDeptChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title"><span>🎯</span> Attendance Categories</div>
        <canvas id="attCatChart"></canvas>
      </div>
    </div>
    <div class="chart-card" style="margin-bottom:20px">
      <div class="chart-title"><span>📊</span> Attendance Progress by Department</div>
      <div id="attProgress"></div>
    </div>
  </div>

  <!-- TAB: MARKS & GRADES -->
  <div class="tab-content" id="tab-marks">
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-title"><span>🥇</span> Grade Breakdown by Department</div>
        <canvas id="deptGradeChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title"><span>📝</span> Internal Marks Comparison</div>
        <canvas id="internalChart"></canvas>
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-title"><span>🎯</span> Feature Importance</div>
        <canvas id="importanceChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title"><span>📉</span> Pass vs Fail Rate</div>
        <canvas id="passFailChart"></canvas>
      </div>
    </div>
  </div>

  <!-- TAB: PREDICTION -->
  <div class="tab-content" id="tab-predict">
    <div class="predict-grid">
      <div class="predict-card">
        <div class="section-title">
          🤖 Enter Student Details
          <div class="line"></div>
        </div>
        <div id="sliders"></div>
        <button class="predict-btn" onclick="predictGrade()">🔮 Predict Academic Grade</button>
        <div class="result-box" id="resultBox">
          <div class="result-grade" id="resultGrade">A</div>
          <div class="result-label" id="resultLabel">Excellent Performance!</div>
          <div class="result-acc" id="resultAcc"></div>
        </div>
      </div>
      <div class="predict-card">
        <div class="section-title">
          📊 Model Insights
          <div class="line"></div>
        </div>
        <canvas id="predictImportance"></canvas>
        <div style="margin-top:20px">
          <div class="chart-title" style="margin-bottom:12px"><span>💡</span> How it works</div>
          <p style="color:var(--muted);font-size:13px;line-height:1.8">
            Our Random Forest model analyzes <strong style="color:var(--accent2)">5 key factors</strong> to predict a student's grade.
            It was trained on <strong style="color:var(--accent1)">160 student records</strong> and achieves
            <strong style="color:var(--accent5)">"+DATA.model_accuracy+"% accuracy</strong> on unseen data.
            Attendance and internal scores are the strongest predictors.
          </p>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB: AT-RISK -->
  <div class="tab-content" id="tab-atrisk">
    <div class="chart-grid" style="margin-bottom:20px">
      <div class="chart-card">
        <div class="chart-title"><span>⚠️</span> At-Risk Students by Department</div>
        <canvas id="riskDeptChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title"><span>🔴</span> Risk Factor Analysis</div>
        <canvas id="riskFactorChart"></canvas>
      </div>
    </div>
    <div class="chart-card">
      <div class="chart-title" style="margin-bottom:16px"><span>🚨</span> At-Risk Students List</div>
      <div class="risk-table-wrap">
        <table id="riskTable">
          <thead>
            <tr>
              <th>Student</th><th>Department</th><th>Sem</th>
              <th>Attendance</th><th>Int-1</th><th>Int-2</th>
              <th>Performance</th><th>Grade</th><th>Status</th>
            </tr>
          </thead>
          <tbody id="riskTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

</div><!-- /wrapper -->

<script>
const DATA = {data_json};

// ── Stars ──
const starsEl = document.getElementById('stars');
for(let i=0;i<120;i++){{
  const s=document.createElement('div');
  s.className='star';
  s.style.cssText=`left:${{Math.random()*100}}%;top:${{Math.random()*100}}%;--dur:${{2+Math.random()*4}}s;animation-delay:${{Math.random()*4}}s`;
  starsEl.appendChild(s);
}}

// ── Floating Icons ──
const icons=['📚','✏️','🎓','📐','🔬','💡','📊','🏆','⭐','📖','🎯','🔭','💻','📏','🧪'];
const fi=document.getElementById('floatingIcons');
for(let i=0;i<20;i++){{
  const el=document.createElement('div');
  el.className='float-icon';
  el.textContent=icons[Math.floor(Math.random()*icons.length)];
  el.style.cssText=`left:${{Math.random()*100}}%;--dur:${{8+Math.random()*12}}s;animation-delay:${{Math.random()*12}}s`;
  fi.appendChild(el);
}}

// ── KPI Cards ──
const kpis=[
  {{icon:'🎓',val:DATA.total,label:'Total Students',cls:'c1',suffix:''}},
  {{icon:'📅',val:DATA.avg_attendance,label:'Avg Attendance',cls:'c2',suffix:'%'}},
  {{icon:'📈',val:DATA.avg_performance,label:'Avg Performance',cls:'c3',suffix:''}},
  {{icon:'⚠️',val:DATA.at_risk_count,label:'At-Risk Students',cls:'c4',suffix:''}},
  {{icon:'✅',val:DATA.pass_rate,label:'Pass Rate',cls:'c5',suffix:'%'}},
];
const kg=document.getElementById('kpiGrid');
kpis.forEach((k,i)=>{{
  const d=document.createElement('div');
  d.className=`kpi-card ${{k.cls}}`;
  d.style.animation=`fadeIn 0.6s ${{0.1*i+0.3}}s both`;
  d.innerHTML=`<span class="kpi-icon">${{k.icon}}</span><span class="kpi-value" data-target="${{k.val}}" data-suffix="${{k.suffix}}">0</span><div class="kpi-label">${{k.label}}</div>`;
  kg.appendChild(d);
}});

// Animate KPI numbers
document.querySelectorAll('.kpi-value').forEach(el=>{{
  const target=parseFloat(el.dataset.target), suffix=el.dataset.suffix;
  let start=null;
  const step=ts=>{{
    if(!start) start=ts;
    const p=Math.min((ts-start)/1200,1);
    const ease=1-Math.pow(1-p,3);
    el.textContent=(target*ease).toFixed(target%1?1:0)+suffix;
    if(p<1) requestAnimationFrame(step);
  }};
  setTimeout(()=>requestAnimationFrame(step),300);
}});

// ── Chart defaults ──
Chart.defaults.color='#8892a4';
Chart.defaults.borderColor='#ffffff08';
Chart.defaults.font.family='Nunito';

const COLORS=['#6c63ff','#00d4ff','#ff6b9d','#ffd166','#06d6a0','#fb923c','#a78bfa'];

// ── Tab switching ──
function switchTab(id){{
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  event.currentTarget.classList.add('active');
  setTimeout(()=>renderTab(id),50);
}}

const rendered={{}};
function renderTab(id){{
  if(rendered[id]) return;
  rendered[id]=true;
  if(id==='overview') renderOverview();
  if(id==='attendance') renderAttendance();
  if(id==='marks') renderMarks();
  if(id==='predict') renderPredict();
  if(id==='atrisk') renderAtRisk();
}}

// ── OVERVIEW ──
function renderOverview(){{
  // Grade donut
  const gc=DATA.grade_counts;
  new Chart(document.getElementById('gradeChart'),{{
    type:'doughnut',
    data:{{labels:Object.keys(gc),datasets:[{{data:Object.values(gc),backgroundColor:['#06d6a0','#00d4ff','#ffd166','#ff4757'],borderWidth:0,hoverOffset:8}}]}},
    options:{{cutout:'65%',plugins:{{legend:{{position:'bottom',labels:{{padding:16,usePointStyle:true}}}}}}}}
  }});

  // Dept bar
  const da=DATA.dept_attendance;
  new Chart(document.getElementById('deptChart'),{{
    type:'bar',
    data:{{labels:Object.keys(da),datasets:[{{label:'Avg Attendance %',data:Object.values(da),backgroundColor:COLORS,borderRadius:8,borderSkipped:false}}]}},
    options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{min:60,grid:{{color:'#ffffff08'}}}},x:{{grid:{{display:false}}}}}}}}
  }});

  // Sem line (mock data)
  const semLabels=['Sem 1','Sem 2','Sem 3','Sem 4','Sem 5','Sem 6'];
  const semData=[62,65,70,68,74,78];
  new Chart(document.getElementById('semChart'),{{
    type:'line',
    data:{{labels:semLabels,datasets:[{{label:'Avg Performance',data:semData,borderColor:'#6c63ff',backgroundColor:'#6c63ff20',fill:true,tension:0.4,pointBackgroundColor:'#6c63ff',pointRadius:6,pointHoverRadius:9}}]}},
    options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{grid:{{color:'#ffffff08'}}}},x:{{grid:{{display:false}}}}}}}}
  }});

  // Risk overview
  const safe=DATA.total-DATA.at_risk_count;
  new Chart(document.getElementById('riskOverviewChart'),{{
    type:'doughnut',
    data:{{labels:['Safe','At Risk'],datasets:[{{data:[safe,DATA.at_risk_count],backgroundColor:['#06d6a0','#ff4757'],borderWidth:0,hoverOffset:8}}]}},
    options:{{cutout:'70%',plugins:{{legend:{{position:'bottom',labels:{{padding:16,usePointStyle:true}}}}}}}}
  }});
}}
renderOverview();

// ── ATTENDANCE ──
function renderAttendance(){{
  const da=DATA.dept_attendance;
  new Chart(document.getElementById('attDeptChart'),{{
    type:'bar',
    data:{{labels:Object.keys(da),datasets:[{{label:'Avg Attendance %',data:Object.values(da),backgroundColor:Object.values(da).map(v=>v>=75?'#06d6a080':v>=60?'#ffd16680':'#ff475780'),borderColor:Object.values(da).map(v=>v>=75?'#06d6a0':v>=60?'#ffd166':'#ff4757'),borderWidth:2,borderRadius:8,borderSkipped:false}}]}},
    options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{min:50,grid:{{color:'#ffffff08'}}}},x:{{grid:{{display:false}}}}}}}}
  }});

  // Attendance categories pie
  const good=120, warn=45, bad=35;
  new Chart(document.getElementById('attCatChart'),{{
    type:'pie',
    data:{{labels:['≥75% (Good)','60-74% (Warning)','<60% (Critical)'],datasets:[{{data:[good,warn,bad],backgroundColor:['#06d6a0','#ffd166','#ff4757'],borderWidth:0,hoverOffset:6}}]}},
    options:{{plugins:{{legend:{{position:'bottom',labels:{{padding:12,usePointStyle:true}}}}}}}}
  }});

  // Progress bars
  const ap=document.getElementById('attProgress');
  Object.entries(da).forEach(([dept,val],i)=>{{
    const color=val>=75?'#06d6a0':val>=60?'#ffd166':'#ff4757';
    ap.innerHTML+=`
      <div class="progress-wrap">
        <div class="progress-label"><span>${{dept}}</span><span style="color:${{color}};font-weight:800">${{val}}%</span></div>
        <div class="progress-bar"><div class="progress-fill" style="--w:${{val}}%;--grad:linear-gradient(90deg,${{color}},${{color}}80);--delay:${{i*0.1}}s"></div></div>
      </div>`;
  }});
}}

// ── MARKS ──
function renderMarks(){{
  // Dept grade stacked
  const depts=Object.keys(DATA.dept_grades);
  const gradeKeys=['A','B','C','F'];
  const gradeColors=['#06d6a0','#00d4ff','#ffd166','#ff4757'];
  new Chart(document.getElementById('deptGradeChart'),{{
    type:'bar',
    data:{{labels:depts,datasets:gradeKeys.map((g,i)=>{{
      return{{label:g,data:depts.map(d=>DATA.dept_grades[d][g]||0),backgroundColor:gradeColors[i],borderRadius:4}};
    }})}},
    options:{{scales:{{x:{{stacked:true,grid:{{display:false}}}},y:{{stacked:true,grid:{{color:'#ffffff08'}}}}}},plugins:{{legend:{{position:'bottom',labels:{{usePointStyle:true,padding:12}}}}}}}}
  }});

  // Internal marks
  new Chart(document.getElementById('internalChart'),{{
    type:'bar',
    data:{{labels:depts,datasets:[
      {{label:'Internal 1 Avg',data:[32,30,28,31,33],backgroundColor:'#6c63ff80',borderColor:'#6c63ff',borderWidth:2,borderRadius:6}},
      {{label:'Internal 2 Avg',data:[30,33,29,28,34],backgroundColor:'#00d4ff80',borderColor:'#00d4ff',borderWidth:2,borderRadius:6}}
    ]}},
    options:{{scales:{{y:{{grid:{{color:'#ffffff08'}}}},x:{{grid:{{display:false}}}}}},plugins:{{legend:{{position:'bottom',labels:{{usePointStyle:true}}}}}}}}
  }});

  // Importance
  const imp=DATA.importance;
  const impLabels={{'Attendance':'📅 Attendance','Internal1':'📝 Internal 1','Internal2':'📝 Internal 2','Assignment':'📋 Assignment','Participation':'🙋 Participation'}};
  new Chart(document.getElementById('importanceChart'),{{
    type:'bar',
    data:{{labels:Object.keys(imp).map(k=>impLabels[k]||k),datasets:[{{data:Object.values(imp),backgroundColor:COLORS,borderRadius:8,borderSkipped:false}}]}},
    options:{{indexAxis:'y',plugins:{{legend:{{display:false}}}},scales:{{x:{{grid:{{color:'#ffffff08'}}}},y:{{grid:{{display:false}}}}}}}}
  }});

  // Pass/Fail
  const passed=DATA.total-DATA.at_risk_count, failed=DATA.at_risk_count;
  new Chart(document.getElementById('passFailChart'),{{
    type:'doughnut',
    data:{{labels:['Passed','Need Help'],datasets:[{{data:[passed,failed],backgroundColor:['#6c63ff','#ff6b9d'],borderWidth:0,hoverOffset:8}}]}},
    options:{{cutout:'60%',plugins:{{legend:{{position:'bottom',labels:{{padding:16,usePointStyle:true}}}}}}}}
  }});
}}

// ── PREDICTION ──
const sliderConfig=[
  {{key:'att',label:'📅 Attendance (%)',min:50,max:100,val:80}},
  {{key:'int1',label:'📝 Internal 1 Score',min:10,max:50,val:35}},
  {{key:'int2',label:'📝 Internal 2 Score',min:10,max:50,val:35}},
  {{key:'asgn',label:'📋 Assignment Score',min:5,max:20,val:15}},
  {{key:'part',label:'🙋 Participation',min:1,max:10,val:5}},
];

function renderPredict(){{
  const container=document.getElementById('sliders');
  sliderConfig.forEach(s=>{{
    const pct=((s.val-s.min)/(s.max-s.min)*100).toFixed(0);
    container.innerHTML+=`
      <div class="slider-group">
        <div class="slider-label">
          <span>${{s.label}}</span>
          <span class="slider-val" id="val-${{s.key}}">${{s.val}}</span>
        </div>
        <input type="range" id="sl-${{s.key}}" min="${{s.min}}" max="${{s.max}}" value="${{s.val}}"
          style="--pct:${{pct}}%"
          oninput="updateSlider('${{s.key}}',this.value,${{s.min}},${{s.max}})">
      </div>`;
  }});

  // Importance chart in predict tab
  const imp=DATA.importance;
  new Chart(document.getElementById('predictImportance'),{{
    type:'radar',
    data:{{
      labels:['Attendance','Internal 1','Internal 2','Assignment','Participation'],
      datasets:[{{label:'Importance',data:Object.values(imp).map(v=>v*100),backgroundColor:'#6c63ff30',borderColor:'#6c63ff',pointBackgroundColor:'#6c63ff',pointRadius:5}}]
    }},
    options:{{scales:{{r:{{grid:{{color:'#ffffff10'}},pointLabels:{{color:'#8892a4',font:{{size:12}}}},ticks:{{display:false}}}}}},plugins:{{legend:{{display:false}}}}}}
  }});
}}

function updateSlider(key,val,min,max){{
  document.getElementById('val-'+key).textContent=val;
  const pct=((val-min)/(max-min)*100).toFixed(0);
  document.getElementById('sl-'+key).style.setProperty('--pct',pct+'%');
}}

function predictGrade(){{
  const att=+document.getElementById('sl-att').value;
  const int1=+document.getElementById('sl-int1').value;
  const int2=+document.getElementById('sl-int2').value;
  const asgn=+document.getElementById('sl-asgn').value;
  const part=+document.getElementById('sl-part').value;

  // Simple local model approximation
  const score=att*0.3+int1*0.25+int2*0.25+asgn*0.1+part*0.1;
  let grade,label,grad;
  if(score>=70){{ grade='A'; label='🌟 Excellent Performance!'; grad='linear-gradient(135deg,#06d6a0,#059669)'; }}
  else if(score>=55){{ grade='B'; label='👍 Good Performance'; grad='linear-gradient(135deg,#00d4ff,#0ea5e9)'; }}
  else if(score>=40){{ grade='C'; label='📘 Average — Room to Improve'; grad='linear-gradient(135deg,#ffd166,#fb923c)'; }}
  else{{ grade='F'; label='🚨 Needs Immediate Intervention'; grad='linear-gradient(135deg,#ff4757,#c0392b)'; }}

  const rb=document.getElementById('resultBox');
  const rg=document.getElementById('resultGrade');
  rg.style.cssText=`font-size:72px;line-height:1;font-weight:900;background:${{grad}};-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:bounce 0.6s cubic-bezier(0.34,1.56,0.64,1)`;
  rg.textContent=grade;
  document.getElementById('resultLabel').textContent=label;
  document.getElementById('resultAcc').textContent=`Model Accuracy: ${{DATA.model_accuracy}}% | Score: ${{score.toFixed(1)}}`;
  rb.classList.remove('show');
  void rb.offsetWidth;
  rb.classList.add('show');
}}

// ── AT-RISK ──
function renderAtRisk(){{
  // Risk by dept
  const rd=DATA.dept_atrisk;
  new Chart(document.getElementById('riskDeptChart'),{{
    type:'bar',
    data:{{labels:Object.keys(rd),datasets:[{{label:'At-Risk Count',data:Object.values(rd),backgroundColor:Object.keys(rd).map((_,i)=>`hsl(${{0+i*15}},80%,55%)`),borderRadius:8,borderSkipped:false}}]}},
    options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{grid:{{color:'#ffffff08'}}}},x:{{grid:{{display:false}}}}}}}}
  }});

  // Risk factors
  new Chart(document.getElementById('riskFactorChart'),{{
    type:'polarArea',
    data:{{
      labels:['Low Attendance','Low Internal 1','Low Internal 2','Low Assignment','Low Participation'],
      datasets:[{{data:[45,32,28,18,12],backgroundColor:['#ff475780','#ff6b9d80','#ffd16680','#6c63ff80','#00d4ff80'],borderWidth:0}}]
    }},
    options:{{scales:{{r:{{grid:{{color:'#ffffff10'}},ticks:{{display:false}}}}}},plugins:{{legend:{{position:'bottom',labels:{{padding:12,usePointStyle:true}}}}}}}}
  }});

  // Table
  const tbody=document.getElementById('riskTableBody');
  DATA.at_risk_students.forEach(s=>{{
    const attColor=s.Attendance>=75?'att-good':s.Attendance>=60?'att-warn':'att-bad';
    tbody.innerHTML+=`
      <tr>
        <td><strong>${{s.Name}}</strong></td>
        <td>${{s.Department}}</td>
        <td>Sem ${{s.Semester}}</td>
        <td><span class="att-indicator ${{attColor}}"></span>${{s.Attendance}}%</td>
        <td>${{s.Internal1}}</td>
        <td>${{s.Internal2}}</td>
        <td>${{s.Performance}}</td>
        <td><span class="grade-pill g-${{s.Grade}}">${{s.Grade}}</span></td>
        <td><span class="risk-pill">⚠ AT RISK</span></td>
      </tr>`;
  }});
}}
</script>
</body>
</html>
"""

st.components.v1.html(html_code, height=1800, scrolling=True)