import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data import generate_student_data

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="EduPredict Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ─── LOAD DATA ───
df = generate_student_data()

# ─── HEADER ───
st.markdown("""
    <h1 style='text-align:center; color:#1f77b4;'>🎓 EduPredict Dashboard</h1>
    <h4 style='text-align:center; color:gray;'>Data-Driven Prediction of Student Academic Performance</h4>
    <p style='text-align:center; color:green;'>🌍 SDG Goal 4: Quality Education</p>
    <hr>
""", unsafe_allow_html=True)

# ─── SIDEBAR FILTERS ───
st.sidebar.header("🔍 Filter Students")
dept = st.sidebar.multiselect("Department", options=df['Department'].unique(), default=df['Department'].unique())
sem  = st.sidebar.multiselect("Semester",   options=sorted(df['Semester'].unique()), default=df['Semester'].unique())

filtered = df[df['Department'].isin(dept) & df['Semester'].isin(sem)]

# ─── KPI CARDS ───
st.subheader("📊 Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students",     len(filtered))
col2.metric("Avg Attendance",     f"{filtered['Attendance (%)'].mean():.1f}%")
col3.metric("Avg Performance",    f"{filtered['Performance Score'].mean():.1f}")
col4.metric("At-Risk Students",   int(filtered['At Risk'].sum()))

st.markdown("---")

# ─── ROW 1: Attendance + Grade Distribution ───
col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Attendance Analysis")
    fig1 = px.histogram(filtered, x='Attendance (%)', nbins=20,
                        color_discrete_sequence=['#1f77b4'],
                        title="Attendance Distribution")
    fig1.add_vline(x=75, line_dash="dash", line_color="red",
                   annotation_text="75% threshold")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("🏅 Grade Distribution")
    grade_counts = filtered['Grade'].value_counts().reset_index()
    grade_counts.columns = ['Grade', 'Count']
    fig2 = px.pie(grade_counts, names='Grade', values='Count',
                  color_discrete_sequence=px.colors.qualitative.Set2,
                  title="Grade Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# ─── ROW 2: Internal Marks + At-Risk by Dept ───
col3, col4 = st.columns(2)

with col3:
    st.subheader("📝 Internal Marks Overview")
    fig3 = px.box(filtered, x='Department', y='Internal 1',
                  color='Department',
                  title="Internal 1 Marks by Department")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("⚠️ At-Risk Students by Department")
    at_risk_dept = filtered[filtered['At Risk'] == 1]['Department'].value_counts().reset_index()
    at_risk_dept.columns = ['Department', 'At Risk Count']
    fig4 = px.bar(at_risk_dept, x='Department', y='At Risk Count',
                  color='Department',
                  title="At-Risk Students per Department")
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ─── PERFORMANCE PREDICTION ───
st.subheader("🤖 Student Performance Prediction")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Enter Student Details")
    att  = st.slider("Attendance (%)",   50, 100, 80)
    int1 = st.slider("Internal 1 Score", 10, 50,  35)
    int2 = st.slider("Internal 2 Score", 10, 50,  35)
    asgn = st.slider("Assignment Score",  5, 20,  15)
    part = st.slider("Participation",     1, 10,   5)

    features = ['Attendance (%)', 'Internal 1', 'Internal 2', 'Assignment Score', 'Participation']
    X = df[features]
    y = df['Grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    if st.button("🔮 Predict Grade"):
        input_data = pd.DataFrame([[att, int1, int2, asgn, part]], columns=features)
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        grade_colors = {'A': '🟢', 'B': '🔵', 'C': '🟡', 'F': '🔴'}
        st.success(f"Predicted Grade: {grade_colors.get(prediction, '')} **{prediction}**")
        st.info(f"Model Accuracy: **{acc*100:.1f}%**")

with col_b:
    st.markdown("#### Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    fig5 = px.bar(importance_df, x='Importance', y='Feature',
                  orientation='h', color='Importance',
                  color_continuous_scale='blues',
                  title="Which Factors Matter Most?")
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ─── AT-RISK STUDENTS TABLE ───
st.subheader("🚨 At-Risk Students List")
at_risk_df = filtered[filtered['At Risk'] == 1][['Name', 'Department', 'Semester',
                                                   'Attendance (%)', 'Internal 1',
                                                   'Internal 2', 'Performance Score', 'Grade']]
st.dataframe(at_risk_df.reset_index(drop=True), use_container_width=True)

# ─── FOOTER ───
st.markdown("""
    <hr>
    <p style='text-align:center; color:gray;'>
    EduPredict | SRM Institute of Science and Technology |
    SDG Goal 4: Quality Education 🌍
    </p>
""", unsafe_allow_html=True)