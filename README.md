<div align="center">
  <img src="https://img.icons8.com/color/96/000000/student-center.png" alt="EduPredict Logo"/>
  <h1>🎓 EduPredict</h1>
  <h3>Academic Intelligence & Performance Prediction Platform</h3>

  <p align="center">
    <a href="https://github.com/ansari6926/EduPredict/graphs/contributors"><img src="https://img.shields.io/github/contributors/ansari6926/EduPredict.svg?style=for-the-badge&color=00d4ff" alt="Contributors"></a>
    <a href="https://github.com/ansari6926/EduPredict/network/members"><img src="https://img.shields.io/github/forks/ansari6926/EduPredict.svg?style=for-the-badge&color=6c63ff" alt="Forks"></a>
    <a href="https://github.com/ansari6926/EduPredict/stargazers"><img src="https://img.shields.io/github/stars/ansari6926/EduPredict.svg?style=for-the-badge&color=ffd166" alt="Stars"></a>
    <a href="https://github.com/ansari6926/EduPredict/issues"><img src="https://img.shields.io/github/issues/ansari6926/EduPredict.svg?style=for-the-badge&color=ff6b9d" alt="Issues"></a>
    <a href="https://github.com/ansari6926/EduPredict/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ansari6926/EduPredict.svg?style=for-the-badge&color=06d6a0" alt="License"></a>
  </p>

  <i>Data Science Mini-Project for University Submission • SRM Institute of Science and Technology</i>
</div>

---

## 1. Project Title
**EduPredict: Academic Intelligence Platform**

## 2. Abstract
EduPredict is a sophisticated data science mini-project designed to analyze and predict student academic performance. Utilizing machine learning algorithms, specifically Random Forest Classifiers, this platform processes variables such as attendance, internal assessments, and participation to forecast student grades and identify at-risk individuals. The primary objective is to enable early interventions by educators, fostering an environment where targeted support can be provided. This repository includes a fully functional Streamlit dashboard that visualizes department-wise metrics, attendance progress, and feature importance, bridging the gap between raw educational data and actionable academic insights.

## 3. Problem Statement
Educational institutions collect vast amounts of data regarding student attendance, test scores, and participation. However, this data is rarely utilized proactively to identify students who may be falling behind. The lack of an integrated predictive system often leads to delayed interventions, resulting in higher failure rates and decreased academic success. EduPredict aims to solve this by providing a real-time, data-driven dashboard that accurately predicts student performance and highlights at-risk students using robust machine learning techniques.

## 4. Dataset Source
The dataset used in this project is synthetically generated to mimic real-world academic scenarios. The data generation script `src/preprocessing.py` ensures randomized distribution across 5 departments, 6 semesters, and varying performance metrics. 
* Total Records: 200 Students
* Target Variable: `Grade` (A, B, C, F)
* Features: Attendance, Internal 1, Internal 2, Assignment, Participation.

## 5. Methodology / Workflow
1. **Data Generation & Collection:** Generating synthetic data mimicking academic metrics.
2. **Data Preprocessing:** Cleaning data, handling types, and feature engineering (calculating Total Score).
3. **Exploratory Data Analysis (EDA):** Visualizing data distribution across departments and semesters.
4. **Model Training:** Splitting data into 80/20 train-test sets and fitting a Random Forest Classifier.
5. **Dashboard Deployment:** Building an interactive UI with Streamlit to visualize predictions.

## 6. Tools Used
| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core Programming |
| **Streamlit** | UI/UX and Web Application |
| **Jupyter Notebook** | EDA and Experimentation |
| **Git / GitHub** | Version Control & Hosting |

## 7. Technologies Used
* **Pandas & NumPy:** Data Manipulation
* **Scikit-Learn:** Machine Learning (Random Forest)
* **Matplotlib & Seaborn:** Data Visualization
* **Chart.js:** Frontend interactive charting

## 8. Results / Findings
* The Random Forest model achieved highly accurate grade predictions on unseen test data.
* **Attendance** and **Internal Assessments** were identified as the most crucial predictors of final performance.
* Early risk flagging successfully identifies students with < 75% attendance or < 40% performance scores.

## 9. Screenshots Section
*(Placeholders for actual dashboard screenshots)*
| **Dashboard Overview** | **Prediction Panel** |
|:---:|:---:|
| ![Overview](https://via.placeholder.com/400x250.png?text=Dashboard+Overview) | ![Predict](https://via.placeholder.com/400x250.png?text=Prediction+Panel) |

## 10. Folder Structure
```text
MiniProject_DS_AIML-B_2026_EduPredict/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── abstract.pdf
│   ├── problem_statement.pdf
│   └── presentation.pptx
│
├── dataset/
│   ├── raw_data/
│   └── processed_data/
│
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── preprocessing.ipynb
│   └── visualization.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── analysis.py
│   └── model.py
│
├── outputs/
│   ├── graphs/
│   └── results/
│
└── report/
    └── mini_project_report.pdf
```

## 11. Execution Steps
1. Navigate to the `src` directory and run preprocessing:
   ```bash
   python src/preprocessing.py
   ```
2. Train the machine learning model:
   ```bash
   python src/model.py
   ```
3. Run the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```

## 12. Installation Guide
Clone the repository and install the dependencies:
```bash
git clone https://github.com/ansari6926/EduPredict.git
cd EduPredict
python -m venv .venv
# Activate the virtual environment
source .venv/bin/activate  # On Linux/Mac
.venv\Scripts\activate     # On Windows
pip install -r requirements.txt
```

## 13. Visualizations
The project features several key visualizations:
* **Grade Distribution Doughnut Chart**
* **Department Performance Bar Chart**
* **Risk Overview Panel**
* **Feature Importance Plot**

These graphs are available in the Jupyter notebooks (`notebooks/visualization.ipynb`) and within the live Streamlit dashboard.

## 14. Future Scope
* **Integration with Real LMS Data:** Connect directly to APIs of learning management systems.
* **Deep Learning Models:** Implement Neural Networks for more complex pattern recognition.
* **Automated Email Alerts:** Send automated notifications to at-risk students and mentors.
* **Time-Series Analysis:** Track student performance progress over multiple semesters.

## 15. Conclusion
EduPredict successfully demonstrates the power of machine learning in the educational sector. By converting raw academic data into a predictive platform, it empowers educators to make timely, data-driven decisions that can significantly improve student success rates and overall institutional performance.

## 16. Team Members
* **[Your Name]** - Data Science / AIML-B (2026 Batch)

## 17. GitHub Badges
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 18. License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
