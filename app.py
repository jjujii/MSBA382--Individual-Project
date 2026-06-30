import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION & AUTHENTICATION
# ============================================================================
st.set_page_config(
    page_title="Stroke Risk Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Refined, elegant CSS design
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        color: #1a202c;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: #fafbfc;
    }
    
    .main-header {
        font-size: 1.4em;
        font-weight: 600;
        text-align: center;
        padding: 6px 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 6px;
        margin-bottom: 8px;
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }
    .main-header span {
        font-size: 0.65em;
        font-weight: 400;
        display: block;
        margin-top: 1px;
    }

    .filter-box {
        background: #dcd0ff;
        padding: 8px;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #dda0dd;
        font-weight: 600;
        color: #4b0082;
    }

    .calc-box {
        background: #dcd0ff;
        padding: 8px;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #dda0dd;
        font-weight: 600;
        color: #4b0082;
    }

    .spacer {
        margin-bottom: 1px;
    }
    
    .risk-high {
        background: white;
        padding: 16px;
        border-radius: 6px;
        border-left: 4px solid #dc2626;
        border-top: 3px solid #dc2626;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.1);
        margin-top: 10px;
    }
    
    .risk-high h3 {
        color: #991b1b;
        font-size: 1.4em;
        margin-bottom: 6px;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    
    .risk-high p {
        color: #7f1d1d;
        font-size: 1em;
        font-weight: 500;
        line-height: 1.6;
    }
    
    .risk-medium {
        background: white;
        padding: 16px;
        border-radius: 6px;
        border-left: 4px solid #f59e0b;
        border-top: 3px solid #f59e0b;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
        margin-top: 10px;
    }
    
    .risk-medium h3 {
        color: #92400e;
        font-size: 1.4em;
        margin-bottom: 6px;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    
    .risk-medium p {
        color: #78350f;
        font-size: 1em;
        font-weight: 500;
        line-height: 1.6;
    }
    
    .risk-low {
        background: white;
        padding: 16px;
        border-radius: 6px;
        border-left: 4px solid #16a34a;
        border-top: 3px solid #16a34a;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.1);
        margin-top: 10px;
    }
    
    .risk-low h3 {
        color: #15803d;
        font-size: 1.4em;
        margin-bottom: 6px;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    
    .risk-low p {
        color: #166534;
        font-size: 1em;
        font-weight: 500;
        line-height: 1.6;
    }
    
    .prediction-section {
        background: white;
        padding: 32px;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        border-top: 3px solid #667eea;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 24px;
    }
    
    .section-title {
        font-size: 1.2em;
        color: #1a202c;
        font-weight: 600;
        margin-bottom: 8px;
        letter-spacing: -0.3px;
    }
    
    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    .right-side {
        background-color:#f4f0ec;       /* clean white background */
        border-left: 3px solid #d1d5db;  /* light grey border separating from left side */
        padding: 20px;
        border-radius: 2;                /* keep edges square for column look */
    }
        
    .left-side {
    background-color: #f9fafb;       /* very light gray */
    padding: 20px;
    border-right: 3px solid #d1d5db; /* soft grey vertical divider */
    }
            
    [data-testid="stMetricContainer"] {
        background: white;
        padding: 20px !important;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85em;
        color: #718096;
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    
    [data-testid="stMetricValue"] {
        color: #667eea;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE CONFIGURATION & AUTHENTICATION
# ============================================================================
st.set_page_config(
    page_title="Gender Differences in Stroke Prevalence and Risk Factors.",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ============================================================================
# AUTHENTICATION
# ============================================================================
def check_password():
    """Simple password check."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.markdown("""
        <div style='text-align: center; padding: 100px 20px;'>
            <h1 style='color: #667eea; font-size: 2.2em; margin-bottom: 8px; font-weight: 300; letter-spacing: -1px;'>Stroke Risk Analytics</h1>
            <p style='color: #718096; font-size: 1em; font-weight: 400; letter-spacing: 0.2px;'>Advanced Health Intelligence Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        password = st.text_input("Enter Password", type="password", placeholder="")
        
        if password == "stroke2026":
            st.session_state.password_correct = True
            st.rerun()
        elif password:
            st.error("Incorrect password. Please try again.")
            return False
        else:
            st.info("Enter your password to continue: stroke2026")
            return False
    
    return True

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    """Load cleaned stroke dataset."""
    df = pd.read_csv('stroke_data_cleaned (1).csv')
    return df

# ============================================================================
# TRAIN STROKE PREDICTION MODEL
# ============================================================================
@st.cache_resource
def train_prediction_model(df):
    """Train a Random Forest classifier for stroke prediction."""
    feature_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                    'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
                    'work_type_Self-employed', 'work_type_children',
                    'smoking_status_Unknown', 'smoking_status_formerly smoked',
                    'smoking_status_never smoked', 'smoking_status_smokes',
                    'BMI_category_Underweight', 'BMI_category_Normal',
                    'BMI_category_Overweight', 'BMI_category_Obese']
    
    X = df[feature_cols].fillna(0)
    y = df['stroke']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_scaled, y)
    
    return model, scaler, feature_cols

# ============================================================================
# GLOBAL FILTERS
# ============================================================================
def apply_global_filters(df, age_range, gender_selected, work_type_selected):
    """Apply global filters to dataset."""
    filtered_df = df.copy()
    
    filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
    
    if gender_selected:
        gender_codes = [0 if g == 'Male' else 1 for g in gender_selected]
        filtered_df = filtered_df[filtered_df['gender'].isin(gender_codes)]
    
    if work_type_selected:
        work_type_mask = False
        for work_type in work_type_selected:
            col_name = f'work_type_{work_type}'
            work_type_mask = work_type_mask | (filtered_df[col_name] == True)
        filtered_df = filtered_df[work_type_mask]
    
    return filtered_df

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_overview_metrics(df):
    """Display key metrics with refined styling."""
    col1, col2, col3, col4, col5 = st.columns(5, gap="small")
    
    total = len(df)
    strokes = df['stroke'].sum()
    males = (df['gender'] == 0).sum()
    females = (df['gender'] == 1).sum()
    avg_age = df['age'].mean()
    
    with col1:
        st.metric("Patients", f"{total:,}", "")
    with col2:
        st.metric("Strokes", f"{strokes}", f"{(strokes/total)*100:.1f}%" if total > 0 else "0%")
    with col3:
        st.metric("Males", f"{males:,}", "")
    with col4:
        st.metric("Females", f"{females:,}", "")
    with col5:
        st.metric("Avg Age", f"{avg_age:.0f}", "")

def plot_gender_stroke_rates(df):
    """Gender-specific stroke rates with refined design."""
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return None
    
    stroke_by_gender = df.groupby('gender').agg({
        'stroke': ['sum', 'count']
    }).reset_index()
    stroke_by_gender.columns = ['gender', 'stroke_cases', 'total']
    stroke_by_gender['rate'] = (stroke_by_gender['stroke_cases'] / stroke_by_gender['total'] * 100)
    stroke_by_gender['gender_name'] = stroke_by_gender['gender'].map({0: 'Male', 1: 'Female'})
    
    fig = go.Figure()
    
    colors = {0: '#667eea', 1: '#ffb6c1 '}
    
    for _, row in stroke_by_gender.iterrows():
        fig.add_trace(go.Bar(
            x=[row['gender_name']],
            y=[row['rate']],
            marker=dict(color=colors[row['gender']], line=dict(color='rgba(255,255,255,0)', width=0)),
            text=[f"{row['rate']:.1f}%"],
            textposition='outside',
            textfont=dict(size=12, color='#1a202c', family='Arial'),
            hovertemplate=f"<b>{row['gender_name']}</b><br>Rate: {row['rate']:.2f}%<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        title={'text': 'Stroke Risk by Gender', 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 16, 'color': '#1a202c', 'family': 'Arial'}},
        xaxis_title='',
        yaxis_title='Stroke Rate (%)',
        template='plotly_white',
        height=360,
        margin=dict(l=40, r=15, t=70, b=15),
        hovermode='x unified',
        font=dict(color='#2d3748', size=11, family='Arial'),
        plot_bgcolor='#fafbfc',
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#e2e8f0', zeroline=False),
        xaxis=dict(showgrid=False),
        paper_bgcolor='white'
    )
    return fig

def plot_age_gender(df, gender_filter='All Genders'):
    """Age distribution with optional gender filter."""
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return None
    
    # Apply gender filter
    if gender_filter == 'Male':
        df = df[df['gender'] == 0]
    elif gender_filter == 'Female':
        df = df[df['gender'] == 1]
    
    if len(df) == 0:
        st.warning("No data available for selected gender filter")
        return None
    
    stroke_by_age = df.groupby('age_group').agg({
        'stroke': ['sum', 'count']
    }).reset_index()
    stroke_by_age.columns = ['age_group', 'stroke_cases', 'total']
    stroke_by_age['rate'] = (stroke_by_age['stroke_cases'] / stroke_by_age['total'] * 100)
    
    age_order = ['Child', 'Young Adult', 'Middle Age', 'Senior']
    stroke_by_age['age_group'] = pd.Categorical(stroke_by_age['age_group'], categories=age_order, ordered=True)
    stroke_by_age = stroke_by_age.sort_values('age_group')

    
    # Choose color based on gender filter
    if gender_filter == 'Male':
        bar_color = '#667eea'
    elif gender_filter == 'Female':
        bar_color = '#764ba2'
    else:
        bar_color = '#764ba2'
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stroke_by_age['age_group'],
        y=stroke_by_age['rate'],
        marker=dict(color=bar_color, line=dict(color='rgba(255,255,255,0)', width=0)),
        text=[f"{val:.1f}%" for val in stroke_by_age['rate']],
        textposition='outside',
        textfont=dict(size=12, color='#1a202c', family='Arial'),
        hovertemplate='<b>%{x}</b><br>Rate: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': 'Stroke Risk by Age Group', 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 16, 'color': '#1a202c', 'family': 'Arial'}},
        xaxis_title='',
        yaxis_title='Stroke Rate (%)',
        template='plotly_white',
        height=340,
        margin=dict(l=40, r=15, t=70, b=15),
        showlegend=False,
        font=dict(color='#2d3748', size=11, family='Arial'),
        plot_bgcolor='#fafbfc',
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#e2e8f0', zeroline=False),
        xaxis=dict(showgrid=False),
        paper_bgcolor='white'
    )

    return fig


def plot_glucose_htn(df):
    """Glucose and HTN interaction."""
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Males", "Females"),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]],
        shared_yaxes=True
    )
    
    for col_idx, (gender_val, gender_name) in enumerate([(0, 'Male'), (1, 'Female')], 1):
        gender_df = df[df['gender'] == gender_val]
        
        for hyper_val, hyper_name in [(0, 'No HTN'), (1, 'With HTN')]:
            hyper_df = gender_df[gender_df['hypertension'] == hyper_val].copy()
            
            if len(hyper_df) == 0:
                continue
            
            glucose_bins = [0, 100, 125, 200, 500]
            glucose_labels = ['Normal', 'Prediab', 'Diabetic', 'V.High']
            hyper_df['glucose_cat'] = pd.cut(hyper_df['avg_glucose_level'], bins=glucose_bins, labels=glucose_labels, right=False)
            
            glucose_stroke = hyper_df.groupby('glucose_cat').agg({
                'stroke': ['sum', 'count']
            }).reset_index()
            glucose_stroke.columns = ['glucose_cat', 'stroke_cases', 'total']
            glucose_stroke['rate'] = (glucose_stroke['stroke_cases'] / glucose_stroke['total'] * 100)
            
            color = '#764ba2' if hyper_val == 1 else '#667eea'
            
            fig.add_trace(go.Bar(
                x=glucose_stroke['glucose_cat'],
                y=glucose_stroke['rate'],
                name=hyper_name,
                marker=dict(color=color),
                text=[f"{val:.0f}%" for val in glucose_stroke['rate']],
                textposition='outside',
                textfont=dict(size=8, color='#1a202c'),
                hovertemplate='%{x}: %{y:.0f}%<extra></extra>',
                showlegend=(col_idx == 1)
            ), row=1, col=col_idx)
    
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="Rate (%)", row=1, col=1, matches='y')
    fig.update_yaxes(title_text="", row=1, col=2)

    
    fig.update_layout(
        title={'text': 'Glucose x Hypertension Interaction', 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 16, 'color': '#1a202c', 'family': 'Arial'}},
        height=390,
        template='plotly_white',
        margin=dict(l=40, r=15, t=50, b=15),
        barmode='group',
        legend=dict(font=dict(size=8)),
        font=dict(color='#2d3748', size=10, family='Arial'),
        plot_bgcolor='#fafbfc',
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#e2e8f0', zeroline=False),
        paper_bgcolor='white'
    )
    return fig

def plot_smoking_bmi(df):
    """Smoking and BMI."""
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Smoking Status", "BMI Category"),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]],
        shared_yaxes=True
    )
    
    # Smoking
    smoking_data = []
    for gender_val, gender_name in [(0, 'Male'), (1, 'Female')]:
        gender_df = df[df['gender'] == gender_val]
        smoking_categories = {
            'Never': 'smoking_status_never smoked',
            'Former': 'smoking_status_formerly smoked',
            'Current': 'smoking_status_smokes'
        }
        
        for status, col in smoking_categories.items():
            status_df = gender_df[gender_df[col] == True]
            if len(status_df) > 0:
                smoking_data.append({
                    'Gender': gender_name,
                    'Status': status,
                    'Rate': status_df['stroke'].mean() * 100
                })
    
    if smoking_data:
        smoking_df = pd.DataFrame(smoking_data)
        for status in ['Never', 'Former', 'Current']:
            data = smoking_df[smoking_df['Status'] == status]
            if len(data) > 0:
                color = '#667eea' if status == 'Never' else '#764ba2' if status == 'Former' else '#a78bfa'
                fig.add_trace(go.Bar(
                    x=data['Gender'],
                    y=data['Rate'],
                    name=status,
                    marker=dict(color=color),
                    text=[f"{val:.0f}%" for val in data['Rate']],
                    textposition='outside',
                    textfont=dict(size=8, color='#1a202c'),
                    showlegend=True
                ), row=1, col=1)
    
    # BMI
    bmi_data = []
    for gender_val, gender_name in [(0, 'Male'), (1, 'Female')]:
        gender_df = df[df['gender'] == gender_val]
        
        for bmi_cat in ['BMI_category_Normal', 'BMI_category_Overweight', 'BMI_category_Obese']:
            bmi_df = gender_df[gender_df[bmi_cat] == True]
            if len(bmi_df) > 0:
                bmi_name = bmi_cat.replace('BMI_category_', '')
                bmi_data.append({
                    'Gender': gender_name,
                    'BMI': bmi_name,
                    'Rate': bmi_df['stroke'].mean() * 100
                })
    
    if bmi_data:
        bmi_df_plot = pd.DataFrame(bmi_data)
        for bmi in ['Normal', 'Overweight', 'Obese']:
            data = bmi_df_plot[bmi_df_plot['BMI'] == bmi]
            if len(data) > 0:
                color = '#4682b4 ' if bmi == 'Normal' else '#000080' if bmi == 'Overweight' else '#483d8b'
                fig.add_trace(go.Bar(
                    x=data['Gender'],
                    y=data['Rate'],
                    name=bmi,
                    marker=dict(color=color),
                    text=[f"{val:.0f}%" for val in data['Rate']],
                    textposition='outside',
                    textfont=dict(size=8, color='#1a202c'),
                    showlegend=True
                ), row=1, col=2)
    
    fig.update_yaxes(title_text="Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="Rate (%)", row=1, col=1, matches='y')
    fig.update_yaxes(title_text="", row=1, col=2)
    
    fig.update_layout(
        title={'text': 'Smoking & BMI Impact', 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 16, 'color': '#1a202c', 'family': 'Arial'}},
        height=340,
        template='plotly_white',
        margin=dict(l=40, r=15, t=50, b=15),
        legend=dict(font=dict(size=8)),
        barmode='group',
        hovermode='x unified',
        font=dict(color='#2d3748', size=10, family='Arial'),
        plot_bgcolor='#fafbfc',
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#e2e8f0', zeroline=False),
        paper_bgcolor='white'
    )
    return fig

# ============================================================================
# STROKE RISK PREDICTION TOOL
# ============================================================================

def predict_stroke_risk(model, scaler, feature_cols, patient_data):
    """Predict stroke risk for a patient."""
    input_array = np.zeros(len(feature_cols))
    
    for i, col in enumerate(feature_cols):
        if col in patient_data:
            input_array[i] = patient_data[col]
    
    input_scaled = scaler.transform([input_array])[0]
    risk_probability = model.predict_proba([input_scaled])[0][1]
    
    return risk_probability

def stroke_risk_calculator(df):
    """Interactive stroke risk calculator."""
    st.markdown('<div class="calc-box">Personal Stroke Risk Prediction</div>', unsafe_allow_html=True)
    model, scaler, feature_cols = train_prediction_model(df)
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=0, max_value=100, value=45)
        hypertension = st.checkbox("Hypertension", value=False)
        heart_disease = st.checkbox("Heart Disease", value=False)
    
    with col2:
        avg_glucose = st.slider("Average Glucose Level", min_value=50, max_value=300, value=120)
        bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    
    # Prepare patient data
    patient_data = {
        'age': age,
        'hypertension': 1 if hypertension else 0,
        'heart_disease': 1 if heart_disease else 0,
        'avg_glucose_level': avg_glucose,
        'bmi': bmi,
        'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
        'work_type_Never_worked': 1 if work_type == 'Never_worked' else 0,
        'work_type_Private': 1 if work_type == 'Private' else 0,
        'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
        'work_type_children': 1 if work_type == 'children' else 0,
        'smoking_status_Unknown': 1 if smoking == 'Unknown' else 0,
        'smoking_status_formerly smoked': 1 if smoking == 'formerly smoked' else 0,
        'smoking_status_never smoked': 1 if smoking == 'never smoked' else 0,
        'smoking_status_smokes': 1 if smoking == 'smokes' else 0,
        'BMI_category_Underweight': 1 if bmi < 18.5 else 0,
        'BMI_category_Normal': 1 if 18.5 <= bmi < 25 else 0,
        'BMI_category_Overweight': 1 if 25 <= bmi < 30 else 0,
        'BMI_category_Obese': 1 if bmi >= 30 else 0,
    }

    # Predict
    risk = predict_stroke_risk(model, scaler, feature_cols, patient_data)

    # Display result
    if risk < 0.15:
        st.markdown(f'<div class="risk-low"><h3>Low Risk</h3><p>Stroke Risk: <strong>{risk*100:.1f}%</strong></p><p style="margin-top: 12px; font-size: 0.9em;">You are in a favorable health range. Continue maintaining healthy habits.</p></div>', unsafe_allow_html=True)
    elif risk < 0.35:
        st.markdown(f'<div class="risk-medium"><h3>Medium Risk</h3><p>Stroke Risk: <strong>{risk*100:.1f}%</strong></p><p style="margin-top: 12px; font-size: 0.9em;">Consider consulting with a healthcare provider about lifestyle modifications.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-high"><h3>High Risk</h3><p>Stroke Risk: <strong>{risk*100:.1f}%</strong></p><p style="margin-top: 12px; font-size: 0.9em;">Please consult with a healthcare professional immediately for personalized advice.</p></div>', unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================
if check_password():
    # Refined header
    st.markdown(
        '<div class="main-header">Gender Differences in Stroke Prevalence and Risk Factors.'
        '<br><span>MSBA382 Healthcare Analytics</span></div>',
        unsafe_allow_html=True
    )

    # Load data
    df_original = load_data()

    # Two-column split: left = filters + charts, right = calculator
    left_col, right_col = st.columns([2,1], gap="large")

    with left_col:
        # Global filters (moved left side only)
        filter_col1, filter_col2 = st.columns(2, gap="small")
        with filter_col1:
            age_range = st.slider(
                "Age Range",
                min_value=int(df_original['age'].min()),
                max_value=int(df_original['age'].max()),
                value=(int(df_original['age'].min()), int(df_original['age'].max())),
                step=1
            )
        with filter_col2:
            work_type_selected = st.multiselect(
                "Work Type",
                options=['Private','Self-employed','Govt_job','children','Never_worked'],
                default=['Private','Self-employed','Govt_job','children','Never_worked']
            )

        # Apply filters (gender set to None)
        df_filtered = apply_global_filters(df_original, age_range, None, work_type_selected)

        # Overview metrics
        plot_overview_metrics(df_filtered)
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # Charts row 1
        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
            fig = plot_gender_stroke_rates(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        with col2:
            age_gender_filter = st.radio("View by:", options=['All Genders', 'Male', 'Female'], horizontal=True, key='age_gender_filter', label_visibility="collapsed")
            fig = plot_age_gender(df_filtered, gender_filter=age_gender_filter)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Charts row 2
        col3, col4 = st.columns(2, gap="small")
        with col3:
            fig = plot_glucose_htn(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        with col4:
            fig = plot_smoking_bmi(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with right_col:
        stroke_risk_calculator(df_original)
        st.markdown('</div>', unsafe_allow_html=True)


