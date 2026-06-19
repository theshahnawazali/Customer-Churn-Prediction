import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Machine Learning-Based Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_PATH = Path(__file__).resolve().parents[1]

st.title("Machine Learning-Based Customer Churn Prediction")
st.divider()
df = pd.read_csv(BASE_PATH / "data" / "Churn-Prediction-dataset.csv")


st.markdown("""
<style>
.kpi-card{
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px 16px;
    height: 80px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 0 8px rgba(0,0,0,0.25);
}

.kpi-title{
    font-size: 12px;
    color: #cbd5e1;
    margin-bottom: 6px;
}

.kpi-value{
    font-size: 28px;
    font-weight: 700;
}

.icon{
    font-size: 28px;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

cards = [
    ("Total Customers", "7,043", "#3b82f6", "👥"),
    ("Churned Customers", "1,869", "#ef4444", "👤"),
    ("Churn Rate", "26.54%", "#ef4444", "◔"),
    ("Active Customers", "5,174", "#22c55e", "👤"),
    ("Avg. Tenure (Months)", "32.37", "#a855f7", "📅"),
    ("Avg. Monthly Charges", "$64.76", "#3b82f6", "💲"),
]

cols = st.columns(len(cards))

for col, (title, value, color, icon) in zip(cols, cards):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value" style="color:{color}">
                        {value}
                    </div>
                </div>
                <div class="icon" style="color:{color}">
                    {icon}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
col1, col2, col3= st.columns([1,1,1])

with col1:
    fig = px.pie(
        df,
        names="Churn",
        hole=0.35,
        color="Churn",
        color_discrete_map={
            "Yes": "#EF553B",
            "No": "#2f6df6"
        }
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="#04142d",
        plot_bgcolor="#04142d",
        font_color="white",
        legend=dict(
            x=0.80,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        title={
            "text" : "Churn Distribution",
            "x" : 0.05
        },
        margin=dict(
            l=50,
            r=50,
            t=40,
            b=10
        )
    )
    st.plotly_chart(fig, width="stretch")


with col2:
    fig = px.bar(
        df,
        y=["Contract","Churn"],
        x=df["Churn"],
        barmode="group",
        color="Churn",
        color_discrete_map={
            "Yes": "#EF553B",
            "No": "#2f6df6"
        }
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="#04142d",
        plot_bgcolor="#04142d",
        font_color="white",
        legend=dict(
            x=0.85,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        title={
            "text" : "Contact Type VS Churn",
            "x" : 0.05
        },
        margin=dict(
            l=50,
            r=50,
            t=40,
            b=10
        )
    )

    st.plotly_chart(fig,width="stretch", key="fig1")


with col3:
    fig = px.bar(
        df,
        x=df["Churn"],
        y=["gender","Churn"],
        barmode="group",
        color="Churn",
        color_discrete_map={
            "Yes": "#EF553B",
            "No": "#2f6df6"
        }
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="#04142d",
        plot_bgcolor="#04142d",
        font_color="white",
        legend=dict(
            x=0.85,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        title={
            "text" : "Gender VS Churn",
            "x" : 0.05
        },
        margin=dict(
            l=50,
            r=50,
            t=40,
            b=10
        )
    )

    st.plotly_chart(fig,width="stretch", key="fig3")


st.divider()

# =================== Model Evalution Graphs ========================

evaluation = pd.DataFrame({
    "Models": [
        "Logistic Regression",
        "KNN (Tuned)",
        "Random Forest (Tuned)",
        "Decision Tree (Tuned)",
        "KNN (Base)",
        "Random Forest (Base)",
        "Decision Tree (Base)"
    ],
    "Accuracy": [0.8219, 0.8141, 0.8148, 0.8020, 0.7764, 0.7949, 0.7182],
    "Precision": [0.6860, 0.6637, 0.7000, 0.6382, 0.5863, 0.6544, 0.4672],
    "Recall": [0.6032, 0.6032, 0.5255, 0.5818, 0.5282, 0.4772, 0.4584],
    "F1 Score": [0.6419, 0.6320, 0.6003, 0.6087, 0.5557, 0.5519, 0.4628],
    "ROC-AUC": [0.7519, 0.7466, 0.7222, 0.7315, 0.6970, 0.6932, 0.6351]
})

model_fig = px.bar(
    evaluation,
    x="Models",
    y=["Accuracy",'Precision','Recall','F1 Score','ROC-AUC'],
    barmode="group",
    labels={
        "Models" : "Machine Leaning Models",
        "value" : "Score",
        "variable" : "Matrics"
    },
    color="variable",
    color_discrete_map={
    "Accuracy": "#2f6df6",   # Plotly Blue
    "Precision": "#EF553B",  # Plotly Orange/Red
    "Recall": "#00CC96",     # Plotly Green
    "F1 Score": "#AB63FA",   # Plotly Purple
    "ROC-AUC": "#FFA15A"     # Plotly Orange
}
)
model_fig.update_layout(
    height=350,
    paper_bgcolor="#04142d",
    plot_bgcolor="#04142d",
    font_color="white",
    legend=dict(
        x=1,
        y=1,
        xanchor="left",
        yanchor="top"
    ),
    title={
        "text" : "Model Evaluation Matrics",
        "x" : 0.05
    },
    margin=dict(
        l=50,
        r=50,
        t=40,
        b=10
    )
)

st.plotly_chart(model_fig, width="stretch")

st.info("Note: Percentage may not be 100 percent due to rounding.")

st.divider()

dataset, try_now = st.columns([1,1])

with dataset:
    if st.button("View Dataset"):
        st.switch_page("pages/datasetpage.py")


with try_now:
    if st.button("Try Now"):
        st.switch_page("pages/trynow.py")