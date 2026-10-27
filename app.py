import gradio as gr
import numpy as np
import pandas as pd
import pickle

# 1. Load your pre-trained model and scaler
with open("attrition_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


def predict_attrition(
    age,
    distance_from_home,
    job_satisfaction,
    monthly_income,
    overtime,
    years_at_company,
):
    """Processes user input, applies feature transformation, and returns predictions."""
    # Convert categorical inputs to numeric (adjust binary mapping if your training used different values)
    overtime_val = 1 if overtime == "Yes" else 0

    # Create array of features in the exact order expected by your model/scaler
    raw_features = np.array(
        [
            [
                age,
                distance_from_home,
                job_satisfaction,
                monthly_income,
                overtime_val,
                years_at_company,
            ]
        ]
    )

    # Transform features using scaler if applicable
    try:
        scaled_features = scaler.transform(raw_features)
    except Exception:
        scaled_features = raw_features

    # Predict probabilities
    probabilities = model.predict_proba(scaled_features)[0]
    stay_prob = float(probabilities[0])
    leave_prob = float(probabilities[1])

    # Dynamic status formatting
    if leave_prob > 0.5:
        risk_status = "### 🔴 **High Attrition Risk**"
        rec = "Recommendation: Consider reviewing compensation, workload/overtime balance, or career path progression."
    else:
        risk_status = "### 🟢 **Low Attrition Risk**"
        rec = "Recommendation: Employee engagement parameters are within healthy retention thresholds."

    confidence_dict = {"Stay": stay_prob, "Leave (Attrition Risk)": leave_prob}

    summary_md = (
        f"{risk_status}\n\n"
        f"Probability of Leaving: **{leave_prob:.1%}**\n\n"
        f"*{rec}*"
    )

    return confidence_dict, summary_md


# 2. Build Modern Gradio UI
theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

with gr.Blocks(theme=theme, title="Employee Attrition Predictor") as demo:
    gr.Markdown(
        """
        # 🏢 Employee Attrition Classification System
        Predict employee flight risk and evaluate retention factors using Machine Learning.
        """
    )

    with gr.Row():
        # Left Panel: User Inputs
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Employee Details")
            age = gr.Slider(
                minimum=18, maximum=65, value=30, step=1, label="Age"
            )
            distance_from_home = gr.Slider(
                minimum=1,
                maximum=50,
                value=10,
                step=1,
                label="Distance From Home (km)",
            )
            job_satisfaction = gr.Slider(
                minimum=1,
                maximum=4,
                value=3,
                step=1,
                label="Job Satisfaction Rating (1-4)",
            )
            monthly_income = gr.
