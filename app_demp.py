"""
app_demo.py

Gradio web interface for Telco Customer Churn Prediction.
This demo provides an interactive UI for predicting customer churn probability.


"""
import gradio as gr
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.predict import init_artifacts, predict
except ImportError as e:
    print("ERROR: Could not import predict module from src/")
    print(f"Details: {e}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize artifacts at startup
try:
    init_artifacts("config.yaml")
    logger.info(" Successfully initialized prediction artifacts")
except Exception as e:
    logger.error(f" Failed to initialize artifacts: {e}")
    logger.error("  The demo will not work without proper initialization")
    logger.error("Please ensure:")
    logger.error("  1. config.yaml exists in the root directory")
    logger.error("  2. Model artifacts exist at the paths specified in config.yaml")


def predict_churn(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):

    try:
        # Prepare input data dictionary
        input_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }
        
        # Get prediction
        result = predict(input_data)
        
        # Extract results
        churn_risk = result["churn_risk"]
        pred_prob = result["pred_prob"]
        pred_label = result["pred_label"]
        
        # Create detailed message
        risk_message = f"### Churn Risk: **{churn_risk}**"
        confidence_message = f"**Probability of Churn:** {pred_prob:.2%}\n\n"
        
        if churn_risk == "High":
            confidence_message += "⚠️ **Recommendation:** This customer has a high risk of churning. Consider:\n"
            confidence_message += "- Offering retention incentives\n"
            confidence_message += "- Improving customer service touchpoints\n"
            confidence_message += "- Reviewing pricing and contract terms"
        else:
            confidence_message += "✅ **Recommendation:** This customer has a low risk of churning. Consider:\n"
            confidence_message += "- Maintaining current engagement levels\n"
            confidence_message += "- Monitoring for any changes in usage patterns\n"
            confidence_message += "- Exploring upsell opportunities"
        
        return risk_message, confidence_message
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        logger.error(error_msg)
        return f"### Error", f"❌ {error_msg}\n\nPlease check your inputs and try again."


# Define the Gradio interface
def create_demo():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Telco Customer Churn Prediction", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📊 Telco Customer Churn Prediction
            
            This tool predicts the likelihood of a telecom customer churning (leaving the service).
            Fill in the customer information below and click **Predict** to get the churn risk assessment.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 👤 Customer Demographics")
                
                gender = gr.Radio(
                    choices=["Male", "Female"],
                    label="Gender",
                    value="Male"
                )
                
                senior_citizen = gr.Radio(
                    choices=["Yes", "No"],
                    label="Senior Citizen (65+)",
                    value="No"
                )
                
                partner = gr.Radio(
                    choices=["Yes", "No"],
                    label="Has Partner",
                    value="No"
                )
                
                dependents = gr.Radio(
                    choices=["Yes", "No"],
                    label="Has Dependents",
                    value="No"
                )
                
                tenure = gr.Slider(
                    minimum=0,
                    maximum=72,
                    value=1,
                    step=1,
                    label="Tenure (months with company)"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📞 Service Information")
                
                phone_service = gr.Radio(
                    choices=["Yes", "No"],
                    label="Phone Service",
                    value="Yes"
                )
                
                multiple_lines = gr.Radio(
                    choices=["Yes", "No", "No phone service"],
                    label="Multiple Lines",
                    value="No"
                )
                
                internet_service = gr.Radio(
                    choices=["DSL", "Fiber optic", "No"],
                    label="Internet Service",
                    value="DSL"
                )
                
                online_security = gr.Radio(
                    choices=["Yes", "No", "No internet service"],
                    label="Online Security",
                    value="No"
                )
                
                online_backup = gr.Radio(
                    choices=["Yes", "No", "No internet service"],
                    label="Online Backup",
                    value="No"
                )
                
                device_protection = gr.Radio(
                    choices=["Yes", "No", "No internet service"],
                    label="Device Protection",
                    value="No"
                )
                
                tech_support = gr.Radio(
                    choices=["Yes", "No", "No internet service"],
                    label="Tech Support",
                    value="No"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📺 Streaming Services")
                
                streaming_tv = gr.Radio(
                    choices=["Yes", "No", "No internet service"],
                    label="Streaming TV",
                    value="No"
                )
                
                streaming_movies = gr.Radio(
                    choices=["Yes", "No", "No internet service"],
                    label="Streaming Movies",
                    value="No"
                )
                
                gr.Markdown("### 💳 Account Information")
                
                contract = gr.Radio(
                    choices=["Month-to-month", "One year", "Two year"],
                    label="Contract Type",
                    value="Month-to-month"
                )
                
                paperless_billing = gr.Radio(
                    choices=["Yes", "No"],
                    label="Paperless Billing",
                    value="Yes"
                )
                
                payment_method = gr.Dropdown(
                    choices=[
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ],
                    label="Payment Method",
                    value="Electronic check"
                )
                
                monthly_charges = gr.Number(
                    label="Monthly Charges ($)",
                    value=29.85,
                    minimum=0,
                    maximum=200
                )
                
                total_charges = gr.Number(
                    label="Total Charges ($)",
                    value=29.85,
                    minimum=0
                )
        
        with gr.Row():
            predict_btn = gr.Button("🔮 Predict Churn Risk", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                risk_output = gr.Markdown(label="Prediction Result")
                detail_output = gr.Markdown(label="Details")
        
        # Set up the prediction action
        predict_btn.click(
            fn=predict_churn,
            inputs=[
                gender, senior_citizen, partner, dependents, tenure,
                phone_service, multiple_lines, internet_service,
                online_security, online_backup, device_protection,
                tech_support, streaming_tv, streaming_movies,
                contract, paperless_billing, payment_method,
                monthly_charges, total_charges
            ],
            outputs=[risk_output, detail_output]
        )
        
        # Add examples
        gr.Markdown("### 📋 Example Customers")
        gr.Examples(
            examples=[
                # High risk customer
                [
                    "Female", "No", "No", "No", 1,
                    "Yes", "No", "Fiber optic",
                    "No", "No", "No", "No", "No", "No",
                    "Month-to-month", "Yes", "Electronic check",
                    70.70, 70.70
                ],
                # Low risk customer
                [
                    "Male", "No", "Yes", "Yes", 48,
                    "Yes", "Yes", "Fiber optic",
                    "Yes", "Yes", "Yes", "Yes", "Yes", "Yes",
                    "Two year", "No", "Credit card (automatic)",
                    105.50, 5066.40
                ],
                # Medium tenure customer
                [
                    "Female", "Yes", "Yes", "No", 24,
                    "Yes", "Yes", "DSL",
                    "Yes", "No", "Yes", "No", "Yes", "No",
                    "One year", "No", "Bank transfer (automatic)",
                    75.20, 1804.80
                ],
            ],
            inputs=[
                gender, senior_citizen, partner, dependents, tenure,
                phone_service, multiple_lines, internet_service,
                online_security, online_backup, device_protection,
                tech_support, streaming_tv, streaming_movies,
                contract, paperless_billing, payment_method,
                monthly_charges, total_charges
            ],
            label="Click on an example to load it"
        )
        
        gr.Markdown(
            """
            ---
            ### ℹ️ About This Tool
            
            This prediction model analyzes customer data to estimate churn risk. The model considers:
            - **Customer demographics**: Age, family status
            - **Service usage**: Phone, internet, and streaming services
            - **Account details**: Contract type, billing method, charges
            
            **Note:** This is a predictive model and should be used as one input in decision-making, 
            not as the sole determinant of customer retention strategies.
            """
        )
    
    return demo


if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo()
    
    # Launch with sharing disabled by default
    demo.launch(
        server_name="0.0.0.0",  
        server_port=None,        
        share=True,             
        debug=True               
    )