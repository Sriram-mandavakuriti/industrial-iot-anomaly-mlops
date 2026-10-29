import streamlit as st
import requests

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Industrial IoT Monitoring",
    page_icon="🏭",
    layout="wide"
)

# --------------------------------
# Header
# --------------------------------
st.title("🏭 Industrial IoT Anomaly Detection")
st.markdown(
    "### AI-powered Machine Condition Monitoring Dashboard"
)

st.divider()

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.header("⚙️ Machine Sensor Input")

air_temperature = st.sidebar.number_input(
    "Air Temperature [K]",
    value=298.0
)

process_temperature = st.sidebar.number_input(
    "Process Temperature [K]",
    value=309.0
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed [rpm]",
    value=1500.0
)

torque = st.sidebar.number_input(
    "Torque [Nm]",
    value=40.0
)

tool_wear = st.sidebar.number_input(
    "Tool Wear [min]",
    value=100.0
)

# --------------------------------
# Prediction Button
# --------------------------------
if st.sidebar.button(
    "🔍 Detect Machine Anomaly",
    use_container_width=True
):

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            params={
                "air_temperature": air_temperature,
                "process_temperature": process_temperature,
                "rotational_speed": rotational_speed,
                "torque": torque,
                "tool_wear": tool_wear
            }
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            anomaly_score = result["anomaly_score"]

            # --------------------------------
            # Machine Status
            # --------------------------------
            if prediction == "ANOMALY":

                st.error("🚨 MACHINE ANOMALY DETECTED")

            else:

                st.success("✅ MACHINE CONDITION NORMAL")

            st.divider()

            # --------------------------------
            # KPI Cards
            # --------------------------------
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Machine Status",
                    prediction
                )

            with col2:
                st.metric(
                    "Anomaly Score",
                    anomaly_score
                )

            with col3:
                st.metric(
                    "Rotational Speed",
                    f"{rotational_speed:.0f} rpm"
                )

            st.divider()

            # --------------------------------
            # Sensor Information
            # --------------------------------
            st.subheader("📊 Current Sensor Readings")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "Air Temp",
                    f"{air_temperature:.1f} K"
                )

            with col2:
                st.metric(
                    "Process Temp",
                    f"{process_temperature:.1f} K"
                )

            with col3:
                st.metric(
                    "RPM",
                    f"{rotational_speed:.0f}"
                )

            with col4:
                st.metric(
                    "Torque",
                    f"{torque:.1f} Nm"
                )

            with col5:
                st.metric(
                    "Tool Wear",
                    f"{tool_wear:.0f} min"
                )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to FastAPI backend. "
            "Please start the API first."
        )

else:

    st.info(
        "Enter the machine sensor values from the sidebar "
        "and click **Detect Machine Anomaly**."
    )

# --------------------------------
# Project Information
# --------------------------------
st.divider()

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Algorithm**")
    st.write("Isolation Forest")

with col2:
    st.write("**Learning Type**")
    st.write("Unsupervised Learning")

with col3:
    st.write("**Application**")
    st.write("Industrial Predictive Maintenance")