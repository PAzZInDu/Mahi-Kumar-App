import pickle

import pandas as pd
import streamlit as st



st.set_page_config(
    page_title="Numerical Clinical Inputs",
    page_icon="📊",
    layout="centered",
)


if "patient_records" not in st.session_state:
    st.session_state.patient_records = []



NUMERIC_IMPUTER = "numeric_imputer.pkl"
NUMERIC_SCALER = "numeric_scaler.pkl"


@st.cache_resource
def load_model(model_path):
    """Load and cache a serialized model or transformer."""
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


HISTORY_FIELDS = {
    "study_eye_dmetrtfocallashx": "Previous focal-laser treatment for DME",
    "study_eye_dmetrtivthx": "Previous intravitreal treatment for DME",
    "study_eye_dmetrtvegfhx": "Previous anti-VEGF treatment for DME",
    "study_eye_prphx": "Previous panretinal photocoagulation",
    "study_eye_cataractexthx": "Previous cataract extraction",
    "study_eye_vitrectomyhx": "Previous vitrectomy",
    "study_eye_glaucsurghx": "Previous glaucoma surgery",
}


def history_input(column_name: str, label: str) -> float:
    """Collect a study-eye history value in the source dataset's representation."""
    answer = st.selectbox(
        label,
        options=["No / not recorded", "Yes"],
        key=column_name,
        help=(
            f"Output column: {column_name}. Yes is stored as 1; "
            "No/not recorded is stored as 0."
        ),
    )
    return 1.0 if answer == "Yes" else 0.0


st.title("Numerical Clinical Inputs")
st.caption(
    "Enter raw baseline values. Scaling should be performed later by the fitted "
    "preprocessor."
)

with st.form("numerical_patient_inputs"):
    st.subheader("Patient information")
    age_column, diabetes_age_column = st.columns(2)
    age_at_enrollment = age_column.number_input(
        "Age at enrollment (years)",
        min_value=18,
        max_value=110,
        value=60,
        step=1,
        key="age_at_enrollment",
    )
    diabetes_age = diabetes_age_column.number_input(
        "Age at diabetes diagnosis (years)",
        min_value=0,
        max_value=110,
        value=44,
        step=1,
        key="DiabAge",
    )

    st.subheader("Study-eye treatment and surgery history")
    st.caption("Choose whether each event was recorded for the randomized study eye.")
    history_values = {}
    history_columns = st.columns(2)
    for index, (column_name, label) in enumerate(HISTORY_FIELDS.items()):
        with history_columns[index % 2]:
            history_values[column_name] = history_input(column_name, label)

    st.subheader("Baseline systemic measurements")
    systemic_left, systemic_right = st.columns(2)
    baseline_systolic_avg = systemic_left.number_input(
        "Average systolic BP (mmHg)",
        min_value=50.0,
        max_value=250.0,
        value=138.3,
        step=0.1,
        format="%.1f",
        key="baseline_systolic_avg",
    )
    baseline_diastolic_avg = systemic_right.number_input(
        "Average diastolic BP (mmHg)",
        min_value=30.0,
        max_value=150.0,
        value=83.0,
        step=0.1,
        format="%.1f",
        key="baseline_diastolic_avg",
    )
    baseline_hba1c = systemic_left.number_input(
        "HbA1c (%)",
        min_value=3.0,
        max_value=20.0,
        value=7.8,
        step=0.1,
        format="%.1f",
        key="baseline_hba1c",
    )
    baseline_iop = systemic_right.number_input(
        "Study-eye IOP (mmHg)",
        min_value=0.0,
        max_value=80.0,
        value=15.0,
        step=0.5,
        format="%.1f",
        key="baseline_iop",
    )

    st.subheader("Baseline OCT measurements")
    oct_left, oct_right = st.columns(2)
    baseline_site_oct_center = oct_left.number_input(
        "Site OCT center thickness",
        min_value=0.0,
        max_value=2000.0,
        value=426.5,
        step=0.5,
        key="baseline_site_oct_center",
    )
    baseline_site_oct_signal = oct_right.number_input(
        "Site OCT signal strength",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="baseline_site_oct_signal",
    )
    baseline_rc_oct_cent_point = oct_left.number_input(
        "Reading-center OCT central point",
        min_value=0.0,
        max_value=2000.0,
        value=441.0,
        step=1.0,
        key="baseline_rc_oct_cent_point",
    )
    baseline_rc_oct_center = oct_right.number_input(
        "Reading-center OCT center thickness",
        min_value=0.0,
        max_value=2000.0,
        value=426.0,
        step=1.0,
        key="baseline_rc_oct_center",
    )
    baseline_rc_oct_center_calc = oct_left.number_input(
        "Calculated OCT central-subfield thickness",
        min_value=0.0,
        max_value=2000.0,
        value=429.0,
        step=1.0,
        key="baseline_rc_oct_center_calc",
    )
    baseline_rc_oct_signal = oct_right.number_input(
        "Reading-center OCT signal strength",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="baseline_rc_oct_signal",
    )

    st.subheader("Baseline visual acuity")
    study_eye_etdrs_baseline = st.number_input(
        "Study-eye ETDRS letter score",
        min_value=0.0,
        max_value=100.0,
        value=69.0,
        step=1.0,
        key="study_eye_etdrs_baseline",
    )

    submitted = st.form_submit_button(
        "Save numerical inputs",
        type="primary",
        use_container_width=True,
    )

if submitted:

    with st.spinner("Wait for it...", show_time=True):
        numerical_values = {
            "age_at_enrollment": age_at_enrollment,
            "DiabAge": diabetes_age,
            **history_values,
            "baseline_iop": baseline_iop,
            "baseline_systolic_avg": baseline_systolic_avg,
            "baseline_diastolic_avg": baseline_diastolic_avg,
            "baseline_hba1c": baseline_hba1c,
            "baseline_site_oct_center": baseline_site_oct_center,
            "baseline_site_oct_signal": baseline_site_oct_signal,
            "baseline_rc_oct_cent_point": baseline_rc_oct_cent_point,
            "baseline_rc_oct_center": baseline_rc_oct_center,
            "baseline_rc_oct_center_calc": baseline_rc_oct_center_calc,
            "baseline_rc_oct_signal": baseline_rc_oct_signal,
            "study_eye_etdrs_baseline": study_eye_etdrs_baseline,
        }

        if diabetes_age > age_at_enrollment:
            st.warning(
                "Age at diabetes diagnosis is greater than age at enrollment. "
                "Please verify these values."
            )


            # Create input dataframe
        numerical_input = pd.DataFrame([numerical_values])

        # Load models
        numeric_imputer = load_model(NUMERIC_IMPUTER)
        numeric_scaler = load_model(NUMERIC_SCALER)


        imputed = numeric_imputer.transform(numerical_input)
        scaled = numeric_scaler.transform(imputed)

        # Restore the 7 binary columns to their original/imputed 0/1 values
        scaled[:, 2:9] = imputed[:, 2:9]

        st.session_state.patient_records = scaled

        st.toast("✅ Patient Details Recored")