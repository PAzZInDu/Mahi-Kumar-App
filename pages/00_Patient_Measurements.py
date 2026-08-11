import pickle

import pandas as pd
import streamlit as st

# Require user authentication
if not st.user.is_logged_in:
    st.error("Please log in to access the app.")
    st.stop()


st.set_page_config(
    page_title="Patient Measurements",
    page_icon="📊",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main .block-container {max-width: 850px; padding-top: 2rem;}
    [data-testid="stAppViewContainer"] {background: #050505;}
    div[data-testid="stForm"] {background: #0f0f0f; border: 1px solid #333333; border-radius: 18px; padding: 1.4rem;}
    div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label {font-weight: 600;}
    .info-card {padding: 1rem 1.2rem; border-radius: 14px; background: #171717;
        border: 1px solid #7f1d1d; border-left: 5px solid #ef4444;
        color: #fee2e2; margin: .5rem 0 1.5rem;}
    div[data-testid="stNumberInput"]:focus-within,
    div[data-testid="stSelectbox"]:focus-within {
        border: 1px solid #ef4444; border-radius: 10px;
        box-shadow: 0 0 16px rgba(239, 68, 68, .22);
    }
    button[kind="primary"] {
        background: linear-gradient(90deg, #ef4444 0%, #b91c1c 55%, #7f1d1d 100%);
        border: 1px solid #ef4444; color: #ffffff;
    }
    button[kind="primary"]:hover {border-color: #fca5a5; filter: brightness(1.08);}
    </style>
    """,
    unsafe_allow_html=True,
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
        help="Choose Yes only when this treatment or operation is documented in the patient's medical record.",
    )
    return 1.0 if answer == "Yes" else 0.0


st.title("📋 Patient Measurements")
st.markdown(
    """
    <div class="info-card"><strong>Step 1 of 2</strong><br>
    Enter measurements taken before treatment began. Most values can be found in the
    baseline visit notes, blood-test report, eye examination, and OCT report.</div>
    """,
    unsafe_allow_html=True,
)
st.caption("Use the ⓘ beside a field for guidance. Replace all example values with the patient's actual results.")

with st.form("numerical_patient_inputs"):
    st.subheader("👤 About the patient")
    st.caption("Start with the patient's current age and the age when diabetes was first diagnosed.")
    age_column, diabetes_age_column = st.columns(2)
    age_at_enrollment = age_column.number_input(
        "Age at enrollment (years)",
        min_value=18,
        max_value=110,
        value=60,
        step=1,
        key="age_at_enrollment",
        help="Age in completed years at the start of the study or treatment.",
    )
    diabetes_age = diabetes_age_column.number_input(
        "Age at diabetes diagnosis (years)",
        min_value=0,
        max_value=110,
        value=44,
        step=1,
        key="DiabAge",
        help="Age when a clinician first diagnosed diabetes. This should normally not be greater than the age above.",
    )

    st.subheader("🩺 Previous treatment and surgery")
    st.caption("Answer for the study eye—the eye being assessed—not the other eye.")
    history_values = {}
    history_columns = st.columns(2)
    for index, (column_name, label) in enumerate(HISTORY_FIELDS.items()):
        with history_columns[index % 2]:
            history_values[column_name] = history_input(column_name, label)

    st.subheader("❤️ General health measurements")
    st.caption("Use readings recorded at the baseline visit, before the new eye treatment began.")
    systemic_left, systemic_right = st.columns(2)
    baseline_systolic_avg = systemic_left.number_input(
        "Average systolic BP (mmHg)",
        min_value=50.0,
        max_value=250.0,
        value=138.3,
        step=0.1,
        format="%.1f",
        key="baseline_systolic_avg",
        help="The upper number in a blood-pressure reading. Enter the recorded average if several readings were taken.",
    )
    baseline_diastolic_avg = systemic_right.number_input(
        "Average diastolic BP (mmHg)",
        min_value=30.0,
        max_value=150.0,
        value=83.0,
        step=0.1,
        format="%.1f",
        key="baseline_diastolic_avg",
        help="The lower number in a blood-pressure reading. Enter the recorded average if several readings were taken.",
    )
    baseline_hba1c = systemic_left.number_input(
        "HbA1c (%)",
        min_value=3.0,
        max_value=20.0,
        value=7.8,
        step=0.1,
        format="%.1f",
        key="baseline_hba1c",
        help="HbA1c shows average blood sugar over roughly 2–3 months. Copy the percentage from the baseline lab report.",
    )
    baseline_iop = systemic_right.number_input(
        "Study-eye IOP (mmHg)",
        min_value=0.0,
        max_value=80.0,
        value=15.0,
        step=0.5,
        format="%.1f",
        key="baseline_iop",
        help="Intraocular pressure (IOP) is the pressure inside the study eye, measured during the eye examination.",
    )

    st.subheader("👁️ OCT scan measurements")
    st.caption("OCT is the scan used to measure retinal thickness. Copy these values from the baseline OCT report.")
    oct_left, oct_right = st.columns(2)
    baseline_site_oct_center = oct_left.number_input(
        "Site OCT center thickness",
        min_value=0.0,
        max_value=2000.0,
        value=426.5,
        step=0.5,
        key="baseline_site_oct_center",
        help="Central retinal thickness measured by the clinic or study site, usually in micrometres (µm).",
    )
    baseline_site_oct_signal = oct_right.number_input(
        "Site OCT signal strength",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="baseline_site_oct_signal",
        help="The image-quality or signal-strength score shown by the clinic's OCT machine.",
    )
    baseline_rc_oct_cent_point = oct_left.number_input(
        "Reading-center OCT central point",
        min_value=0.0,
        max_value=2000.0,
        value=441.0,
        step=1.0,
        key="baseline_rc_oct_cent_point",
        help="Thickness at the exact centre of the retina measured by the reading centre, usually in micrometres (µm).",
    )
    baseline_rc_oct_center = oct_right.number_input(
        "Reading-center OCT center thickness",
        min_value=0.0,
        max_value=2000.0,
        value=426.0,
        step=1.0,
        key="baseline_rc_oct_center",
        help="Average thickness of the central retinal area reported by the independent reading centre.",
    )
    baseline_rc_oct_center_calc = oct_left.number_input(
        "Calculated OCT central-subfield thickness",
        min_value=0.0,
        max_value=2000.0,
        value=429.0,
        step=1.0,
        key="baseline_rc_oct_center_calc",
        help="Calculated central-subfield thickness shown in the reading-centre OCT report.",
    )
    baseline_rc_oct_signal = oct_right.number_input(
        "Reading-center OCT signal strength",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="baseline_rc_oct_signal",
        help="The OCT image-quality score assigned by the independent reading centre.",
    )

    st.subheader("🔤 Vision test")
    st.caption("Enter the letter score from the study eye's baseline ETDRS vision-chart test.")
    study_eye_etdrs_baseline = st.number_input(
        "Study-eye ETDRS letter score",
        min_value=0.0,
        max_value=100.0,
        value=69.0,
        step=1.0,
        key="study_eye_etdrs_baseline",
        help="Number of ETDRS chart letters correctly identified. A higher score generally means better visual acuity.",
    )

    submitted = st.form_submit_button(
        "Save measurements and continue",
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

        st.success("Measurements saved. Continue to Step 2 to add the remaining patient details.")

        st.toast("✅ Patient Details Recored")

if len(st.session_state.get("patient_records", [])):
    st.info("Step 1 is complete. Your measurements will be used on the next page.")
    if st.button(
        "Continue to Treatment Response →",
        type="primary",
        use_container_width=True,
    ):
        st.switch_page("pages/01_Treatment_Response.py")
