import streamlit as st


st.set_page_config(
    page_title="Visual Response Predictor",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {max-width: 900px; padding-top: 3rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {background: #f8fafc;}
    .hero {
        padding: 2.2rem; border-radius: 24px;
        background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
        border: 1px solid #dbeafe; margin-bottom: 1.5rem;
    }
    .hero-badge {
        display: inline-block; padding: .35rem .75rem; border-radius: 999px;
        background: #dbeafe; color: #1d4ed8; font-size: .82rem;
        font-weight: 700; letter-spacing: .03em; margin-bottom: .8rem;
    }
    .hero h1 {font-size: 2.45rem; line-height: 1.12; margin: 0 0 .8rem; color: #0f172a;}
    .hero p {font-size: 1.05rem; line-height: 1.65; color: #475569; margin: 0;}
    .step-card {
        min-height: 170px; padding: 1.25rem; border-radius: 18px;
        background: #ffffff; border: 1px solid #e2e8f0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, .05);
    }
    .step-number {
        width: 34px; height: 34px; display: inline-flex; align-items: center;
        justify-content: center; border-radius: 10px; background: #2563eb;
        color: white; font-weight: 700; margin-bottom: .7rem;
    }
    .step-card h3 {font-size: 1.05rem; margin: 0 0 .45rem; color: #0f172a;}
    .step-card p {font-size: .92rem; line-height: 1.5; color: #64748b; margin: 0;}
    .privacy-note {
        margin-top: 1.5rem; padding: 1rem 1.15rem; border-radius: 14px;
        background: #f8fafc; border-left: 4px solid #64748b; color: #475569;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">CLINICAL DECISION SUPPORT</div>
        <h1>Visual Response Predictor</h1>
        <p>
            Organize baseline patient information and estimate whether a study eye is
            likely to show a major visual response following treatment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("How it works")
step_one, step_two, step_three = st.columns(3)

with step_one:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">1</div>
            <h3>Enter measurements</h3>
            <p>Add baseline health, eye-pressure, OCT, and vision-test measurements.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_two:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">2</div>
            <h3>Add clinical findings</h3>
            <p>Select treatment, examination, OCT, and fundus-photography findings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_three:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">3</div>
            <h3>Review the estimate</h3>
            <p>Use the model result as supporting information alongside clinical judgement.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="privacy-note">
        <strong>Important:</strong> This tool provides an estimate from a trained model.
        It does not provide a diagnosis or replace advice from a qualified clinician.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Welcome")

if not st.user.is_logged_in:
    st.sidebar.write("Sign in securely to enter patient information and use the predictor.")
    if st.sidebar.button(
        "Log in with Google",
        type="primary",
        icon=":material/login:",
        use_container_width=True,
    ):
        st.login()
    st.info("Please log in with Google to begin the two-step assessment.", icon="🔐")
else:
    user_name = getattr(st.user, "name", None) or "there"
    st.sidebar.success(f"Signed in as {user_name}")

    if st.sidebar.button(
        "Log out",
        type="secondary",
        icon=":material/logout:",
        use_container_width=True,
    ):
        st.logout()
        st.stop()

    st.success(f"Welcome, {user_name}. You can begin with Step 1.", icon="✅")
    if st.button(
        "Start Patient Measurements →",
        type="primary",
        use_container_width=True,
    ):
        st.switch_page("pages/00_Patient_Measurements.py")
