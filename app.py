import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import time
import os
import base64
import streamlit.components.v1 as components
from coupler_engine import run_asymmetric_simulation

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(
    page_title="Asymmetric Directional Coupler Solver",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GOOGLE ANALYTICS INTEGRATION ---
def inject_google_analytics(measurement_id):
    ga_code = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', '{measurement_id}', {{
          'page_path': window.parent.location.pathname,
          'cookie_flags': 'SameSite=None;Secure'
      }});
    </script>
    """
    components.html(ga_code, height=0, width=0)

inject_google_analytics("G-7776KX662W")

st.title("⚡ Asymmetric Directional Coupler (ADC) Solver")
st.markdown("### 2D Mode Solver & Mismatched Coupled-Mode Analysis (`Si / Si3N4 / Al2O3 / SiO2`)")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🧪 Material Selection")
core_material = st.sidebar.selectbox(
    "Core Material",
    options=[
        "Si3N4 (Stoichiometric)",
        "SiN (Low Stress)",
        "Al2O3 (Alumina)",
        "Si (Silicon)"
    ],
    index=0
)

st.sidebar.header("🛠️ Asymmetric Geometry")
w1 = st.sidebar.number_input("Waveguide 1 Width w1 [μm]", value=1.0, step=0.05)
w2 = st.sidebar.number_input("Waveguide 2 Width w2 [μm]", value=0.8, step=0.05)
h_core = st.sidebar.number_input("Waveguide Height h [μm]", value=0.3, step=0.05)
gap = st.sidebar.number_input("Coupler Gap [μm]", value=0.3, step=0.05)
coupler_L = st.sidebar.number_input("Straight Length L [μm]", value=35.0, step=5.0)
ring_R = st.sidebar.number_input("Ring Radius R [μm] (0=Straight)", value=100.0, step=10.0)
bottom_ox = st.sidebar.number_input("Bottom Oxide Height [μm]", value=4.0, step=0.5)
top_ox = st.sidebar.number_input("Top Oxide Height [μm]", value=1.0, step=0.1)

st.sidebar.header("🎯 Loss / Q_L Settings")
loss_1 = st.sidebar.number_input("Loss 1 [dB/cm]", value=0.5, step=0.1)
loss_2 = st.sidebar.number_input("Loss 2 [dB/cm]", value=1.5, step=0.1)
loss_3 = st.sidebar.number_input("Loss 3 [dB/cm]", value=5.0, step=0.5)
custom_losses = [loss_1, loss_2, loss_3]

st.sidebar.header("🔬 Simulation Settings")
lambda_start = st.sidebar.number_input("Start Wavelength [μm]", value=1.5, step=0.05)
lambda_end = st.sidebar.number_input("End Wavelength [μm]", value=1.6, step=0.05)
n_lambda = st.sidebar.slider("Wavelength Points", min_value=3, max_value=21, value=11, step=2)
polarization = st.sidebar.selectbox("Polarization", options=["ex", "ey"], index=0)
res_mode = st.sidebar.selectbox("Mesh Resolution", options=["lr (0.02μm)", "mr (0.01μm)", "hr (0.005μm)"], index=0)

run_btn = st.sidebar.button("🚀 Run ADC Simulation", type="primary", use_container_width=True)

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# --- EXECUTION & DISPLAY ---
if run_btn or 'sim_adc_results' in st.session_state:
    if run_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            pct = int((current / total) * 100)
            progress_bar.progress(pct)
            status_text.markdown(f"⏳ **Calculating ADC Wavelength {current} of {total} ({pct}%)...**")

        results = run_asymmetric_simulation(
            w1, w2, h_core, gap, coupler_L, ring_R,
            lambda_start, lambda_end, n_lambda, polarization, res_mode, top_ox, bottom_ox,
            core_material, progress_callback=update_progress
        )
        
        status_text.success("✅ Asymmetric coupler simulation completed successfully!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        alpha_db_vals = np.array(custom_losses)
        alpha_cm = alpha_db_vals * (np.log(10) / 10.0)
        L_ring_cm = results['L_ring_um'] * 1e-4
        round_trip_loss_pct = (1.0 - np.exp(-alpha_cm * L_ring_cm)) * 100.0
        
        neff_avg_vec = (results['neff_even'] + results['neff_odd']) / 2.0
        lambda_cm_center = results['lambda_center_val'] * 1e-4
        dneff_dlambda = (neff_avg_vec[-1] - neff_avg_vec[0]) / ((results['lambda_vec'][-1] - results['lambda_vec'][0]) * 1e-4)
        n_group = neff_avg_vec[results['idx_center']] - lambda_cm_center * dneff_dlambda
        
        Q0_vals = (2.0 * np.pi * n_group) / (lambda_cm_center * alpha_cm)
        QL_vals = Q0_vals / 2.0
        
        results['alpha_db_vals'] = alpha_db_vals
        results['round_trip_loss_pct'] = round_trip_loss_pct
        results['QL_vals'] = QL_vals
        
        st.session_state['sim_adc_results'] = results

    d = st.session_state['sim_adc_results']
    c_idx = d['idx_center']

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Pure Coupling (κ)", f"{d['kappa_pure_vec'][c_idx]:.4f} μm⁻¹")
    m2.metric("Phase Mismatch (δ)", f"{d['delta_vec'][c_idx]:.4f} μm⁻¹")
    m3.metric("Max Transfer (F=P_max)", f"{d['f_max_vec'][c_idx]*100:.1f} %")
    m4.metric("Cross Power Transferred", f"{d['p_cross_vec'][c_idx]:.1f} %")
    m5.metric(f"Q_L (at α = {d['alpha_db_vals'][1]} dB/cm)", f"{d['QL_vals'][1]/1e3:.1f} k")

    st.markdown("---")
    
    df_results = pd.DataFrame({
        "Wavelength_um": d['lambda_vec'],
        "Neff_WG1": d['neff1_vec'],
        "Neff_WG2": d['neff2_vec'],
        "Neff_Even": d['neff_even'],
        "Neff_Odd": d['neff_odd'],
        "Delta_PhaseMismatch": d['delta_vec'],
        "Kappa_Pure": d['kappa_pure_vec'],
        "Kappa_Eff": d['kappa_eff_vec'],
        "F_MaxPowerFraction": d['f_max_vec'],
        "P_cross_percent": d['p_cross_vec'],
        "P_bar_percent": d['p_bar_vec']
    })

    def draw_boxes(ax):
        for l, r in [(d['box1_l'], d['box1_r']), (d['box2_l'], d['box2_r'])]:
            ax.plot([l, r, r, l, l], [d['b_y'], d['b_y'], d['t_y'], d['t_y'], d['b_y']], 'k--', lw=1.5)

    fig_idx, ax_idx = plt.subplots(figsize=(6, 4))
    im_idx = ax_idx.imshow(np.sqrt(d['eps_center']).T, origin='lower', extent=[d['xc'][0], d['xc'][-1], d['yc'][0], d['yc'][-1]], cmap='viridis', aspect='auto')
    fig_idx.colorbar(im_idx, ax=ax_idx, label='Index (n)')
    draw_boxes(ax_idx)
    ax_idx.set_title(f"Asymmetric Index Profile (w1={d['w1']}μm, w2={d['w2']}μm)")

    fig_even, ax_even = plt.subplots(figsize=(6, 4))
    im_even = ax_even.imshow(d['phi_even'].T, origin='lower', extent=[d['xc'][0], d['xc'][-1], d['yc'][0], d['yc'][-1]], cmap='jet', vmin=0, vmax=1, aspect='auto')
    fig_even.colorbar(im_even, ax=ax_even, label='Field')
    draw_boxes(ax_even)
    ax_even.set_title("Asymmetric Even Supermode")

    fig_odd, ax_odd = plt.subplots(figsize=(6, 4))
    im_odd = ax_odd.imshow(d['phi_odd'].T, origin='lower', extent=[d['xc'][0], d['xc'][-1], d['yc'][0], d['yc'][-1]], cmap='jet', vmin=-1, vmax=1, aspect='auto')
    fig_odd.colorbar(im_odd, ax=ax_odd, label='Field')
    draw_boxes(ax_odd)
    ax_odd.set_title("Asymmetric Odd Supermode")

    fig_1d, ax_1d = plt.subplots(figsize=(6, 4))
    ax_1d.plot(d['xc'], d['phi_even'][:, d['mid_y_idx']], 'b-', lw=2, label='Even')
    ax_1d.plot(d['xc'], d['phi_odd'][:, d['mid_y_idx']], 'r--', lw=2, label='Odd')
    ax_1d.grid(True)
    ax_1d.legend()
    ax_1d.set_title("1D Cutline at Core Center")

    fig_neff, ax_neff = plt.subplots(figsize=(6, 4))
    ax_neff.plot(d['lambda_vec'], d['neff1_vec'], 'b--', lw=2, label='n_eff WG1')
    ax_neff.plot(d['lambda_vec'], d['neff2_vec'], 'g--', lw=2, label='n_eff WG2')
    ax_neff.plot(d['lambda_vec'], d['neff_even'], 'ro-', lw=1.5, label='n_eff Even')
    ax_neff.plot(d['lambda_vec'], d['neff_odd'], 'm^-', lw=1.5, label='n_eff Odd')
    ax_neff.grid(True)
    ax_neff.legend()
    ax_neff.set_xlabel('Wavelength [μm]')
    ax_neff.set_ylabel('Effective Index')
    ax_neff.set_title("Mismatched & Supermode Index Curves")

    fig_kappa, ax_kappa = plt.subplots(figsize=(6, 4))
    ax_kappa.plot(d['lambda_vec'], d['kappa_pure_vec'], 'k-', lw=2, label='Pure Coupling κ')
    ax_kappa.plot(d['lambda_vec'], d['delta_vec'], 'r--', lw=2, label='Phase Mismatch δ')
    ax_kappa.plot(d['lambda_vec'], d['kappa_eff_vec'], 'g:', lw=2, label='Effective Coupling κ_eff')
    ax_kappa.grid(True)
    ax_kappa.legend()
    ax_kappa.set_xlabel('Wavelength [μm]')
    ax_kappa.set_ylabel('Parameter Value [μm⁻¹]')
    ax_kappa.set_title("Coupling & Phase Mismatch Spectra")

    fig_power, ax_power = plt.subplots(figsize=(7, 4))
    ax_power.plot(d['lambda_vec'], d['p_cross_vec'], 'ro-', lw=2, label='Cross Port Power P_cross')
    ax_power.plot(d['lambda_vec'], d['p_bar_vec'], 'bo-', lw=2, label='Bar Port Power P_bar')
    ax_power.plot(d['lambda_vec'], d['f_max_vec']*100.0, 'k--', lw=1.5, label='Max Power Fraction F')
    ax_power.grid(True)
    ax_power.set_ylim(0, 105)
    ax_power.set_xlabel('Wavelength [μm]')
    ax_power.set_ylabel('Power Transfer [%]')
    ax_power.legend()
    ax_power.set_title("Power Transfer Spectrum with Mismatch Limit F")

    csv_bytes = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📄 Download Raw Data (CSV)", data=csv_bytes, file_name="asymmetric_coupler_results.csv", mime="text/csv")

    tab1, tab2, tab3 = st.tabs(["🖼️ Cross-Sections & Modes", "📈 Dispersion & Coupling", "⚡ Power Transfer Spectrum"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_idx)
            st.pyplot(fig_odd)
        with col2:
            st.pyplot(fig_even)
            st.pyplot(fig_1d)

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(fig_neff)
        with col4:
            st.pyplot(fig_kappa)

    with tab3:
        st.pyplot(fig_power)

else:
    # --- REFERENCE BENCHMARKS PREVIEW ---
    st.info("👈 Select waveguide geometry in the sidebar, then click **Run ADC Simulation** 🚀")
    
    st.markdown("### 🔬 Reference Asymmetric Modal Profiles & Benchmarks 🎨")
    st.markdown("Below are standard reference solutions calculated for an asymmetric directional coupler:")

    preview_items = [
        {"file": "index_profile.png", "title": "1. Asymmetric Cross-Sectional Index Distribution n(x,y) 📐"},
        {"file": "even_mode.png", "title": "2. Asymmetric Even Supermode Distribution ⚡"},
        {"file": "odd_mode.png", "title": "3. Asymmetric Odd Supermode Distribution 🌊"},
        {"file": "1d_profiles.png", "title": "4. 1D Transverse Cutline Field Distributions 📊"},
        {"file": "dispersion.png", "title": "5. Mismatched Supermode Dispersion Curves n_eff(λ) 📈"},
        {"file": "ring_loss_QL.png", "title": "6. Power Coupling & Mismatch Fraction F vs. Wavelength 🎯"}
    ]
    
    valid_items = [item for item in preview_items if os.path.exists(item["file"])]
    
    if valid_items:
        encoded_slides = []
        for idx, item in enumerate(valid_items):
            with open(item["file"], "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode()
            encoded_slides.append(f"""
                <div class="mySlides fade" style="display: {'block' if idx==0 else 'none'}; text-align: center;">
                    <div style="font-weight: 600; font-size: 15px; margin-bottom: 10px; color: #0F172A; font-family: sans-serif;">
                        {item['title']}
                    </div>
                    <img src="data:image/png;base64,{b64}" style="max-width: 82%; height: auto; border-radius: 8px; border: 1px solid #CBD5E1; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                </div>
            """)

        carousel_html = f"""
        <div id="slideshow-container" style="max-width: 760px; position: relative; margin: 10px auto; padding: 18px; background: #F8FAFC; border-radius: 12px; border: 1px solid #E2E8F0;">
            {''.join(encoded_slides)}
        </div>
        <script>
            let slideIndex = 0;
            showSlides();
            function showSlides() {{
                let i;
                let slides = document.getElementsByClassName("mySlides");
                for (i = 0; i < slides.length; i++) {{
                    slides[i].style.display = "none";  
                }}
                slideIndex++;
                if (slideIndex > slides.length) {{slideIndex = 1}}    
                if (slides[slideIndex-1]) {{
                    slides[slideIndex-1].style.display = "block";  
                }}
                setTimeout(showSlides, 3000);
            }}
        </script>
        """
        st.components.v1.html(carousel_html, height=490)
