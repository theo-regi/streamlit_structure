import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scripts.products import SSVICalibration, DupireLocalVol, OptionMarket
import constants
#______________________________________INITIALISATION PAGE____________________________________________
STYLE_PATH=st.session_state.STYLE_PATH
st.set_page_config(page_title="Volatility Models Viewer", layout="wide")
def apply_css(STYLE_PATH):
    with open(STYLE_PATH, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
apply_css(STYLE_PATH)

#-----------------------------------------------------------------------------------------------------
#------------------------------------Page construction du portefeuille---------------------------------
#-----------------------------------------------------------------------------------------------------
st.title("Volatility Surface Analysis")
# Use pricing date from session
pricing_date = st.session_state.pricing_date
st.info(f"Pricing Date: **{pricing_date}**")

# Tabs for SVI, SSVI, Dupire
svi_tab, ssvi_tab, dupire_tab = st.tabs(["SVI", "SSVI", "Dupire Local Volatility"])
spot = st.session_state.spot
with svi_tab:
    st.header("SVI Volatility Slice")
    cache_key = (constants.FILE_PATH, constants.FILE_UNDERLYING)
    svis = constants.get_from_cache("SVI_PARAMS", cache_key)
    if svis is None:
        dup = DupireLocalVol("Black-Scholes-Merton", constants.FILE_PATH, constants.FILE_UNDERLYING, pricing_date=pricing_date)
        svis = constants.get_from_cache("SVI_PARAMS", cache_key)
    
    available_dates = svis.keys()
    maturity = st.selectbox("Available Dates :", available_dates)

    if st.button("Calibrate SVI", key="svi_btn"):
        try:
            om = constants.get_from_cache("OptionMarket", cache_key)
            if om is None:
                om = OptionMarket("Black-Scholes-Merton", constants.FILE_PATH, constants.FILE_UNDERLYING, pricing_date)

            types, strikes, prices, spot, T, options = om.get_values_for_calibration_SVI(pricing_date, maturity)
            types = ["Call" if tp == constants.OptionType.CALL else "Put" for tp in types]
            st.success(f"Loaded from calibration {len(strikes)} options | Spot: {spot:.2f} | T = {T:.2f}y")

            st.metric(label="Spot on valuation date", value=f"{spot:.2f}")

            # Retrieve SVI parameters from cache
            svi_params = svis.get(maturity) if svis else None
            if svi_params is None:
                st.warning("⚠️ SVI parameters not found in cache for the selected maturity.")
            else:
                col1, col2 = st.columns([4,4])

                a, b, p, m, s = svi_params
                strikes_sim = np.linspace(min(strikes) * 0.8, max(strikes) * 1.2, 50)
                # Estimate implied vol from SVI for each strike
                k_vec_sim = np.log(np.array(strikes_sim) / spot)
                w_vec_sim = a + b * (p * (k_vec_sim - m) + np.sqrt((k_vec_sim - m)**2 + s**2))
                implied_vols_sim = np.sqrt(w_vec_sim / T)

                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=strikes_sim, y=implied_vols_sim,
                                            mode='lines',
                                            name='SVI Implied Vol',
                                            line=dict(color='#002060')))
                    fig.update_layout(title="SVI Implied Volatility by Strike",
                                    xaxis_title="Strike",
                                    yaxis_title="Implied Volatility",
                                    yaxis=dict(range=[0, max(implied_vols_sim)*1.1]),
                                    height=500)
                    st.plotly_chart(fig, use_container_width=True, height=600)

                    with col2:
                        df_params = pd.DataFrame({
                            "α (alpha)": ["{:.4f}".format(a)],
                            "β (beta)": ["{:.4f}".format(b)],
                            "ρ (rho)": ["{:.4f}".format(p)],
                            "μ (mu)": ["{:.4f}".format(m)],
                            "σ (sigma)": ["{:.4f}".format(s)]
                        })
                        df_params.index = ['']* len(df_params.index)
                        st.write(f"**SVI Parameters for {maturity}:**")
                        st.table(df_params)

                        df_result = pd.DataFrame({
                            "Type": types,
                            "Strike": strikes,
                            "Market Price": prices,
                            "Implied Vol (SVI)": np.interp(strikes, strikes_sim, implied_vols_sim)
                        })
                        df_result = df_result.reset_index(drop=True)
                        st.dataframe(df_result, use_container_width=True, height=500)

        except Exception as e:
            st.error(f"❌ Error during SVI calibration: {e}")

with ssvi_tab:
    cache_key = (constants.FILE_PATH, constants.FILE_UNDERLYING)
    ssvi = constants.get_from_cache("SSVICalibration", cache_key)
    if ssvi is None:
        ssvi = SSVICalibration("Black-Scholes-Merton", constants.FILE_PATH, constants.FILE_UNDERLYING, pricing_date)

    st.header("SSVI Volatility Surface")
    if st.button("Calibrate SSVI", key="ssvi_btn"):
        try:
            params = ssvi.get_ssvi_params()
            # Display parameters nicely
            df_params = pd.DataFrame({k: [f"{v:.4f}"] for k, v in params.items()})
            df_params.index = ['']
            st.markdown("**SSVI Parameters:**")
            st.table(df_params)

            # Create a 3D vol surface
            maturities = ssvi._maturities_t.values()
            K = np.linspace(spot * 0.5, spot * 1.5, 30)
            T = np.linspace(min(maturities), max(maturities), 30)
            K_grid, T_grid = np.meshgrid(K, T)
            vol_grid = np.vectorize(ssvi.__call__)(K_grid, T_grid)

            fig3d = go.Figure(data=[go.Surface(x=K_grid, y=T_grid, z=vol_grid,
                                               colorscale='Viridis')])
            fig3d.update_layout(title='SSVI Implied Volatility Surface',
                                scene=dict(
                                    xaxis_title='Strike',
                                    yaxis_title='Maturity (T)',
                                    zaxis_title='Implied Volatility',
                                    zaxis=dict(range=[vol_grid.min() * 0.7, vol_grid.max() * 1.3])
                                    ),
                                height=800, width=800)
            st.plotly_chart(fig3d, use_container_width=True)

            # Show a matrix with some sampled values
            sampled_K = np.linspace(spot * 0.5, spot * 1.5, 20)
            sampled_T = np.linspace(min(maturities), max(maturities), 20)
            matrix_data = [[round(ssvi(k, t), 4) for k in sampled_K] for t in sampled_T]
            df_matrix = pd.DataFrame(matrix_data, index=[f"T={t:.2f}" for t in sampled_T], columns=[f"K={k:.2f}" for k in sampled_K])
            st.markdown("**Sampled Implied Volatilities from SSVI σ(K, T)**")
            st.dataframe(df_matrix, use_container_width=True, height=600)

        except Exception as e:
            st.error(f"❌ Error during SSVI calibration: {e}")

with dupire_tab:
    cache_key = (constants.FILE_PATH, constants.FILE_UNDERLYING)
    dupire = constants.get_from_cache("DupireLocalVol", cache_key)
    if dupire is None:
        dupire = DupireLocalVol("Black-Scholes-Merton", constants.FILE_PATH, constants.FILE_UNDERLYING, pricing_date)

    st.header("Dupire Local Volatility Surface")
    if st.button("Calibrate Dupire Local Vol", key="dupire_btn"):
        try:
            vol_matrix = dupire.get_implied_vol_matrix()
            min_T = 0.02
            vol_matrix = vol_matrix[vol_matrix.index.astype(float) >= min_T]
            # 3D Surface plot from vol_matrix
            strikes = np.array([round(strike,2) for strike in vol_matrix.columns])
            maturities = np.array(vol_matrix.index, dtype=float)
            K_grid, T_grid = np.meshgrid(strikes, maturities)
            Z = vol_matrix.to_numpy()

            fig3d = go.Figure(data=[go.Surface(x=K_grid, y=T_grid, z=Z,
                                               colorscale='Viridis')])
            fig3d.update_layout(title='Dupire Local Volatility Surface',
                                scene=dict(
                                    xaxis_title='Strike',
                                    yaxis_title='Maturity (T)',
                                    zaxis_title='Implied Volatility',
                                    zaxis=dict(range=[0, np.nanmax(Z) * 1.3])
                                ),
                                height=800, width=800)
            st.plotly_chart(fig3d, use_container_width=True)
            
            # Display matrix preview
            st.markdown("**Implied Volatility Matrix σ(K, T) Sample:**")
            st.dataframe(vol_matrix, use_container_width=True, height=600)

        except Exception as e:
            st.error(f"❌ Error during Dupire calibration: {e}")
