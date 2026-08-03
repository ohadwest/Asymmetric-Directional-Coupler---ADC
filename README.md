# ⚡ Asymmetric Directional Coupler (ADC) Solver

A 2D Semi-Vectorial Finite Difference (SVFD) mode solver and phase-mismatched coupled-mode theory framework for asymmetric integrated optical couplers.

---

## 🔬 Physics & Asymmetric Coupling Formulation

### 1. Phase Mismatch ($\delta$)
For two waveguides of different widths ($w_1 \neq w_2$), the isolated propagation constants are:

$$\beta_1 = \frac{2\pi}{\lambda} n_{\text{eff},1}, \quad \beta_2 = \frac{2\pi}{\lambda} n_{\text{eff},2}$$

The phase mismatch parameter $\delta$ is defined as:

$$\delta = \frac{\beta_1 - \beta_2}{2} = \frac{\pi}{\lambda} \left( n_{\text{eff},1} - n_{\text{eff},2} \right)$$

---

### 2. Pure vs. Effective Coupling Coefficients ($\kappa$ and $\kappa_{\text{eff}}$)
From the 2D SVFD supermode eigenvalue solution ($n_{\text{eff, even}}$ and $n_{\text{eff, odd}}$), the total supermode index difference yields:

$$\kappa_{\text{eff}} = \frac{\beta_{\text{even}} - \beta_{\text{odd}}}{2} = \frac{\pi}{\lambda} \left( n_{\text{eff, even}} - n_{\text{eff, odd}} \right)$$

The pure coupling coefficient $\kappa$ is extracted via:

$$\kappa = \sqrt{\kappa_{\text{eff}}^2 - \delta^2} = \sqrt{\left[ \frac{\pi}{\lambda} (n_{\text{eff, even}} - n_{\text{eff, odd}}) \right]^2 - \left[ \frac{\pi}{\lambda} (n_{\text{eff},1} - n_{\text{eff},2}) \right]^2}$$

---

### 3. Power Transfer Dynamics ($P_{\text{cross}}$ / $P_{\text{bar}}$)
Due to phase mismatch, maximum power transfer is bounded by $F$:

$$F = \frac{\kappa^2}{\kappa^2 + \delta^2} = \frac{\kappa^2}{\kappa_{\text{eff}}^2} \le 100\%$$

$$P_{\text{cross}}(\lambda) = F \cdot \sin^2\left( \kappa_{\text{eff}}(\lambda) \cdot L_{\text{total}}(\lambda) \right) \times 100\%$$

$$P_{\text{bar}}(\lambda) = 100\% - P_{\text{cross}}(\lambda)$$

---

## 🚀 Local Setup

```bash
git clone [https://github.com/YOUR_USERNAME/asymmetric_coupler_simulator.git](https://github.com/YOUR_USERNAME/asymmetric_coupler_simulator.git)
cd asymmetric_coupler_simulator
pip install -r requirements.txt
streamlit run app.py
