"""
Visual binary orbital-element solver.

Fits all seven Campbell elements (P, T, e, a, i, Omega, omega) directly to
(theta, rho, t) astrometry by modeling the true Keplerian orbit and
projecting it onto the sky with the Thiele-Innes constants.

Assumptions:
- Both stars are approximately the same distance from Earth
- Binding energy < 0 (orbit is elliptical)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Target Name
target = "Test System"
unit = '"' # arcsec
csv_file = "test.csv"


n_starts = 100
guess_fractional_range = 0.3

# Inputs for period/semi-major axis constrainments (optional)
m1_guess = 1 # Solar masses
m2_guess = 1 # Solar masses

m1_PM_RA = None # mas/yr
m1_PM_Dec = None # mas/yr
m1_PM_RV = None # m/s

m2_PM_RA = None # mas/yr
m2_PM_Dec = None # mas/yr
m2_PM_RV = None # m/s
# ----------------------------------------------------------------------

# Constants (DO NOT CHANGE)
M_Sun = 1.98847e30  # kg
AU = 1.495978707e11  # m
# ----------------------------------------------------------------------


# For short observation arcs (<~30% of the period), P is unconstrained by
# the data and must be fixed from an external source (literature orbit,
# or a scan). Set to None to fit P freely (only sensible for long arcs).

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
data = np.genfromtxt(csv_file, delimiter=",", skip_header=1, usecols = (0, 1, 2))
theta_obs = np.deg2rad(data[:, 0])   # position angle, N through E
rho_obs = data[:, 1]                 # separation (arcsec)
t_obs = data[:, 2]                   # epoch (decimal year)

x_obs = rho_obs * np.cos(theta_obs)  #
y_obs = rho_obs * np.sin(theta_obs)  #

data = np.genfromtxt(csv_file, delimiter=",", skip_header=1, usecols=(0, 1, 2))
theta_obs = np.deg2rad(data[:, 0])
rho_obs = data[:, 1]
t_obs = data[:, 2]

# Parallax lives only on the first data row, column 4
first_row = np.genfromtxt(csv_file, delimiter=",", skip_header=1, max_rows=1)
parallax_mas = first_row[3]
parallax_arcsec = parallax_mas / 1000.0
# ----------------------------------------------------------------------
# Kepler Equation
# ----------------------------------------------------------------------
def solve_kepler(M, e, tol=1e-12, itmax=100):
    """Solve Kepler's equation M = E - e sin E (vectorized Newton)."""
    M = np.mod(M, 2 * np.pi)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(itmax):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def thiele_innes(a, i, Omega, omega):
    """Campbell elements -> Thiele-Innes constants A, B, F, G."""
    cO, sO = np.cos(Omega), np.sin(Omega)
    co, so = np.cos(omega), np.sin(omega)
    ci = np.cos(i)
    A = a * (co * cO - so * sO * ci)
    B = a * (co * sO + so * cO * ci)
    F = a * (-so * cO - co * sO * ci)
    G = a * (-so * sO + co * cO * ci)
    return A, B, F, G


def model_xy(params, t):
    """Predicted sky position (x=N, y=E) at epochs t for the 7 elements."""
    P, T, e, a, i, Omega, omega = params
    M = 2 * np.pi * (t - T) / P
    E = solve_kepler(M, e)
    # Elliptical rectangular coordinates in the true orbital plane
    X = np.cos(E) - e
    Y = np.sqrt(1 - e**2) * np.sin(E)
    A, B, F, G = thiele_innes(a, i, Omega, omega)
    x = A * X + F * Y   # North
    y = B * X + G * Y   # East
    return x, y


def residuals(params, t, x, y):
    xm, ym = model_xy(params, t)
    return np.concatenate([xm - x, ym - y])

"""
# Constrain orbital period based on mass guess and last observed separation
def constrain_period(m1_guess, m2_guess, m1_PM_RA, 
                     m1_PM_Dec, m1_PM_RV, m2_PM_RA, 
                     m2_PM_Dec, m2_PM_RV):
    m_total = (m1_guess + m2_guess) * M_Sun # kg
    m1 = m1_guess * M_Sun # kg
    m2 = m2_guess * M_Sun # kg
    last_sep = rho_obs[-1] 
    dist_pc = 1.0 / parallax_arcsec
    dist_AU = (1/parallax_arcsec) * 206265
    d_AU = (last_sep * dist_AU) / 206265
    
    # Convert proper motion from mas/yr to m/s using distance in parsecs
    if m1_PM_RA is not None and m1_PM_Dec is not None:
        m1_PM_RA_m_s = (m1_PM_RA/1000) * 4.7406 * dist_pc  # m/s
        m1_PM_Dec_m_s = (m1_PM_Dec/1000) * 4.7406 * dist_pc  # m/s

    if m2_PM_RA is not None and m2_PM_Dec is not None:
        m2_PM_RA_m_s = (m2_PM_RA/1000) * 4.7406 * dist_pc  # m/s
        m2_PM_Dec_m_s = (m2_PM_Dec/1000) * 4.7406 * dist_pc  # m/s
    

if (m1_guess is not None and m2_guess is not None
    and m1_PM_RV is not None and m2_PM_RV is not None 
    and m1_PM_RA is not None and m1_PM_Dec is not None 
    and m2_PM_RA is not None and m2_PM_Dec is not None 
    and m1_PM_RV is not None and m2_PM_RV is not None):
    constrain_period(m1_guess, m2_guess, 
                     m1_PM_RA, m1_PM_Dec, 
                     m1_PM_RV, m2_PM_RA, 
                     m2_PM_Dec, m2_PM_RV)
"""
# ----------------------------------------------------------------------
# Fit: multi-start nonlinear least squares
# ----------------------------------------------------------------------
#          P (yr)   T (yr)   e    a (")  i      Omega  omega
lower = [1e-1,   0.0,     0.0,   1e-3,  0.0,   0.0,   0.0]
upper = [3000.0, 3000.0,  0.999, 100.0, np.pi, 2*np.pi, 2*np.pi]

best = None

P_bounds = [0.1, 1000]  # Period bounds for random initialization
T_bounds = [0, 3000]   # Time of periastron bounds for random initialization
e_bounds = [0.00, 0.999]  # Eccentricity bounds for random initialization
a_bounds = [0.1, 60]  # Semi-major axis bounds for random initialization
i_bounds = [0, np.pi]  # Inclination bounds for random initialization
Omega_bounds = [0, 2 * np.pi]  # Longitude of ascending node bounds for random initialization
omega_bounds = [0, 2 * np.pi]  # Argument of periastron bounds for random initialization

bestcost_list = []
solcost_list = []
for k in range(n_starts):
    p0_full = [
        np.random.uniform(*P_bounds),          # P
        np.random.uniform(*T_bounds),         # T
        np.random.uniform(*e_bounds),          # e
        np.random.uniform(*a_bounds),              # a
        np.random.uniform(*i_bounds),   # i
        np.random.uniform(*Omega_bounds),       # Omega
        np.random.uniform(*omega_bounds),       # omega
    ]
    try:
        sol = least_squares(residuals, p0_full,
                            args=(t_obs, x_obs, y_obs),
                            bounds=(lower, upper), method="trf",
                            x_scale="jac", max_nfev=1000)
    except:
        continue
    
    
    if best is None or sol.cost < best.cost:
        best = sol

    bestcost_list.append(best.cost)
    solcost_list.append(sol.cost)
    """
    Print all values every iteration
    print(f'Iteration {k+1:2d}/{n_starts}: P={best.x[0]:.3f} yr, a"={best.x[3]:.3f}", e={best.x[2]:.3f}, i={np.degrees(best.x[4]):.3f}$^\\circ$, Omega={np.degrees(best.x[5]):.3f}$^\\circ$, omega={np.degrees(best.x[6]):.3f}$^\\circ$, T = {best.x[1]:.3f} yr, cost={best.cost:.6f}')
    """

    # P, a", e, cost
    print(f'Iteration {k+1:2d}/{n_starts}: P={best.x[0]} yr, a"={best.x[3]}", e={best.x[2]}, cost={best.cost}')

    P_bounds[0], P_bounds[1] = (1 - guess_fractional_range) * best.x[0], (1 + guess_fractional_range) * best.x[0]
    T_bounds[0], T_bounds[1] = (1 - guess_fractional_range) * best.x[1], (1 + guess_fractional_range) * best.x[1]
    e_bounds[0], e_bounds[1] = max(0.0, (1 - guess_fractional_range) * best.x[2]), min(0.999, (1 + guess_fractional_range) * best.x[2])
    a_bounds[0], a_bounds[1] = (1 - guess_fractional_range) * best.x[3], (1 + guess_fractional_range) * best.x[3]
    i_bounds[0], i_bounds[1] = max(0.0, (1 - guess_fractional_range) * best.x[4]), min(np.pi, (1 + guess_fractional_range) * best.x[4])
    Omega_bounds[0], Omega_bounds[1] = (1 - guess_fractional_range) * best.x[5], (1 + guess_fractional_range) * best.x[5]
    omega_bounds[0], omega_bounds[1] = (1 - guess_fractional_range) * best.x[6], (1 + guess_fractional_range) * best.x[6]


# Plot cost function convergence in log scale
plt.figure(figsize=(8, 5))
plt.semilogy(range(1, len(bestcost_list) + 1), bestcost_list, marker="o", markersize=4, color="tab:blue", label = "Current Fit Residuals", zorder = 2)
plt.semilogy(range(1, len(solcost_list) + 1), solcost_list, "k-", marker="x", markersize=4, label = "Random Fit Residuals", zorder = 1)

plt.xlabel("Iteration")
plt.xlim(1, len(bestcost_list))

plt.ylabel("Residuals")
#plt.ylim(0.99 * min(bestcost_list), 1.01 * max(bestcost_list))

plt.title("Residual Convergence")
plt.legend()
plt.savefig("residuals_convergence.png", dpi=200, bbox_inches="tight")
print("Cost function convergence plot saved to residuals_convergence.png")

P, T, e, a, i, Omega, omega = best.x

r = residuals(best.x, t_obs, x_obs, y_obs)
ss_res = np.sum(r**2)                     # = 2 * best.cost
obs = np.concatenate([x_obs, y_obs])
ss_tot = np.sum((obs - obs.mean())**2)
r_squared = 1 - ss_res / ss_tot


# ----------------------------------------------------------------------
# Physical quantities: distance scaling + Kepler's third law
# ----------------------------------------------------------------------
d_pc = 1.0 / parallax_arcsec
a_AU = a * d_pc
M_total = a_AU**3 / P**2   # solar masses



names = ["P (yr)", "T (yr)", "e", f"a ({unit})",
         "i (deg)", "Omega (deg)", "omega (deg)"]
vals = [P, T, e, a, np.degrees(i), np.degrees(Omega), np.degrees(omega)]




# ----------------------------------------------------------------------
# Plotting: Sky-Projected Orbit Fit and True Orbit Fit
# ----------------------------------------------------------------------
E_dense = np.linspace(0, 2 * np.pi, 2000)
X_d = np.cos(E_dense) - e
Y_d = np.sqrt(1 - e**2) * np.sin(E_dense)
A_, B_, F_, G_ = thiele_innes(a, i, Omega, omega)
x_fit = A_ * X_d + F_ * Y_d
y_fit = B_ * X_d + G_ * Y_d

# Periastron: E = 0
x_peri, y_peri = A_ * (1 - e), B_ * (1 - e)
# Apastron: E = pi
x_apo, y_apo = A_ * (-1 - e), B_ * (-1 - e)


# Set up Sky-Projected Orbit Fit and True Orbit Fit subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 7),
                               layout="constrained")
ax1.set_aspect("equal")
ax2.set_aspect("equal")

# Plot Primary Star
ax1.scatter(0, 0, color="red", s=180, marker="o", label="Primary",
            zorder=7)

# Plot Secondary Star at last observation
ax1.scatter(y_obs[-1], x_obs[-1], color="orange", s=120, marker="o",
            label="Secondary", zorder=6)

ax1.scatter(y_obs, x_obs, c="tab:blue", marker="x", s=25,
            label="Observations", zorder=5)

# Plot Periastron and Apastron on ax1
ax1.scatter(y_peri, x_peri, color="green", marker="x", s=120,
           label="Periastron", zorder=4)
ax1.scatter(y_apo, x_apo, color="purple", marker="x", s=120,
           label="Apastron", zorder=3)

# Line of nodes (intersection of orbital and sky planes)
normal = np.array([-np.sin(Omega), np.cos(Omega)])
node_vals = normal[0] * x_fit + normal[1] * y_fit
crossings = np.where(np.sign(node_vals[:-1]) != np.sign(node_vals[1:]))[0]
node_points = []
node_types = []

# Compute z coordinate along the true orbit to distinguish ascending vs descending nodes
z_fit = np.sin(i) * (np.sin(omega) * X_d + np.cos(omega) * Y_d)
for j in crossings:
    t = node_vals[j] / (node_vals[j] - node_vals[j + 1])
    x_int = x_fit[j] + t * (x_fit[j + 1] - x_fit[j])
    y_int = y_fit[j] + t * (y_fit[j + 1] - y_fit[j])
    node_points.append((x_int, y_int))
    if z_fit[j] < 0 and z_fit[j + 1] > 0:
        node_types.append("ascending")
    elif z_fit[j] > 0 and z_fit[j + 1] < 0:
        node_types.append("descending")
    else:
        node_types.append("unknown")

if len(node_points) >= 2:
    node_x = np.array([node_points[0][0], node_points[1][0]])
    node_y = np.array([node_points[0][1], node_points[1][1]])
else:
    node_x = np.array([-1 * a, 1 * a]) * np.cos(Omega) * 1.2
    node_y = np.array([-1 * a, 1 * a]) * np.sin(Omega) * 1.2
ax1.plot(node_y, node_x, "--", color="gray", lw=0.8, label="Line of nodes", zorder = 2)

# Plot Sky-Projected Orbit Fit
ax1.plot(y_fit, x_fit, "k-", lw=1.2, zorder=1)


ax1.set_xlabel(f"$\\Delta$E ({unit})")
ax1.set_ylabel(f"$\\Delta$N ({unit})")
ax1.grid(True, alpha=0.3)

fig.legend(fontsize=8, ncol=6, loc="outside lower center", frameon=False, borderaxespad=0.2)
# Plot true orbit ---------------------------------------
def polar_ellipse(a_AU, e, theta):
    return a_AU * (1 - e**2) / (1 - e * np.cos(theta-np.pi))

ax2.grid(True, alpha=0.3)

theta = np.linspace(0, 2 * np.pi, 1000)
r = polar_ellipse(a_AU, e, theta)
x_true = r * np.cos(theta)
y_true = r * np.sin(theta)

# Plot True Orbit
ax2.plot(x_true, y_true, "k-", lw=1.2, label="True Orbit", zorder=1)

# Plot Primary Star on ax2
ax2.scatter(0, 0, color="red", s=180, marker="o", label="Primary",
            zorder=5)

# Plot Periastron and Apastron on ax2
ax2.scatter(a_AU * (1 - e), 0, color="green", marker="x", s=120,
           label="Periastron", zorder=3)
ax2.scatter(a_AU * (-1 - e), 0, color="purple", marker="x", s=120,
           label="Apastron", zorder=3)

# Plot secondary star and observations on the True orbital plane
det = A_ * G_ - B_ * F_
X_obs = ( G_ * x_obs - F_ * y_obs) / det
Y_obs = (-B_ * x_obs + A_ * y_obs) / det

# Convert to AU
X_obs_AU = a_AU * X_obs
Y_obs_AU = a_AU * Y_obs

# Get true anomaly at last observation
true_anom = np.arctan2(Y_obs_AU[-1], X_obs_AU[-1])

ax2.scatter(X_obs_AU, Y_obs_AU, c="tab:blue", marker="x", s=25,
            label="Observations", zorder=4)
ax2.scatter(X_obs_AU[-1], Y_obs_AU[-1], color="orange", s=120, marker="o",
            label="Secondary", zorder=5)


ax2.set_xlabel("X (AU)")
ax2.set_ylabel("Y (AU)")

# Titles

while T <= t_obs[0] - P:
    T += P

fig.suptitle(
    f'{target} | Obs Arc: {t_obs[0]:.0f} - {t_obs[-1]:.0f} |  n = {n_starts}, R² = {r_squared}\n'
    f'Parallax = {parallax_mas:.4f} mas, Distance = {d_pc:.2f} pc, $M_{{total}}$ = {M_total:.3f} M$_\\odot$\n'
    f'P = {P:.3f} yr, T = {T:.3f} yr'
    , fontsize=12)

ax1.set_title(
    f'Sky-Projected Orbit Fit\n'
    f'a = {a:.3f}{unit}, e = {e:.3f}, '
    f'i = {np.degrees(i):.3f}$^\\circ$, $\\Omega$ = {np.degrees(Omega):.3f}$^\\circ$, $\\omega$ = {np.degrees(omega):.3f}$^\\circ$'
    f'\nApastron = {a*(1+e):.3f}{unit}, Periastron = {a*(1-e):.3f}{unit}'
    , fontsize = 11
)
ax2.set_title(
    f'True Orbit Fit\n'
    f'a = {a_AU:.3f} AU, e = {e:.3f}, $\\nu$ = {np.degrees(true_anom):.3f}$^\\circ$\n'
    f'Apastron = {a_AU*(1+e):.3f} AU, Periastron = {a_AU*(1-e):.3f} AU'
    , fontsize = 11
)

plt.savefig("orbit_fit.png", dpi=200, bbox_inches="tight")
print("Plot saved to orbit_fit.png")
