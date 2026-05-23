import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


unit = "mas" # milliarcseconds

# Read obs.csv
data = np.genfromtxt('obs.csv', delimiter=',', skip_header=1)

# Get x and y columns
x = data[:, 0]
y = data[:, 1]

# Convert to polar coordinates (origin = primary star = focus)
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# --- Orbit model: ellipse in polar form with focus at origin ---
# r = a(1 - e^2) / (1 + e*cos(theta - omega))
def orbit_model(theta, a, e, omega):
    return a * (1 - e**2) / (1 + e * np.cos(theta - omega))

# Initial guesses: semi-major axis, eccentricity, argument of periapsis
p0 = [np.mean(r), 0.5, 0.0]
bounds = ([0, 0, -np.pi], [np.inf, 0.9999, np.pi])

popt, pcov = curve_fit(orbit_model, theta, r, p0=p0, bounds=bounds)
a_fit, e_fit, omega_fit = popt


# --- Plot ---
theta_dense = np.linspace(0, 2 * np.pi, 1000)
r_fit = orbit_model(theta_dense, *popt)
x_fit = r_fit * np.cos(theta_dense)
y_fit = r_fit * np.sin(theta_dense)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot fitted orbit
ax.plot(x_fit, y_fit, color='black', label='Fitted orbit', zorder=1)

# Plot fitted periapsis and apoapsis points
peri_x = a_fit * (1 - e_fit) * np.cos(omega_fit)
peri_y = a_fit * (1 - e_fit) * np.sin(omega_fit)
apo_x = a_fit * (1 + e_fit) * np.cos(np.pi + omega_fit)
apo_y = a_fit * (1 + e_fit) * np.sin(np.pi + omega_fit)
ax.scatter(peri_x, peri_y, color='green', marker='+', s=100, label='Periapsis', zorder=2)
ax.scatter(apo_x, apo_y, color='orange', marker='+', s=100, label='Apoapsis', zorder=3)

# Plot observations
ax.scatter(x, y, color='blue', marker = "x",zorder=4, label='Observations', s=20) 

# Plot primary star at origin
ax.scatter(0, 0, color='red', s=200, marker='o', zorder=5, label='_Hidden Item')

# Get total residual for fitted orbit R^2
r_obs_fit = orbit_model(theta, *popt)
residuals = r - r_obs_fit
ss_res = np.sum(residuals**2)
ss_tot = np.sum((r - np.mean(r))**2)
r_squared = 1 - (ss_res / ss_tot)

ax.set_xlabel(f'X ({unit})')
ax.set_ylabel(f'Y ({unit})')
ax.set_title(f'Fitted Orbit (Sky-Projected Plane)\nR² = {r_squared:.4f}\na = {a_fit:.3f} {unit}, e = {e_fit:.3f}, ω = {np.degrees(omega_fit):.3f} deg\nPeriapsis = {a_fit*(1-e_fit):.3f} {unit}, Apoapsis = {a_fit*(1+e_fit):.3f} {unit}')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.savefig('orbit_fit.png', dpi=750, bbox_inches='tight')


print(f"Fitted orbital parameters:")
print(f"  Semi-major axis  a = {a_fit:.4f} {unit}")
print(f"  Eccentricity     e = {e_fit:.4f}")
print(f"  Arg. of peri.    ω = {np.degrees(omega_fit):.4f} deg")
print(f"  Periapsis dist.    = {a_fit*(1-e_fit):.4f} {unit}, ({peri_x:.4f}, {peri_y:.4f}) {unit}")
print(f"  Apoapsis dist.     = {a_fit*(1+e_fit):.4f} {unit}, ({apo_x:.4f}, {apo_y:.4f}) {unit}")

print("Plot saved to orbit_fit.png")