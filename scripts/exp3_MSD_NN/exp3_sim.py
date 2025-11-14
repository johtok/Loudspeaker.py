# %% [markdown]
# # Exp2 Mass-Spring-Damper Simulation (Diffrax)
#
# This stripped-down example reproduces the Exp2 mass-spring-damper (MSD)
# configuration using only the core pieces that matter: the MSD dynamics, a
# complex-tone forcing signal, and a Diffrax solver. Everything else from the
# large research scripts is omitted.

# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve

# Physical parameters taken from Exp2 / Loudspeaker.jl.
MASS = 0.05  # kg
NATURAL_FREQ = 25.0  # Hz
DAMPING_RATIO = 0.01
SAMPLE_RATE = 300  # Hz
NUM_SAMPLES = 5  # Exp2 uses only five samples at 300 Hz
SIM_DURATION = (NUM_SAMPLES - 1) / SAMPLE_RATE
INITIAL_STATE = jnp.array([0.0, 0.0])  # position, velocity
TONE_FREQUENCIES = (20.0, 24.0)  # Complex tone used in the Julia Exp2 script

OMEGA = 2 * jnp.pi * NATURAL_FREQ
STIFFNESS = MASS * OMEGA**2
DAMPING = 2 * DAMPING_RATIO * MASS * OMEGA


def exp2_forcing(t: float) -> float:
    """Two-tone excitation that mirrors `TestSignals.ComplexSine`."""
    w1 = 2 * jnp.pi * TONE_FREQUENCIES[0]
    w2 = 2 * jnp.pi * TONE_FREQUENCIES[1]
    return jnp.sin(w1 * t) + jnp.sin(w2 * t)


def msd_vector_field(t, state, args):
    """Position/velocity dynamics for the lightly damped MSD."""
    pos, vel = state
    force = exp2_forcing(t)
    acc = (force - DAMPING * vel - STIFFNESS * pos) / MASS
    return jnp.array([vel, acc])


def simulate_exp2_msd():
    """Integrate the MSD with the Exp2 forcing at 300 Hz sampling."""
    ts = jnp.linspace(0.0, SIM_DURATION, NUM_SAMPLES)
    term = ODETerm(msd_vector_field)
    solver = Tsit5()
    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=SIM_DURATION,
        dt0=1.0 / SAMPLE_RATE,
        y0=INITIAL_STATE,
        saveat=SaveAt(ts=ts),
    )
    return sol.ts, sol.ys


ts, ys = simulate_exp2_msd()
position = ys[:, 0]
velocity = ys[:, 1]

# %%
plt.figure(figsize=(8, 4))
plt.plot(ts, position, label="position (m)")
plt.plot(ts, velocity, label="velocity (m/s)")
plt.title("Exp2 MSD trajectory driven by a complex tone")
plt.xlabel("Time [s]")
plt.ylabel("State")
plt.legend()
plt.tight_layout()
plt.show()
