# %% [markdown]
# # Exp2 MSD Neural ODE Fit (6 parameters)
#
# This example trains the tiniest possible neural ODE on the Exp2
# mass-spring-damper trajectory: a single dense layer without bias whose
# 2×3 weight matrix contains the only six trainable parameters. The setup
# mirrors `exp2_complextone_descent.jl` but trims away every non-essential
# helper so that the procedure is easy to follow inside Docs examples.

# %%
import matplotlib.pyplot as plt
import optax

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve

# Core Exp2 configuration.
MASS = 0.05
NATURAL_FREQ = 25.0  # Hz
DAMPING_RATIO = 0.01
SAMPLE_RATE = 300  # Hz
NUM_SAMPLES = 5  # Five samples at 300 Hz (Exp2 requirement)
SIM_DURATION = (NUM_SAMPLES - 1) / SAMPLE_RATE
INITIAL_STATE = jnp.array([0.0, 0.0])
TONE_FREQUENCIES = (20.0, 24.0)
DT = 1.0 / SAMPLE_RATE

OMEGA = 2 * jnp.pi * NATURAL_FREQ
STIFFNESS = MASS * OMEGA**2
DAMPING = 2 * DAMPING_RATIO * MASS * OMEGA


def exp2_forcing(t: float) -> float:
    """Two-tone excitation (complex tone) used in the Julia Exp2 script."""
    w1 = 2 * jnp.pi * TONE_FREQUENCIES[0]
    w2 = 2 * jnp.pi * TONE_FREQUENCIES[1]
    return jnp.sin(w1 * t) + jnp.sin(w2 * t)


def true_vector_field(t, state, args):
    """Ground-truth MSD acceleration."""
    pos, vel = state
    acc = (exp2_forcing(t) - DAMPING * vel - STIFFNESS * pos) / MASS
    return jnp.array([vel, acc])


def simulate_reference():
    """Generate the five Exp2 samples for supervision."""
    ts = jnp.linspace(0.0, SIM_DURATION, NUM_SAMPLES)
    sol = diffeqsolve(
        ODETerm(true_vector_field),
        Tsit5(),
        t0=0.0,
        t1=SIM_DURATION,
        dt0=DT,
        y0=INITIAL_STATE,
        saveat=SaveAt(ts=ts),
    )
    return sol.ts, sol.ys


class LinearMSD(eqx.Module):
    """2×3 linear map without bias -> exactly six parameters."""

    weight: jax.Array

    def __call__(self, t, state):
        force = exp2_forcing(t)
        inputs = jnp.array([state[0], state[1], force])
        return self.weight @ inputs


def solve_with_model(model: LinearMSD, ts: jax.Array):
    """Integrate the neural ODE with the shared forcing."""

    def vf(t, y, args):
        return model(t, y)

    sol = diffeqsolve(
        ODETerm(vf),
        Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=DT,
        y0=INITIAL_STATE,
        saveat=SaveAt(ts=ts),
    )
    return sol.ys


ts_ref, ys_ref = simulate_reference()

# Initialize the 2x3 matrix close to the analytical system for fast training.
init_weight = jnp.array(
    [
        [0.0, 1.0, 0.0],  # dx/dt ≈ v
        [-STIFFNESS / MASS, -DAMPING / MASS, 1.0 / MASS],
    ],
    dtype=jnp.float64,
)
model = LinearMSD(weight=init_weight + 0.01 * jnp.ones_like(init_weight))

optimizer = optax.adam(1e-2)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


def loss_fn(model):
    pred = solve_with_model(model, ts_ref)
    return jnp.mean((pred - ys_ref) ** 2)


loss_and_grad = eqx.filter_value_and_grad(loss_fn)


@eqx.filter_jit
def train_step(model, opt_state):
    loss, grads = loss_and_grad(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


loss_history = []
for step in range(400):
    model, opt_state, loss = train_step(model, opt_state)
    loss_history.append(float(loss))

trained_weight = model.weight
predicted_traj = solve_with_model(model, ts_ref)

# %%
print("Trained weight matrix (2x3):")
print(trained_weight)
print(f"Final MSE loss: {loss_history[-1]:.3e}")

# %%
plt.figure(figsize=(8, 4))
plt.plot(ts_ref, ys_ref[:, 0], "o-", label="position (target)")
plt.plot(ts_ref, predicted_traj[:, 0], "x--", label="position (NN)")
plt.plot(ts_ref, ys_ref[:, 1], "o-", label="velocity (target)")
plt.plot(ts_ref, predicted_traj[:, 1], "x--", label="velocity (NN)")
plt.xlabel("Time [s]")
plt.ylabel("State")
plt.title("Six-parameter Neural ODE fit to Exp2 MSD data")
plt.legend()
plt.tight_layout()
plt.show()

# %%
plt.figure()
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("MSE loss")
plt.title("Training curve")
plt.tight_layout()
plt.show()
