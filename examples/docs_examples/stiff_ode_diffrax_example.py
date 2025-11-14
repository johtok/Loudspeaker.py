# %% [markdown]
# # Stiff ODE

# %% [markdown]
# This example demonstrates the use of implicit integrators to handle stiff dynamical systems. In this case we consider the Robertson problem.
# 
# This example is available as a Jupyter notebook [here](https://github.com/patrick-kidger/diffrax/blob/main/docs/examples/stiff_ode.ipynb).

# %%
import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp

# %% [markdown]
# Using 64-bit precision is important when solving problems with tolerances of `1e-8` (or smaller).

# %%
jax.config.update("jax_enable_x64", True)

# %%
class Robertson(eqx.Module):
    k1: float
    k2: float
    k3: float

    def __call__(self, t, y, args):
        f0 = -self.k1 * y[0] + self.k3 * y[1] * y[2]
        f1 = self.k1 * y[0] - self.k2 * y[1] ** 2 - self.k3 * y[1] * y[2]
        f2 = self.k2 * y[1] ** 2
        return jnp.stack([f0, f1, f2])

# %% [markdown]
# One should almost always use adaptive step sizes when using implicit integrators. This is so that the step size can be reduced if the nonlinear solve (inside the implicit solve) doesn't converge.
# 
# Note that the solver takes a `root_finder` argument, e.g. `Kvaerno5(root_finder=VeryChord())`. If you want to optimise performance then you can try adjusting the error tolerances, kappa value, and maximum number of steps for the nonlinear solver.

# %%
@jax.jit
def main(k1, k2, k3):
    robertson = Robertson(k1, k2, k3)
    terms = diffrax.ODETerm(robertson)
    t0 = 0.0
    t1 = 100.0
    y0 = jnp.array([1.0, 0.0, 0.0])
    dt0 = 0.0002
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]))
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return sol

# %% [markdown]
# Do one iteration to JIT compile everything. Then time the second iteration.

# %%
main(0.04, 3e7, 1e4)

start = time.time()
sol = main(0.04, 3e7, 1e4)
end = time.time()

print("Results:")
for ti, yi in zip(sol.ts, sol.ys):
    print(f"t={ti.item()}, y={yi.tolist()}")
print(f"Took {sol.stats['num_steps']} steps in {end - start} seconds.")


