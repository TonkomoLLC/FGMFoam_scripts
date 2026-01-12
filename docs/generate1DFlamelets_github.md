This script generates a sequence of 1D counterflow diffusion flame “flamelets” by solving the steady conservation equations for mass, momentum, species, and energy in a planar opposed-flow configuration, while progressively increasing the inlet mass flux (a proxy for strain rate) until extinction.

## Physical model and theory

The code uses Cantera’s `CounterflowDiffusionFlame` model, which represents a steady, laminar diffusion flame between opposing fuel and oxidizer streams.  The coordinate is one-dimensional (normal to the stagnation plane), but the underlying similarity solution corresponds to a 2D opposed jet flow that has been reduced to an effective 1D problem.

At each strain level, the governing equations enforce:
- Overall continuity (constant mass flux across the domain) with varying density from temperature and composition changes.
- Species conservation for all gas-phase species in `gri30.yaml` with convection and diffusion, plus reaction source terms from detailed chemistry.
- Energy conservation including convective and diffusive heat transport, and chemical heat release via enthalpy changes in the reaction network.

The transport model is mixture-averaged or multicomponent, so diffusion fluxes and thermal conduction are treated consistently with the chosen option.  The counterflow **strain** is controlled indirectly through the total inlet mass flux (sum of fuel and oxidizer mass fluxes), which the script ramps up multiplicatively.

## Governing equations (conceptual form)

In simplified form, the model solves along the axial coordinate $z$:

- Continuity (steady 1D):
  - $$ \frac{d}{dz}(\rho u) = 0$$, so the mass flux $ \rho u $ is constant across the domain.

- Species $Y_k$ for each species $k$:
  - $$\rho u \frac{dY_k}{dz} = - \frac{d j_k}{dz} + \dot{\omega}_k W_k$$, where $j_k$ is the diffusive mass flux and $\dot{\omega}_k$ is the molar production rate from chemistry.

- Energy (in terms of temperature $T$ or mixture enthalpy):
  - $$\rho u c_p \frac{dT}{dz} = - \frac{dq}{dz} - \sum_k h_k \frac{d j_k}{dz}$$, where $q$ is conductive heat flux and $h_k$$ are species enthalpies.

The velocity and strain appear via the similarity formulation inside `CounterflowDiffusionFlame`; for the user, the “knob” is the inlet mass fluxes that set the effective strain rate.

## Script logic and numerical strategy

### Initialization and base flame ignition

- The script defines fuel (`CH4:1`), oxidizer (`O2:0.21, N2:0.79`), mechanism (`gri30.yaml`), domain width, pressure, and inlet mass fluxes and temperatures.
- It creates a `ct.Solution` and a `ct.CounterflowDiffusionFlame(gas, width=width)`, sets the transport model, pressure, and boundary conditions at fuel and oxidizer inlets (mdot, T, composition).

Ignition strategy:
- A Gaussian temperature “hotspot” is imposed in the interior of the domain using `seed_hotspot`, which sets a temperature profile peaking at ~1700 K and centered around position 0.55 (slightly off-center) in normalized coordinates.
- The flame is then solved; if the maximum temperature exceeds `T_lit_threshold` (800 K), it is considered lit.
- If that fails, the script briefly preheats the oxidizer inlet (900 → 700 → 500 → 300 K) and re-seeds a hotspot, trying to reach a clearly burning solution.
- If ignition still fails, it reduces total mass flux by 40% once and repeats the ignition routine.

Once a lit base solution is found, the script recenters the flame by adjusting only the **ratio** of oxidizer to fuel mass flux at fixed total mass flux, using a small line search (`recenter_line_search`) so that the maximum temperature location is near 0.55 of the domain.  This helps keep the flame away from boundaries during the continuation.

### Strain continuation and extinction detection

With the base flame lit and recentered:
- The script saves the base flame as `strain_loop_00.yaml` and `strain_loop_00.csv`.
- It then enters a loop (up to `max_steps`) where each iteration attempts to increase the total mass flux by a factor `strain_factor` (e.g., 1.10).

Adaptive stepping (`adaptive_step_solve`):
- The code computes the current total mass flux
$$M_0$ and inflow mass flux ratio $$
r_0 = \dot{m}_{ox}/\dot{m}_{fuel}
$$.
- It tries a series of fractional steps (1.0, 0.8, 0.65, 0.5, 0.35) between the current state and the target increase.
- For each attempt, it scales the total mass flux by a factor `sf` while preserving the ratio $r_0$ and solves the flame, using `set_refine_criteria` to adjust grid refinement.
- A step is accepted if the solve converges and the new maximum temperature exceeds `min_T_accept` (1200 K), which indicates a robustly burning flame; otherwise, the code tries a smaller effective step.

If the primary increase (to the target strain factor) cannot be achieved while keeping the flame lit:
- The script tries a “fine ladder” of smaller multiplicative increases (1.06, 1.04, 1.03, 1.02) near the limit.
- If none of these produce a clearly lit solution, the script declares extinction around that step.

After each successful step:
- The code recenters again using `recenter_line_search` at the new mass flux, then checks `is_lit`.
- If the flame is still lit, it saves the new flamelet and increments the step counter; if not, it saves the marginal/extinct state once, reports extinction, and exits the loop.

At the end, a message reports whether only the base flame is lit or the index of the last lit flamelet (`strain_loop_NN.yaml`).

## Output files and structure

For each strain level (step index `NN`):
- The script writes:
  - `strain_loop_NN.yaml` – a Cantera input/state file containing the full 1D flame solution.
  - `strain_loop_NN.csv` – a CSV export of the same solution, with profiles sampled along the grid.

The YAML file:
- Contains the `ct.Solution` object and the 1D flame state (grid, species, T, etc.) under the name `diff1D`.
- Can be reloaded later with Cantera’s `SolutionArray` or `ct.Solution`+`ct.Sim1D` tools to reconstruct the flamelet.

The CSV file:
- Stores the **spatial profiles** along the 1D grid. Typical columns include:
  - Position $z$ in meters.
  - Temperature $T(z)$.
  - Mass fractions $Y_k(z)$ for each species in the mechanism.
  - Possibly density, velocity, and other state variables depending on Cantera’s CSV export.
- Each row corresponds to one grid point; the grid is adaptively refined where gradients in T or species are steep, so the spacing is nonuniform and denser around the flame zone.

The file naming:
- `strain_loop_00` is the base flamelet at the lowest strain (after any mass-flux reduction used to ignite).
- `strain_loop_01`, `strain_loop_02`, … are successive higher-strain solutions along the continuation path, up to extinction.

## Plain-language interpretation of the outputs

In plain terms, each saved pair (`strain_loop_NN.yaml` / `.csv`) is one **snapshot** of a 1D methane–air diffusion flame held between opposing jets, at a specific “burning intensity” set by how hard the two streams are pushed toward each other (total mass flux / strain).

What each flamelet tells you:
- Where the flame sits: The temperature profile shows where the hot reaction zone is located between the fuel and oxidizer inlets; the script actively keeps that peak near the center of the domain so you don’t hit boundary effects.
- How strong the flame is: The peak temperature and thickness of the high-temperature region indicate how vigorously the flame is burning at that strain level.
- Mixture structure: Species profiles reveal how fuel, oxidizer, radicals, intermediates, and products change from the fuel side, through the flame front, to the oxidizer side.

As you go from `strain_loop_00` to higher indices:
- The inlet jets become “stronger,” increasing the stretch the flame experiences.
- The flame typically thins and may eventually weaken, with lower peak temperatures and more pronounced gradients.
- At some step, the flame can no longer sustain combustion under the high strain, and the solution becomes “cold” (no high-temperature peak); the script flags this as extinction and stops.

From a modeling perspective, this flamelet sequence is suitable for:
- Building a flamelet library for turbulent combustion models (e.g., tabulating species and T as functions of mixture fraction, scalar dissipation, etc.).
- Studying extinction and stabilization mechanisms in counterflow diffusion flames by examining how reaction zones and key radicals respond to increasing strain.


