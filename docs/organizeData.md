This script post-processes each 1D flamelet, converts it into mixture-fraction space, and writes a compact CSV containing geometry, flow, mixture fraction, a progress variable, its chemical source term, and species mass fractions for later tabulation or modeling.

## Theory and derived quantities

The script maps each spatial flamelet solution into:
- Bilger mixture fraction \(Z\), computed from elemental C/H/O mass fractions using a standard linear combination.
- A scalar progress variable \(PV\) built as a weighted sum of mole numbers of H\(_2\)O, CO\(_2\), H\(_2\), and CO, representing how “burned” the mixture is.

For a given state \((T, P, Y)\), the Bilger combination \(b\) is formed from elemental mass fractions \(Y_C, Y_H, Y_O\) as  
\(b = 2\,Y_C/M_C + 0.5\,Y_H/M_H - Y_O/M_O\), and mixture fraction is  
\(Z = (b - b_o)/(b_f - b_o)\), clipped to \([0,1]\), where \(b_f\) and \(b_o\) are the fuel and oxidizer stream values.  The progress variable is
\(PV = 4\,n_{H_2O} + 2\,n_{CO_2} + 0.5\,n_{H_2} + 1\,n_{CO}\), with \(n_k = Y_k/\text{MW}_k\) in kmol/kg.

To characterize mixing and reaction:
- The script computes the axial strain-like quantity \(S_n\) as the derivative \(du/dz\) on a uniform grid, and a proxy scalar dissipation \(\chi_Z = 2 (dZ/dz)^2\), using first-order finite differences.
- It evaluates a **chemical source term for the progress variable**,  
  \( \dot{PV}_\text{chem} = (4\,\dot{\omega}_{H_2O} + 2\,\dot{\omega}_{CO_2} + 0.5\,\dot{\omega}_{H_2} + 1\,\dot{\omega}_{CO}) / \rho\), where \(\dot{\omega}_k\) are net production rates from Cantera and \(\rho\) is density.

These quantities allow you to frame each flamelet in the standard flamelet variables \(Z\), \(\chi_Z\), and \(PV\)/\(\dot{PV}\) for turbulent combustion modeling.

## Script flow and data handling

Key steps for each file:
- It builds a `ct.Solution` for `gri30.yaml` and defines reference fuel (`CH4:1`) and oxidizer (`O2:0.21, N2:0.79`) streams, then precomputes their Bilger values \(b_f\) and \(b_o\).
- It collects all flamelet files matching `strain_loop_*.yaml` (or `.yml`/`.h5` etc.), sorted by name; if none are found, it raises an error.

Robust input loading:
- For each file, it attempts to restore a `SolutionArray` named `diff1D`; if that is empty, it tries unnamed groups; if still empty, it falls back to reading a `.csv` with the same basename.
- It also tries to get a pandas `DataFrame` view (`sa.to_pandas()`), which exposes any “extra” columns like grid, velocity, or spread rate if present.

State extraction:
- Uses `sa.T` and `sa.Y` for temperature and species mass fractions (reliable for both YAML and CSV-based arrays).
- Determines pressure from `sa.P` if available, else uses `P_default = 101325 Pa`.
- For density, it uses `sa.density` if present; otherwise it recomputes \(\rho\) pointwise by setting `gas.TPY` and querying `gas.density`.

Geometry and flow:
- Grid coordinate \(z\) is pulled by trying several column names (`'grid','x','z','position','distance'`); if none exist, it falls back to a normalized surrogate grid from 0 to 1.
- Axial velocity `u` is taken from any of `'velocity','u','axial_velocity'`; transverse/spread rate `V` from `'spread_rate','V','radial_velocity','transverse_velocity'`.
- If `u` or `V` cannot be found, they default to zeros, ensuring arrays are always defined.

Derivatives are then taken on this \(z\)-grid using `derivative_uniform_grid`, which assumes a strictly increasing 1D coordinate and computes simple forward/backward differences.

## Output structure and variables

For each input flamelet, the script writes a single CSV with a front-loaded set of derived scalars followed by all species mass fractions:

- Output filename:  
  `post_strain_loop_XX.csv`, in the same directory as the input, with conversion of `.yaml/.yml/.h5/.hdf5/.hdf` to `.csv`.
- Header columns, in order:  
  1. `z` – spatial coordinate (either true grid from Cantera or normalized  surrogate).[2]
  2. `u` – axial velocity component along the 1D coordinate.
  3. `V` – transverse or spread-rate component, if present.
  4. `T` – temperature.
  5. `rho` – density.
  6. `Z` – Bilger mixture fraction.
  7. `PV` – progress variable.
  8. `SourcePV` – chemical source term of the progress variable (kmol/kg/s).
  9. `chiZ` – scalar-dissipation proxy \(2 (dZ/dz)^2\).
  10. `S_n` – normal strain rate \(du/dz\).
  11+. `Y.<species>` – mass fraction of each species in the mechanism, e.g., `Y.O2`, `Y.N2`, `Y.CH4`, etc., for all species in `gas.species_names`.

The data are written via `np.savetxt` as comma-separated values with the header line describing all the columns.

## Plain-language meaning of the outputs

In everyday terms, each `post_*.csv` file transforms one raw flamelet into a **compact dataset** that tells you, at each point across the flame:

- Where you are between the fuel and air streams: the **mixture fraction `Z`** runs from 0 (pure oxidizer side) to 1 (pure fuel side), so you can interpret each row as “this is a point with this specific fuel–air mix.”
- How far the chemistry has progressed: the **progress variable `PV`** is small where gases are mostly unreacted and large where products like water and carbon dioxide dominate.
- How intense the local mixing is: the **scalar dissipation proxy `chiZ`** tells you where mixture fraction is changing sharply, which is where mixing “beats up” the flame and tends to promote extinction.
- How strong the flame chemistry is locally: the **source term `SourcePV`** shows where chemistry is actively creating products; large positive values mark the reaction zone, while near-zero values correspond to frozen regions upstream or downstream of the flame.

Together, these outputs give a clean, uniform format across all flamelets that is ready for:
- Building flamelet libraries (tabulating `T`, `Y_k`, `PV`, `SourcePV` versus `Z` and `chiZ`).
- Comparing how the flame structure changes with strain across the sequence of `post_strain_loop_XX.csv` files, without having to re-parse Cantera-specific formats.

