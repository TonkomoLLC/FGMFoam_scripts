This script assembles a 2D flamelet-generated manifold (FGM) library in \((Z,PV)\) space from all `post_strain_loop_*.csv` files, and packages it into a compressed `02_tables.tar.xz` archive containing axes, thermo tables, species tables, and metadata for use in FGMFoam or similar solvers.[1]

## Theory and table construction

The tables are constructed in the 2D space of:
- **Mixture fraction \(Z\)** (Bilger, already computed in `post_*.csv`) and  
- **Normalized progress variable \(PV\)**, where raw \(PV\) is linearly rescaled so that its global minimum maps to 0 and a high percentile (default 99.5th) maps to 1, then clipped to \([0,1]\).[1]

For each scalar \(f\) (e.g., \(T\), \(\rho\), species mass fraction), the script builds a **Favre-averaged** table \( \tilde{f}(Z_i, PV_j) \) by binning all points from all flamelets into a 2D grid and computing mass-weighted averages using density as the weight.  Cells with too few samples (less than `MIN_BIN_COUNT`) remain NaN initially and are later filled via nearest-neighbor in index space, then **constant-extrapolated** into the padded boundary rows/columns so that interpolation never sees uninitialized values.[1]

To avoid interpolation edge issues in OpenFOAM:
- The \(Z\) and normalized \(PV\) axes are **padded slightly beyond **: an axis of length \(N\) is built as \([-ε, 0..1, 1+ε]\), with interior points evenly spaced from 0 to 1.[2][1]
- After binning and filling, the script copies interior edge rows/columns into the padded boundaries, ensuring smooth extrapolation just outside the physical domain and preventing `upperBound()` from returning invalid indices.[1]

If a consistent progress-variable source column (e.g., `SourcePV`, `omegaPV`, `PVdot`) exists in all input files, it is also tabled as a thermo field; otherwise, a zero table is written with warnings.[1]

## Script flow and robustness strategy

Inputs and column consistency:
- It gathers all `post_strain_loop_*.csv` files; if none are found, it exits.[1]
- In a first pass, each file is scanned to confirm required columns `Z`, `PV`, `rho`, `T` exist and to determine the **intersection of species columns** (`Y.*`) across all files, preserving the order from the first file.[1]
- It also determines which (if any) candidate `SOURCEPV_CANDIDATES` column is present in **every** file; the first such candidate in priority order is chosen as the `SourcePV` field.[1]

Concatenation and normalization:
- In a second pass, the script reads only necessary columns: base (`Z`, `PV`, `rho`, `T`), all common species `Y.*`, and the chosen `SourcePV` column if available, then concatenates them into a single large `DataFrame`.[1]
- It clamps `Z` to \([0,1]\) and computes normalized `PV` from raw `PV` by:
  - Finding `pv_lo = min(PV_raw)` and `pv_hi = percentile(PV_raw, PV_HI_PERCENTILE)`.[1]
  - Mapping \(PV = \mathrm{clip}((PV_{raw}-pv_{lo})/pv_{span}, 0, 1)\), where `pv_span = pv_hi - pv_lo`; if `pv_span <= 0`, `PV` is set to zero.[1]
- Density is sanitized by enforcing positivity and replacing non-finite values with a small floor (`RHO_FLOOR`), stabilizing Favre weights.[1]
- If a source column exists, it is sanitized (NaNs → 0); otherwise `SourcePV` is set to `None` and later replaced by a zero table.[1]

Grid and binning:
- The padded axes `Zgrid` and `PVgrid` of length `NZ` and `NPV` are built via `_build_padded_axis`, yielding interior 0–1 plus two padded points.[1]
- Physical PV bounds (`pv_lo`, `pv_hi`) are stored as vectors `PVmin_vec` and `PVmax_vec` aligned with the Z axis for later reconstruction of raw PV if needed.[1]
- For each table (T, ρ, SourcePV, and each species), `_favre_bin_2d`:
  - Determines bin centers on each axis and assigns each sample point to a bin index in Z–PV.[1]
  - Accumulates density weights `ρ` and `ρ f` with `np.bincount`, forming denominator and numerator arrays plus a count array.[1]
  - Fills bins where counts ≥ `MIN_BIN_COUNT` and denominator > 0 by `num/den`; others remain NaN.[1]

NaN handling and padding:
- `_nearest_fill` replaces remaining NaNs by nearest-neighbor values in index space, using the closest known cell in the 2D table.[1]
- `_pad_extrapolate_const` copies the first interior row/column outwards to the padded edges, ensuring **constant extrapolation** in the padding region.[1]

This process yields smooth, fully populated 2D tables without holes or uninitialized padding, preventing FGMFoam’s linear interpolation from encountering NaNs or out-of-bounds indices.[1]

## Output archive structure

The script writes a compressed tarball `02_tables.tar.xz` containing a directory tree `02_tables/` with:[1]

- `axes/`:
  - `Z.csv` – column vector of padded mixture-fraction axis values (length `NZ`).[1]
  - `PV.csv` – column vector of padded normalized progress-variable axis values (length `NPV`).[1]
  - `PVmin.csv` – column vector of the physical raw PV minimum (`pv_lo`) per Z-row (currently constant).[1]
  - `PVmax.csv` – column vector of the physical raw PV high percentile (`pv_hi`) per Z-row (currently constant).[1]

- `thermo/`:
  - `T.csv` – 2D array of Favre-averaged temperature \(T(Z_i, PV_j)\), with padding included.[1]
  - `rho.csv` – 2D array of Favre-averaged density \(\tilde{\rho}(Z_i, PV_j)\).[1]
  - `SourcePV.csv` – 2D array of Favre-averaged progress-variable source \(\tilde{\dot{PV}}(Z_i, PV_j)\); either from the selected source column or all zeros if none was available.[1]

- `species/`:
  - For each species in the common intersection, a file `Y_<name>.csv` holding the 2D table of Favre-averaged mass fraction \(\tilde{Y}_k(Z_i, PV_j)\) (e.g., `Y_O2.csv`, `Y_CH4.csv`).[1]

- `metadata.yaml`:
  - A JSON-encoded metadata dictionary with:
    - `schema` (e.g., `"FGM-2D-Z-PV"`), table `shape`, and relative paths to axes and field files.[1]
    - `thermo` and `species` mappings from field names to file paths.[1]
    - `build` parameters: `MIN_BIN_COUNT`, `PAD_EPS`, `PV_HI_PERCENTILE`, `rho_floor`, raw PV min and high percentile, and the chosen `sourcepv_col` (or `None`).[1]
    - `source`: list of all input `post_strain_loop_*.csv` files used to build the tables.[1]

All CSVs are written with `np.savetxt` in scientific notation and comma delimiters, and all are included directly inside the tarball via an in-memory `BytesIO` buffer.[1]

## Plain-language meaning and how to use the tables

In simple terms, `build_fgm_tables.py` takes all your detailed 1D flame profiles, merges them, and distills them into a **look-up library** indexed only by mixture fraction and progress variable.[1]

What the library represents:
- For any point in a CFD simulation, if the FGM model knows its local **Z** and **normalized PV**, it can:
  - Look up the **temperature** and **density** from `thermo/T.csv` and `thermo/rho.csv`.[1]
  - Look up the **mass fraction of each species** from the corresponding `species/Y_*.csv`.[1]
  - Optionally look up the **reaction source term of PV** from `thermo/SourcePV.csv` to close the PV transport equation.[1]

Why the padding and NaN handling matter:
- Padding the axes a tiny bit beyond, and filling/extrapolating all NaNs, ensures the CFD solver’s interpolation routines always see valid data even when Z or PV are slightly outside their ideal range or hit the numerical limits of the grid.[2][1]
- This design directly targets and avoids common segfaults in `linearInterpolation` caused by empty bins, NaNs, or index bounds at the edges.[1]

From a modeling perspective, after this step you no longer need to run Cantera during CFD; the chemistry is encapsulated in `02_tables.tar.xz`, which can be loaded once and sampled cheaply during the simulation based purely on scalar fields \(Z\) and \(PV\).[1]

[1]build_fgm_tables.py
[2]generate1DFlamelets.py
[3]organizeData.py
