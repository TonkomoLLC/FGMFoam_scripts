This script takes the CSV FGM tables from `02_tables` and converts them into OpenFOAM dictionary files (`*_table`, `tableProperties`, `PVtableProperties`) that FGMFoam (OpenFOAM 7) can read directly, including thermo, species, transport, PV source, and PV bounds information.[1]

## Role in the workflow

In the overall pipeline:
- `build_fgm_tables.py` produced a tar/dir with 2D arrays $T(Z,PV)$, $\rho(Z,PV)$, $Y_k(Z,PV)$, `SourcePV`, and the corresponding axes and PV bounds.[2][1]
- `csv2of_tables.py` reads those CSVs (either from `02_tables.tar.xz` or a `02_tables/` folder) and writes OpenFOAM-format **table dictionaries** in the target case directory, ready for FGMFoam’s table solvers.[1]

Conceptually, it maps the continuous FGM library into the specific nested list layout and metadata (dimensions, variable parameters, tablePath, etc.) that OpenFOAM’s `tableSolver` and `PVtableSolver` expect.[1]

## Data loading and interpretation

Input handling:
- The script accepts `--in` as either the tarball (`02_tables.tar.xz`) or the extracted directory, and `--out` as the OpenFOAM case directory where tables will be written.[1]
- It loads:
  - Axes: `axes/Z.csv`, `axes/PV.csv` (1D arrays including padding).[1]
  - Thermo fields: `thermo/T.csv`, `thermo/rho.csv` (2D arrays of shape NZ×NPV).[1]
  - Species fields: all `species/Y_*.csv`, each a 2D NZ×NPV table (e.g., `Y_O2.csv` → field key `Y_O2`).[1]
  - Optional extras: `psi`, `mu`, `Cps`, `alpha`, `SourcePV` from either `thermo/*.csv` or `extras/*.csv` if present.[1]

The loader (`load_tables`) normalizes paths for tar vs directory, enforces:
- `Z` and `PV` must be 1D; `T` and `rho` must be 2D; all other fields must match shape `(NZ, NPV)`.[1]
- Optional `axes/PVmin.csv` and `axes/PVmax.csv` are read and stored as `PVmin`/`PVmax` if available.[1]

The axes are interpreted as:
- `Z`: the mixture-fraction axis, including padded endpoints (same as used to build tables).[1]
- `PV`: the **normalized** progress-variable axis, also padded slightly beyond.[3][1]

## Generation of OpenFOAM tables

### Core 4D integrated tables

OpenFOAM’s `tableSolver` expects tables in the form `List<List<List<List<scalar>>>>` indexed as `[varPV][PV][varZ][Z]`.  The script uses:[1]

- `varPV_param` and `varZ_param` (from `--varPV` and `--varZ`, default `(0 1)` for each) as **dummy outer indices**, so there are at least two entries in each “var” direction.[1]
- `Z` and `PV` arrays define the inner physical axes length NZ and NPV.[1]

`write_integrated_table_4d`:
- Checks shape consistency (array must be NZ×NPV and match `Z` and `PV` lengths).[1]
- Prints a standard OpenFOAM `FoamFile` header and dimension set.[1]
- Writes the nested list structure:
  - First dimension: `nVarPV`  
  - Second: NPV points  
  - Third: `nVarZ`  
  - Fourth: NZ values per Z-line, using the column at each PV-index.[1]

This function is called to create:
- `T_table` (temperature, with energy dimensions).[1]
- `rho_table` (density).[1]
- Species tables: one for each `Y_*` field, e.g., `Y_O2_table`, all dimensionless.[1]
- Optional extras (if present or computed): `psi_table`, `mu_table`, `Cps_table`, `alpha_table`, `SourcePV_table`.[1]

The dimension strings are set according to OpenFOAM’s unit system, e.g.:
- `T_table`: `[0 0 0 1 0 0 0]` (temperature).[1]
- `rho_table`: `[1 -3 0 0 0 0 0]` (density).[1]
- `psi_table`: `[0 -2 2 0 0 0 0]` (compressibility).[1]
- `SourcePV_table`: `[0 0 0 -1 0 0 0]` (PV source with 1/time dimension).[1]

### tableProperties and PVtableProperties

`emit_properties` writes two property dictionaries into `constant/` of the case:[1]

- `tableProperties`:
  - `tablePath` (from `--tablePath`, default `"tables"`; automatically normalized so that bare `"tables"` becomes `"constant/tables"` in the file).[1]
  - `interpolationType linearInterpolation;` for standard 2D interpolation in $(Z,PV)$.[1]
  - `varPV_param` and `varZ_param` as lists, plus `PV_param` and `Z_param` containing the actual axes arrays.[1]

- `PVtableProperties`:
  - Same `tablePath` string as provided.[1]
  - `interpolationType PVlinearInterpolation;` for PV bounds interpolation.[1]
  - `varZ_param` and `Z_param` as above (PV bounds tables are 1D over Z).[1]

These files tell FGMFoam where to find the tables and how to interpret the discrete indices as continuous Z and PV coordinates.[1]

## Thermophysical extras and SourcePV

Thermo extras:
- If `--mech` is provided or `gri30.yaml` exists, `maybe_compute_thermo_extras` tries to recompute `mu`, `Cps`, and `alpha` from Cantera using the tabulated `T` and `Y_k`, with the given pressure.[1]
- It constructs a 3D array of species mass fractions matching the mechanism’s species names, then loops over all Z,PV points to evaluate viscosity, cp, and thermal conductivity, computing `alpha = k/(ρ cp)`.[1]
- If Cantera is not available or no matching species are found, it falls back to `compute_fallback_mu_Cps_alpha`, which writes constant default values (e.g. `mu ≈ 1.8e−5`, `Cps ≈ 1100`, `alpha ≈ 2e−5`).[1]

Psi:
- If `psi_table` is missing or `--rebuild-thermo` is set, it writes `psi_table` as simple `rho/P`, using the provided `--pressure`.[1]

SourcePV:
- The script treats `SourcePV_table` as **crucial** for PV evolution: if no SourcePV 2D field was present in the input (`thermo/SourcePV.csv` or `extras/SourcePV.csv`) and no SourcePV_table has been written yet, it creates a zero-valued table and prints strong warnings.[1]
- A zero SourcePV table means PV will not grow by chemistry and only changes if imposed, which is typically not desired for reacting FGM simulations.[1]

You can also force recomputation of mu/Cps/alpha/psi via `--rebuild-thermo`, even if those tables already exist in the input.[1]

## PV bounds tables and plain-language meaning

PV bounds:
- If `--emit-pv-bounds` is set, the script **forces** normalized PV bounds: `PVmin_table = 0`, `PVmax_table = 1` for all Z, matching the normalized PV axis.[1]
- Else, if `axes/PVmin.csv` and `axes/PVmax.csv` are present, it reads them as physical PV-unit bounds per Z and writes them into `PVmin_table` and `PVmax_table` using `write_pv_table_2d`.[1]
- If bounds files are absent and `--emit-pv-bounds` is not used, it falls back to 0/1 bounds and issues a warning.[1]

In plain language:
- The script finishes the pipeline by translating the FGM CSV library into the exact OpenFOAM file format that FGMFoam understands.[1]
- After running it, your case will have:
  - `constant/tableProperties` and `constant/PVtableProperties`, telling FGMFoam where the tables are and what the Z/PV grids look like.[1]
  - A directory (e.g. `constant/tables/`) full of `*_table` files: temperature, density, compressibility, viscosity, heat capacity, thermal diffusivity, PV source, and species mass fractions, all tabulated as functions of Z and PV.[1]
  - Additional PV bounds tables `PVmin_table` and `PVmax_table` that allow the PVtableSolver to map normalized PV back to physical progress-variable values per mixture fraction.[1]

From the solver’s perspective, once these files exist, FGMFoam can reconstruct local thermochemical states at each cell, solely from the transported scalars $Z$ and $PV$, without ever calling Cantera during the CFD run.[1]

[1]csv2of_tables.py
[2]build_fgm_tables.py
[3]generate1DFlamelets.py
[4]organizeData.py
