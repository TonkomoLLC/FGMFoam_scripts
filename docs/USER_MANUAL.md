# User Manual: Generate and Run Premixed FGM Tables with OpenFOAM 7 `FGMFoam`

## `FGMScripts_OF7_premixed_v6_zeroVarPV`

**Target solver:** OpenFOAM 7 `FGMFoam`  
**Validated case:** `03_testcase`  
**Validated operation:** premixed CH₄/air tables with `useProgressVariableVariance true` at zero `varPV` and zero `varZ`

---

## 1. What this package does

This package builds tables for the OpenFOAM 7 `FGMFoam` solver using premixed methane/air flamelets generated with Cantera.

The workflow supports:

- premixed `FreeFlame` flamelet generation;
- table construction on the case-specific \(Z\) and scaled progress-variable axes;
- OpenFOAM 7 nested table-file writing;
- the auxiliary tables required by:
  ```foam
  useProgressVariableVariance true;
  ```
- validation of the zero-variance lookup path;
- in-place upgrading of an already working premixed table installation.

### Current modeling scope

The generated table files have the full four-dimensional solver storage format, but the stored variance slices are replicated. Use this version only with:

```foam
varPV internalField uniform 0;
varZ  internalField uniform 0;
```

It is not yet a nonzero presumed-PDF variance database.

---

## 2. Required software

The following software must be available in the shell environment used to build or run the case:

| Requirement | Purpose |
|---|---|
| OpenFOAM 7 with compiled `FGMFoam` | CFD solver |
| Python 3 | Table workflow scripts |
| Cantera 3.x | Premixed flamelet generation and restoration |
| NumPy | Numerical table processing |
| pandas | Flamelet CSV processing |
| `unzip`, `tar`, standard shell tools | Package installation and file management |

Check Python/Cantera availability:

```bash
python3 - <<'PY'
import cantera as ct
import numpy as np
import pandas as pd
print("Cantera:", ct.__version__)
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
PY
```

---

## 3. Directory assumptions

The commands in this manual assume the following working layout:

```text
/home/eadaymo/OpenFOAM/eadaymo-7/run/OF7_FGM_run_workflow/
├── 03_testcase/
├── FGMScripts_OF7_premixed_v6_zeroVarPV/
├── gri30.yaml
└── optional generated directories
```

Move into the run/workflow directory:

```bash
cd /home/eadaymo/OpenFOAM/eadaymo-7/run/OF7_FGM_run_workflow
```

Unpack the package:

```bash
unzip /path/to/FGMScripts_OF7_premixed_v6_zeroVarPV.zip
chmod +x FGMScripts_OF7_premixed_v6_zeroVarPV/*.sh
```

---

## 4. Choose a workflow

Use one of the following three workflows.

| Situation | Recommended workflow |
|---|---|
| Current premixed tables already ignite successfully | Upgrade existing installed tables; no Cantera rerun |
| Premixed flamelet YAML files already exist | Rebuild/install tables from existing premixed flamelets |
| Starting from nothing or changing mechanism/mixture inputs | Generate premixed flamelets and tables from scratch |

For the currently working `03_testcase`, use **Workflow A** first.

---

# Workflow A: Upgrade the already working premixed tables

## 5. In-place upgrade to `useProgressVariableVariance true`

This is the recommended immediate path because it leaves the successfully igniting `T_table`, `SourcePV_table`, `PVmin_table`, and `PVmax_table` unchanged.

Run from the directory containing `03_testcase`:

```bash
./FGMScripts_OF7_premixed_v6_zeroVarPV/upgrade_current_working_tables_to_zeroVarPV_true.sh
```

### Actions performed

The script:

1. reads the installed working premixed tables;
2. verifies that the existing variance slices are replicated;
3. makes a timestamped backup of `03_testcase/constant/tables`;
4. creates:
   ```text
   YWI_table
   YuWI_table
   YbWI_table
   Yu2I_table
   YuYbI_table
   Yb2I_table
   ```
5. changes the case setting to:
   ```foam
   useProgressVariableVariance true;
   ```
6. validates the zero-variance identities and verifies `0/varPV` and `0/varZ`.

### Expected output

Expected terminal output includes:

```text
[OK] Wrote YWI, YuWI, YbWI, Yu2I, YuYbI, and Yb2I.
[OK] Set useProgressVariableVariance true.
[OK] Complete useProgressVariableVariance table interface is present.
[OK] Zero-variance identities: ...
[OK] Replicated variance slices: maximum spread=0.000e+00
[LIMITATION] Validated for varPV=0 and varZ=0 only; not a nonzero presumed-PDF database.
```

---

# Workflow B: Rebuild tables from existing premixed flamelets

## 6. Required input files

This workflow requires premixed, not diffusion-flame, YAML files:

```text
./1DPremixedFlameletFiles/premixed_manifest.csv
./1DPremixedFlameletFiles/premixed_Z_*.yaml
```

Do not use files named like:

```text
strain_loop_*.yaml
```

Those correspond to the discarded counterflow diffusion-flame approach.

## 7. Run table processing and installation

```bash
./FGMScripts_OF7_premixed_v6_zeroVarPV/make_03_testcase_tables_from_existing_premixed_flamelets.sh
```

The workflow performs:

```text
premixed YAML flamelets
  -> extracted post_premixed_Z_*.csv data
  -> archive FGMTableBuild_OF7_premixed/02_tables_of7_premixed.tar.xz
  -> OpenFOAM files in 03_testcase/constant/tables
  -> validation
  -> enable useProgressVariableVariance true
```

### Default parameters

| Parameter | Default |
|---|---:|
| Mechanism | `gri30.yaml` |
| Main inlet / table maximum \(Z\) | `0.1559` |
| Pilot \(Z\) used for reporting | `0.04293` |
| Number of \(Z\) entries | `51` |
| Number of scaled-PV entries | `51` |
| Unburned temperature | `294.0 K` |

### Override example

```bash
MECH=myMechanism.yaml NZ=101 NC=101 TIN=300.0 \
./FGMScripts_OF7_premixed_v6_zeroVarPV/make_03_testcase_tables_from_existing_premixed_flamelets.sh
```

Only change these values when the case boundary conditions and intended FGM coordinate system are correspondingly updated.

---

# Workflow C: Full regeneration from premixed flamelets

## 8. Generate flamelets and install tables from scratch

Run:

```bash
./FGMScripts_OF7_premixed_v6_zeroVarPV/make_03_testcase_premixed_tables_from_scratch.sh
```

This generates new flamelets in:

```text
./1DPremixedFlameletFiles/
```

and installs final OpenFOAM tables into:

```text
./03_testcase/constant/tables/
```

### Table-generation method

The script uses:

```python
ct.FreeFlame(...)
```

to generate premixed CH₄/air flame solutions across the case-specific \(Z\) axis.

It does **not** use `CounterflowDiffusionFlame`. The latter generated tables that loaded correctly but failed to ignite the original premixed burner case.

---

## 9. Installed OpenFOAM table files

After either rebuilding or upgrading, the case should contain at least:

```text
03_testcase/constant/tables/
├── T_table
├── psi_table
├── mu_table
├── alpha_table
├── SourcePV_table
├── PVmin_table
├── PVmax_table
├── YWI_table
├── YuWI_table
├── YbWI_table
├── Yu2I_table
├── YuYbI_table
├── Yb2I_table
├── CH4_table
├── O2_table
├── CO2_table
├── H2O_table
├── OH_table
└── N2_table
```

The case must also contain:

```text
03_testcase/constant/tableProperties
03_testcase/constant/PVtableProperties
```

---

## 10. Required OpenFOAM dictionary settings

### 10.1 `constant/combustionProperties`

For the validated v6 zero-variance path:

```foam
FGMModelCoeffs
{
    useProgressVariableVariance true;
    useMixtureFractionVariance  false;
}
```

Check with:

```bash
grep -A8 FGMModelCoeffs 03_testcase/constant/combustionProperties
```

### 10.2 Interpolation type settings

The OpenFOAM 7 runtime-selection names must be:

```foam
// constant/tableProperties
interpolationType   linearInterpolation;

// constant/PVtableProperties
interpolationType   PVlinearInterpolation;
```

Check with:

```bash
grep -H interpolationType \
  03_testcase/constant/tableProperties \
  03_testcase/constant/PVtableProperties
```

### 10.3 Initial variance fields

The current table database is valid only for zero variances. Confirm:

```bash
grep internalField 03_testcase/0/varPV 03_testcase/0/varZ
```

Required result:

```text
03_testcase/0/varPV:internalField   uniform 0;
03_testcase/0/varZ:internalField    uniform 0;
```

Do not initialize either field to a nonzero value with the current generated database.

---

## 11. Clean restart before running the solver

After changing tables or combustion-model settings, restart from `0`. Preserve earlier output first:

```bash
cd /home/eadaymo/OpenFOAM/eadaymo-7/run/OF7_FGM_run_workflow/03_testcase

mkdir -p previous_FGM_run

find . -maxdepth 1 -mindepth 1 -type d \
    -regextype posix-extended \
    -regex './[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?' \
    ! -name '0' \
    -exec mv -t previous_FGM_run {} +
```

Verify:

```bash
ls -d [0-9]* 2>/dev/null
```

Expected:

```text
0
```

---

## 12. Run `FGMFoam`

```bash
FGMFoam | tee log.FGMFoam_zeroVarPV_true
```

The log should show the complete table set being loaded:

```text
Reading table: T
Reading table: psi
Reading table: mu
Reading table: alpha
Reading table: SourcePV
Reading table: YWI
Reading table: YuWI
Reading table: YbWI
Reading table: PVmin
Reading table: PVmax
Reading table: Yu2I
Reading table: YuYbI
Reading table: Yb2I
```

A stable run should also begin from:

```text
Create mesh for time = 0
```

not a previously divergent nonzero time directory.

---

## 13. Recommended validation comparison

The correct numerical validation is a zero-variance A/B comparison.

### 13.1 Baseline case

Run the previously working premixed tables with:

```foam
useProgressVariableVariance false;
```

Store results separately.

### 13.2 Zero-variance PVeta case

Upgrade or rebuild tables using v6 and run with:

```foam
useProgressVariableVariance true;
```

while retaining:

```foam
varPV = 0;
varZ  = 0;
```

### 13.3 Compare fields

Compare at matching times:

```text
T
OH
PV
scaledPV
SourcePV
U
p
```

The flame structure should be effectively unchanged apart from normal numerical differences.

### Example validated fields

Temperature:

![Temperature field for the validated zero-variance premixed FGM case](images/validated_temperature_field.png)

OH:

![OH field for the validated zero-variance premixed FGM case](images/validated_OH_field.png)

---

## 14. Table validation command

The table validator may be run directly:

```bash
python3 FGMScripts_OF7_premixed_v6_zeroVarPV/scripts/validate_of7_premixed_tables.py \
  --case "$PWD/03_testcase" \
  --require-species CH4 O2 CO2 H2O OH N2 \
  --check-z 0.04293 0.1559 \
  --pilot-z 0.04293 \
  --require-zero-input-variances
```

It checks:

- required table presence;
- nested topology;
- table-property interpolation class names;
- required \(Z\) range;
- six auxiliary tables;
- zero-variance moment identities;
- replicated variance slices;
- zero initial `varPV` and `varZ`.

---

## 15. Troubleshooting

### 15.1 Solver crashes immediately after `Reading table: T`

**Likely cause:** incorrect runtime-selection name in table properties.

Required settings:

```foam
interpolationType   linearInterpolation;
```

and:

```foam
interpolationType   PVlinearInterpolation;
```

Do not use:

```foam
interpolationType   linear;
```

### 15.2 `No states restored from ... strain_loop_00.yaml`

**Cause:** older script attempted to restore a full Cantera one-dimensional flame YAML directly as a `SolutionArray`.

**Correction:** use the premixed v6 workflow, which restores a `FreeFlame` object and then obtains its profile data.

### 15.3 Solver runs but main flame does not ignite

**Cause observed previously:** table database was made from counterflow diffusion flamelets rather than premixed flamelets.

**Correction:** use:

```text
generatePremixedFlamelets.py
```

and premixed YAML files from:

```text
1DPremixedFlameletFiles/
```

not diffusion YAML files from:

```text
1DFlameletFiles/
```

### 15.4 Pressure/turbulence divergence at startup

**Indicators:**

```text
Create mesh for time = 5e-05
Courant Number max: very large
epsilon becomes extremely large
```

**Correction:** remove/archive nonzero time directories and restart from `0`. For diagnostic startup stability, use the optional startup patch included with the table package.

### 15.5 Validation fails because `varPV` or `varZ` is nonzero

This is intentional. The current v6 table database is not valid for nonzero variance calculations. Restore zero initial/boundary variance fields before using this database.

### 15.6 Should `useMixtureFractionVariance` be set to `true`?

Not for this release. Retain:

```foam
useMixtureFractionVariance false;
```

The generated database is not PDF-integrated along the mixture-coordinate-variance dimension.

---

## 16. Repository layout recommended for GitHub

A convenient repository layout is:

```text
FGMFoam_OF7_premixed/
├── README.md
├── docs/
│   ├── THEORY_MANUAL.md
│   ├── USER_MANUAL.md
│   └── images/
│       ├── validated_temperature_field.png
│       └── validated_OH_field.png
├── FGMScripts_OF7_premixed_v6_zeroVarPV/
│   ├── make_03_testcase_premixed_tables_from_scratch.sh
│   ├── make_03_testcase_tables_from_existing_premixed_flamelets.sh
│   ├── upgrade_current_working_tables_to_zeroVarPV_true.sh
│   └── scripts/
└── cases/
    └── 03_testcase/
```

Do not commit large generated flamelet or time-directory results unless the repository is intentionally being used for archived validation datasets.

---

## 17. Recommended next development steps

The current baseline is suitable for verifying the developer's variance-enabled lookup path at zero variance.

A subsequent physical model extension should proceed in this order:

1. preserve the current v6 zero-variance result as a regression baseline;
2. implement and validate nonzero `varZ` evolution and PDF integration;
3. then implement and validate nonzero `varPV` evolution and PDF integration;
4. compare true multidimensional tables against this baseline and against experimental or benchmark data.

Until those steps are completed, describe these tables as:

> Four-dimensional solver-compatible premixed FGM tables validated in the zero-variance progress-variable-variance limit.
