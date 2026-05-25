# OpenFOAM 7 FGM premixed-table workflow for `03_testcase`

## Corrections incorporated

This package replaces the previous diffusion-flame workflow. The supplied working table set identifies itself as a **premixed CH4/air FGM** database. The prior patched scripts instead generated `CounterflowDiffusionFlame` data on a `Z=0..1` pure-fuel/air coordinate; that database loaded but did not propagate the pilot flame in this case.

The new workflow includes these changes:

1. `generatePremixedFlamelets.py` generates Cantera `FreeFlame` solutions at multiple premixed CH4/air compositions.
2. `Z` is treated as the **unburned methane/air fuel mass fraction coordinate** used by this case. Defaults bracket the actual boundaries:
   - pilot: `Z = 0.04293`
   - main inlet: `Z = 0.1559`
   - output table axis: `0 <= Z <= 0.1559`
3. `organizePremixedData.py` restores Cantera 3.2 one-dimensional simulation YAML files correctly and extracts the flow solution before computing `PV`, `SourcePV`, thermo, and species fields.
4. `buildPremixedFGMTables.py` creates a `(Z, scaled-PV)` premixed manifold. It uses converged `FreeFlame` profiles in the flammable range and conservative zero-source endpoint blends outside the solved flame range.
5. `csv2of_tables.py` writes the OpenFOAM 7 nested topology and the registered interpolation names:
   - `linearInterpolation` for `tableProperties`
   - `PVlinearInterpolation` for `PVtableProperties`
6. The OpenFOAM writer emits replicated variance slices so the old interpolator has at least two points on each variance axis. Both variance switches are set to `false`.
7. `validate_of7_premixed_tables.py` validates topology, constructor names, and confirms the case inlet `Z` values are inside the table axis.

## Do not reuse the old generated diffusion flamelets

A previous directory named `1DFlameletFiles` containing `strain_loop_*.yaml` is not input to this workflow. Generate a new directory named `1DPremixedFlameletFiles`.

## From scratch

Run from the directory that contains `03_testcase`:

```bash
unzip FGMScripts_OF7_premixed_v5.zip
chmod +x FGMScripts_OF7_premixed_v5/*.sh
./FGMScripts_OF7_premixed_v5/make_03_testcase_premixed_tables_from_scratch.sh
```

The output table files are installed into:

```text
03_testcase/constant/tables
```

## From existing premixed flamelets

Only use this if the following file exists:

```text
./1DPremixedFlameletFiles/premixed_manifest.csv
```

Run:

```bash
./FGMScripts_OF7_premixed_v5/make_03_testcase_tables_from_existing_premixed_flamelets.sh
```

## Important case settings

Inside `constant/combustionProperties`, use:

```foam
FGMModelCoeffs
{
    useProgressVariableVariance false;
    useMixtureFractionVariance  false;
}
```

## Startup after replacing tables

Remove or archive numeric time directories except `0`, then start again from `0`. If the earlier stability patch is not already installed, it is included under `optional_startup_patch/`.

## Configurable values

The defaults match the supplied case. They can be overridden when running the wrapper, for example:

```bash
Z_MAX=0.1559 PILOT_Z=0.04293 NZ=101 NC=101 MECH=gri30.yaml \
./FGMScripts_OF7_premixed_v5/make_03_testcase_premixed_tables_from_scratch.sh
```
