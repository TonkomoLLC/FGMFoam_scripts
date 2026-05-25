# OpenFOAM 7 premixed FGM: validated zero-variance `useProgressVariableVariance true` path

This package extends the working premixed CH4/air table generator to exercise the
developer case setting:

```foam
FGMModelCoeffs
{
    useProgressVariableVariance true;
    useMixtureFractionVariance  false;
}
```

## Scope

The generated database includes the six auxiliary tables loaded by `FGMModel` when
`useProgressVariableVariance true`:

```text
YWI_table   YuWI_table   YbWI_table
Yu2I_table YuYbI_table  Yb2I_table
```

The implementation is valid for the **zero-variance limit** only. The variance-axis
slices are replicated and the initial case must retain:

```foam
varPV internalField uniform 0;
varZ  internalField uniform 0;
```

This package does not yet perform presumed-PDF integration for nonzero variances.

## Identities used

At each mixture coordinate `Z` and scaled progress coordinate `c`:

```text
Yu = PVmin
Yb = PVmax
PV = Yu + c*(Yb - Yu)
W  = SourcePV

YWI   = PV*W
YuWI  = Yu*W
YbWI  = Yb*W

Yu2I  = Yu^2
YuYbI = Yu*Yb
Yb2I  = Yb^2
```

These identities match the supplied developer tables at zero variance and make the
scaled-variance conversion in `FGMModel.C` return zero when `varPV=0`.

## Recommended immediate use: upgrade the table set that already ignited

From the directory containing `03_testcase`:

```bash
unzip /path/to/FGMScripts_OF7_premixed_v6_zeroVarPV.zip
chmod +x FGMScripts_OF7_premixed_v6_zeroVarPV/*.sh

./FGMScripts_OF7_premixed_v6_zeroVarPV/upgrade_current_working_tables_to_zeroVarPV_true.sh
```

This route does not rerun Cantera or change `T_table`/`SourcePV_table`; it backs up
the current installed tables, adds the six auxiliary tables, enables
`useProgressVariableVariance true`, and validates zero-variance equivalence.

Then restart from time zero:

```bash
cd 03_testcase
mkdir -p prior_false_variance_run
mv 0.[0-9]* prior_false_variance_run/ 2>/dev/null || true
FGMFoam | tee log.FGMFoam_zeroVarPV_true
```

## Full regeneration

Generate new premixed flamelets and install the complete table set:

```bash
./FGMScripts_OF7_premixed_v6_zeroVarPV/make_03_testcase_premixed_tables_from_scratch.sh
```

Rebuild tables from already generated premixed flamelets:

```bash
./FGMScripts_OF7_premixed_v6_zeroVarPV/make_03_testcase_tables_from_existing_premixed_flamelets.sh
```

## Comparison requirement

With zero `varPV` and `varZ`, compare `useProgressVariableVariance false` and `true`.
The `T`, `PV`, `scaledPV`, and `SourcePV` solutions should agree apart from ordinary
time-integration numerical differences. A discrepancy indicates an auxiliary-table
or lookup inconsistency.
