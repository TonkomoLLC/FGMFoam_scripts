# FGMFoam for OpenFOAM 7

# Original Source Code

(https://www.tfd.chalmers.se/~hani/kurser/OS_CFD/ ): Description of the reacting flow solver FGMFoam, Michael Bertsch

# Compilation with OpenFOAM 7

To compile:

1. Setup OpenFOAM 7 on your computer

2. Place the contents of this directory in /home/userName-7/applications/solvers/. Within the directory `FGMFoam` there should be two subdirectories, `applications` and `src`

Replace `userName` with your account's username.

From within `src`, edit bashrc if needed. Then, after ensuring you've run `source /opt/openfoam7/etc/bashrc` (use the appropriate path for OpenFOAM 7 on your computer)

```
source bashrc
./Allwmake
```

# Workflow 

## 2D flamelet tables**

```
./make_03_testcase_premixed_tables_from_scratch.sh
cd 03_testcase
FGMFoam
FGMFoamPost -latestTime
```

## 4D flamelet tables**

Technically, 4D in OpenFOAM storage and lookup interface, but still 2D in physical manifold content: it resolves Z and scaled progress variable c, while the Z-variance and progress-variable-variance axes are replicated zero-variance slices.

```
./make_03_testcase_premixed_tables_from_scratch.sh
cd 03_testcase
FGMFoam
FGMFoamPost -latestTime
```

![Sandia D temperature](./docs/images/validated_temperature_field.png)


![Sandia D OH mass fraction](./docs/images/validated_OH_field.png)


# Docs

**OpenFOAM 7 Premixed FGM Documentation Bundle**


- [Theory Manual](docs/THEORY_MANUAL.md)
- [User Manual](docs/USER_MANUAL.md)

The current tables provide a four-dimensional OpenFOAM lookup/storage interface with a physically resolved two-dimensional premixed manifold and a validated zero-variance `useProgressVariableVariance true` execution path.

