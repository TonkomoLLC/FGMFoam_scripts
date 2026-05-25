# FGMFoam for OpenFOAM 7

# FGMFoam-Delft

[Original source code] ( https://www.tfd.chalmers.se/~hani/kurser/OS_CFD/ ): Description of the reacting flow solver FGMFoam, Michael Bertsch

To compile:

1. Setup OpenFOAM 7 on your computer

2. Place the FGMFoam source files in /home/userName-7/applications/solvers/FGMFoam. There should be two subdirectories, `applications` and `src`

From within `src`, 

```
source .bashrc
./Allwmake
```

# OF7_FGM_run_workflow_2D

**2D flamelet tables**

```
./make_03_testcase_premixed_tables_from_scratch.sh
cd 03_testcase
FGMFoam
FGMFoamPost -latestTime
```




