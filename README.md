# Compiling the solver

1. Install OpenFOAM v7

2. copy `applications` to $HOME/OpenFOAM/<username>-7/

3. from `solvers/FGMFoam/src` run `./Allwmake'

# Making the tables

1. Install Cantera 3.2 (3.1 should also work)

2. Build the the tables. The tables will automatically copy to `03_testcase/constant/tables`

```
cd 1DFlameletFiles
./Allrun
```

# Run the case

```
cd 03_testcase
FGMFoam
```

# Issues

1. Result doesn't look like the Sandia D flame!

But at least this runs... 


# Then... 

A. Migrate to OF10, 13+, and/or OFv2512

B. Build a solution for FGSFoam? 

