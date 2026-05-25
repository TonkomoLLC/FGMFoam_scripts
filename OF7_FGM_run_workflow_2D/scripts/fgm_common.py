#!/usr/bin/env python3
"""Shared Cantera/OpenFOAM-7 FGM utilities for a premixed CH4/air manifold."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import cantera as ct
import numpy as np

PV_WEIGHTS = {"H2O": 4.0, "CO2": 2.0, "H2": 0.5, "CO": 1.0}

@dataclass(frozen=True)
class MixtureConfig:
    mechanism: str = "gri30.yaml"
    fuel: str = "CH4:1"
    oxidizer: str = "O2:0.21, N2:0.79"
    tin: float = 294.0
    pressure: float = 101325.0


def validate_progress_species(gas: ct.Solution) -> None:
    missing = [sp for sp in PV_WEIGHTS if sp not in gas.species_names]
    if missing:
        raise ValueError(f"Mechanism is missing progress-variable species: {missing}")


def progress_variable(gas: ct.Solution, Y: np.ndarray | None = None) -> float:
    """Unscaled PV in kmol/kg, matching the supplied OpenFOAM-7 table convention."""
    if Y is not None:
        current = gas.Y.copy()
        gas.Y = Y
    try:
        return float(sum(
            w * gas.Y[gas.species_index(sp)] / gas.molecular_weights[gas.species_index(sp)]
            for sp, w in PV_WEIGHTS.items()
        ))
    finally:
        if Y is not None:
            gas.Y = current


def source_progress_variable(gas: ct.Solution) -> float:
    """Volumetric source of raw PV in kmol/m^3/s."""
    return float(sum(w * gas.net_production_rates[gas.species_index(sp)] for sp, w in PV_WEIGHTS.items()))


def set_unburned_state(gas: ct.Solution, z: float, cfg: MixtureConfig) -> None:
    """Set unburned CH4/air mixture state where z is kg fuel / kg unburned mixture.

    For pure CH4 and air, Cantera's Bilger mixture fraction is numerically equal
    to the unburned CH4 mass fraction. This matches the supplied Sandia-style case,
    whose main inlet uses Z=0.1559 rather than a normalized stream fraction of 1.
    """
    gas.TP = cfg.tin, cfg.pressure
    gas.set_mixture_fraction(float(z), cfg.fuel, cfg.oxidizer, basis="mole")


def stoichiometric_z(gas: ct.Solution, cfg: MixtureConfig) -> float:
    gas.TP = cfg.tin, cfg.pressure
    gas.set_equivalence_ratio(1.0, cfg.fuel, cfg.oxidizer, basis="mole")
    return float(gas.mixture_fraction(cfg.fuel, cfg.oxidizer, basis="mole"))


def equivalence_ratio_at_z(gas: ct.Solution, z: float, cfg: MixtureConfig) -> float:
    set_unburned_state(gas, z, cfg)
    return float(gas.equivalence_ratio(cfg.fuel, cfg.oxidizer, basis="mole"))


def thermo_values(gas: ct.Solution) -> dict[str, float]:
    rho = float(gas.density)
    cp = float(gas.cp_mass)
    return {
        "T": float(gas.T),
        "rho": rho,
        "psi": rho / float(gas.P),
        "mu": float(gas.viscosity),
        "Cps": cp,
        "alpha": float(gas.thermal_conductivity) / (rho * cp),
        "PV": progress_variable(gas),
        "SourcePV": source_progress_variable(gas),
    }


def endpoint_state(gas: ct.Solution, z: float, cfg: MixtureConfig, burned: bool) -> dict[str, float | np.ndarray]:
    set_unburned_state(gas, z, cfg)
    if burned:
        gas.equilibrate("HP")
    values: dict[str, float | np.ndarray] = thermo_values(gas)
    values["SourcePV"] = 0.0
    values["Y"] = gas.Y.copy()
    return values


def sorted_unique(values: Iterable[float]) -> np.ndarray:
    return np.asarray(sorted(set(float(v) for v in values)), dtype=float)
