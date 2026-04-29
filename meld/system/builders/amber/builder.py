#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module to build a System from AmberSubSystems
"""

import logging
import subprocess
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import numpy as np  # type: ignore
import openmm as mm  # type: ignore
from openmm import app  # type: ignore
from openmm import unit as u  # type: ignore
from openmm.app import forcefield as ff  # type: ignore

from meld import util
from meld.system import indexing
from meld.system.builders.amber import amap, subsystem
from meld.system.builders.spec import SystemSpec

logger = logging.getLogger(__name__)

try:
    from gamd.integrator_factory import *    #type: ignore
    has_gamd = True
except:
    has_gamd = False


@partial(dataclass, frozen=True)
class AmberOptions:
    default_temperature: u.Quantity = field(default_factory=lambda: 300.0 * u.kelvin)
    forcefield: str = "ff14sbside"
    solvation: str = "implicit"
    gb_radii: str = "mbondi3"
    implicit_solvent_model: str = "gbNeck2"
    solute_dielectric: Optional[float] = None
    solvent_dielectric: Optional[float] = None
    implicit_solvent_salt_conc: Optional[float] = None
    solvent_forcefield: str = "tip3p"
    solvent_distance: float = 0.6
    explicit_ions: bool = False
    p_ion: str = "Na+"
    p_ioncount: int = 0
    n_ion: str = "Cl-"
    n_ioncount: int = 0
    enable_pme: bool = False
    pme_tolerance: float = 0.0005
    enable_pressure_coupling: bool = False
    pressure: u.Quantity = field(default_factory=lambda: 1.01325 * u.bar)
    pressure_coupling_update_steps: int = 25
    cutoff: Optional[float] = None
    remove_com: bool = True
    use_big_timestep: bool = False
    use_bigger_timestep: bool = False
    enable_amap: bool = False
    amap_alpha_bias: float = 1.0
    amap_beta_bias: float = 1.0
    enable_gamd: bool = False
    boost_type_str: str = "upper-total"
    conventional_md_prep: int = 5
    conventional_md: int = 50
    gamd_equilibration_prep: int = 15
    gamd_equilibration: int = 150
    total_simulation_length: int = 1000
    averaging_window_interval: int = 2500
    sigma0p: float = 6.0
    sigma0d: float = 6.0
    random_seed: int = 0
    friction_coefficient: float = 1.0
    pre_equilibrate: bool = False
    preeq_minimize_maxiter: int = 3000
    preeq_nvt_steps: int = 5000
    preeq_npt_steps: int = 2000
    preeq_use_restraints: bool = False
    preeq_restraint_k_initial: float = 1000.0  # kJ/mol/nm^2
    preeq_restraint_k_final: float = 0.0       # kJ/mol/nm^2
    preeq_restrain_backbone_only: bool = True
    preeq_timestep_fs: float = 0.5              # fs


    def __post_init__(self):
        print("[DEBUG] Entered AmberOptions.__post_init__")
        # Sanity checks for implicit and explicit solvent
        if self.solvation == "implicit":
            if self.enable_pme:
                raise ValueError("Using implicit solvation, but `enable_pme` is True.")
            if self.enable_pressure_coupling:
                raise ValueError(
                    "Using implicit solvation, but `enable_pressure_coupling` is True."
                )
        elif self.solvation == "explicit":
            if not self.enable_pme:
                raise ValueError("Using explicit solvation, but `enable_pme` is False.")
            if not self.enable_pressure_coupling:
                raise ValueError(
                    "Using explicit solvation, but `enable_pressure_coupling` is False."
                )
            if self.enable_amap:
                raise ValueError("Using explicit solvation, but `enable_amap` is True.")
            if self.cutoff is None:
                raise ValueError("Using explicit solvation, but `cutoff` is None.")
        else:
            raise ValueError(f"Unknown solvation model {self.solvation}")

        if self.forcefield not in ["ff12sb", "ff14sb", "ff14sbside", "parmbsc1", "OL15"]:
            raise ValueError(f"Unknown forcefield {self.forcefield}")

        if self.gb_radii not in ["mbondi2", "mbondi3"]:
            raise ValueError(f"Unknown gb_radii {self.gb_radii}")

        if self.solvent_forcefield not in ["spce", "spceb", "opc", "tip3p", "tip4pew"]:
            raise ValueError(f"Unknown solvent_forcefield {self.solvent_forcefield}")

        if self.p_ion not in ["Na+", "K+", "Li+", "Rb+", "Cs+", "Mg+"]:
            raise ValueError(f"Unknown p_ion {self.p_ion}")

        if self.n_ion not in ["Cl-", "I-", "Br-", "F-"]:
            raise ValueError(f"Unknown n_ion {self.n_ion}")

        if isinstance(self.default_temperature, u.Quantity):
            object.__setattr__(
                self,
                "default_temperature",
                self.default_temperature.value_in_unit(u.kelvin),
            )
        if self.default_temperature < 0:
            raise ValueError(f"default_temperature must be >= 0")

        if isinstance(self.pressure, u.Quantity):
            object.__setattr__(self, "pressure", self.pressure.value_in_unit(u.bar))
        if self.pressure < 0:
            raise ValueError(f"pressure must be >= 0")
        print("[DEBUG] Exiting AmberOptions.__post_init__")


class AmberSystemBuilder:
    r"""
    Class to handle building a System from SubSystems.
    """

    options: AmberOptions

    def __init__(self, options: AmberOptions):
        print("[DEBUG] AmberSystemBuilder.__init__ called")
        self.options = options
        self._set_forcefield()
        if self.options.solvation == "explicit":
            self._set_solvent_forcefield()
            self._solvent_dist = self.options.solvent_distance * 10.0  # nm to angstrom
        print("[DEBUG] AmberSystemBuilder.__init__ complete")

    def build_system(
        self,
        subsystems: List[subsystem._AmberSubSystem],
        leap_header_cmds: Optional[List[str]] = None,
    ) -> SystemSpec:
        print("[DEBUG] Entered build_system")
        if not subsystems:
            raise ValueError("len(subsystems) must be > 0")
        print(f"[DEBUG] Number of subsystems: {len(subsystems)}")

        if leap_header_cmds is None:
            leap_header_cmds = []
        if isinstance(leap_header_cmds, str):
            leap_header_cmds = [leap_header_cmds]

        print("[DEBUG] Entering util.in_temp_dir context manager")
        with util.in_temp_dir():
            mol_ids = []
            chains = []
            current_res_index = 0
            leap_cmds = []
            leap_cmds.extend(self._generate_leap_header())
            leap_cmds.extend(leap_header_cmds)
            for index, sub in enumerate(subsystems):
                print(f"[DEBUG] Processing subsystem {index}")
                for chain in sub._info.chains:
                    residues_with_offset = {
                        k: v + current_res_index for k, v in chain.residues.items()
                    }
                    chains.append(indexing._ChainInfo(residues_with_offset))
                current_res_index += sub._info.n_residues

                mol_id = f"mol_{index}"
                mol_ids.append(mol_id)
                print(f"[DEBUG] Calling prepare_for_tleap on subsystem {index} with mol_id {mol_id}")
                sub.prepare_for_tleap(mol_id)
                print(f"[DEBUG] Appending tleap input for subsystem {index}")
                leap_cmds.extend(sub.generate_tleap_input(mol_id))

            if self.options.solvation == "explicit":
                print("[DEBUG] Generating explicit solvent commands")
                leap_cmds.extend(self._generate_solvent(mol_ids))
                leap_cmds.extend(self._generate_leap_footer([f"solute"]))
            else:
                print("[DEBUG] Generating implicit solvent commands")
                leap_cmds.extend(self._generate_leap_footer(mol_ids))

            print("[DEBUG] Writing tleap.in file")
            with open("tleap.in", "w") as tleap_file:
                tleap_string = "\n".join(leap_cmds)
                tleap_file.write(tleap_string)

            print("[DEBUG] Calling tleap subprocess")
            try:
                subprocess.check_call("tleap -f tleap.in > tleap.out", shell=True)
                print("[DEBUG] tleap subprocess completed successfully")
            except subprocess.CalledProcessError:
                print("[DEBUG] Call to tleap failed. Dumping input/output.")
                print("=========")
                print("tleap.in")
                print("=========")
                print(open("tleap.in").read())
                print("=========")
                print("tleap.out")
                print("=========")
                print(open("tleap.out").read())
                print("========")
                print("leap.log")
                print("========")
                print(open("leap.log").read())
                raise

            print("[DEBUG] Reading system.top for AmberPrmtopFile")
            prmtop = app.AmberPrmtopFile("system.top")
            print("[DEBUG] Reading system.mdcrd for AmberInpcrdFile")
            crd = app.AmberInpcrdFile("system.mdcrd")


# [EQ] -------- Optional pre-equilibration before we export arrays (robust) --------
        if self.options.pre_equilibrate:
            print("[DEBUG][EQ] Pre-equilibration requested (minimize -> NVT -> NPT)")
        # Build temporary OpenMM system (may include barostat if options.enable_pressure_coupling True)
            tmp_system, tmp_baro = _create_openmm_system(
                prmtop,
                self.options.solvation,
                self.options.cutoff,
                self.options.use_big_timestep,
                self.options.use_bigger_timestep,
                self.options.implicit_solvent_model,
                self.options.enable_pme,
                self.options.pme_tolerance,
                self.options.enable_pressure_coupling,
                self.options.pressure,
                self.options.pressure_coupling_update_steps,
                self.options.remove_com,
                self.options.default_temperature,
                self.options.implicit_solvent_salt_conc,
                self.options.solute_dielectric,
                self.options.solvent_dielectric
            )
            # Optionally add a barostat (NPT) if the returned tmp_baro is None and user wants pressure coupling.
            added_barostat = False
            if self.options.enable_pressure_coupling and tmp_baro is None:
                try:
                # MonteCarloBarostat expects pressure in bar
                    barostat = mm.MonteCarloBarostat(self.options.pressure * u.bar, self.options.default_temperature * u.kelvin)
                    tmp_system.addForce(barostat)
                    added_barostat = True
                    print("[DEBUG][EQ] Added MonteCarloBarostat to system for NPT.")
                except Exception as e:
                    print(f"[WARNING][EQ] Could not add barostat: {e} -- continuing without explicit barostat")
            # Create positional restraints (optional)
            restr_force = None
            if self.options.preeq_use_restraints:
                print("[DEBUG][EQ] Preparing positional restraints on solute atoms")
                restr_force = mm.CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
                restr_force.addGlobalParameter("k", float(self.options.preeq_restraint_k_initial))
                restr_force.addPerParticleParameter("x0")
                restr_force.addPerParticleParameter("y0")
                restr_force.addPerParticleParameter("z0")
                pos = crd.getPositions(asNumpy=False)
                pos_arr = pos.value_in_unit(u.nanometer)
                # Heuristic solvent residue names (covers common variants)
                solvent_names = {"WAT", "HOH", "TIP3", "TIP3P", "SOL"}
                for atom in prmtop.topology.atoms():
                    resname = atom.residue.name.upper()
                    if resname in solvent_names:
                        continue
                    if atom.element is None:
                        continue
                    if atom.element.symbol == "H":
                        continue
                    # optionally restrain backbone only (N, CA, C)
                    if self.options.preeq_restrain_backbone_only and atom.name.upper() not in {"N", "CA", "C","P","O5'","C5'","O3'"}:
                        continue
                    x0, y0, z0 = pos_arr[atom.index]
                    restr_force.addParticle(atom.index, [float(x0), float(y0), float(z0)])
                if restr_force.getNumParticles() > 0:
                    tmp_system.addForce(restr_force)
                    print(f"[DEBUG][EQ] Restraint force added with {restr_force.getNumParticles()} particles; k_initial={self.options.preeq_restraint_k_initial}")
                else:
                    print("[DEBUG][EQ] No particles added to restraint force; skipping restraints")
                    restr_force = None
            
            # Use a conservative smaller timestep for pre-equilibration integrator
            small_timestep = (self.options.preeq_timestep_fs) * u.femtosecond
            print(f"[DEBUG][EQ] Using a small timestep of {small_timestep} for pre-eq")
            
            tmp_integrator = mm.LangevinIntegrator(
                self.options.default_temperature * u.kelvin, 1.0 / u.picosecond, small_timestep)
            
            # Create Simulation now that tmp_system contains all forces (barostat & restraints)
            sim = app.Simulation(prmtop.topology, tmp_system, tmp_integrator, platform=None)
            sim.context.setPositions(crd.getPositions())
            try:
                boxvecs = crd.getBoxVectors()
                sim.context.setPeriodicBoxVectors(*boxvecs)
            except AttributeError:
                pass
            
            # Set velocities if absent
            try:
                _ = crd.getVelocities()
                print("[DEBUG][EQ] Velocities present in input")
            except AttributeError:
                print("[DEBUG][EQ] No velocities; initializing at target T")
            
            sim.context.setVelocitiesToTemperature(self.options.default_temperature * u.kelvin)
            
            # Minimization (user-configurable iterations)
            print(f"[DEBUG][EQ] Minimizing energy (maxIts={self.options.preeq_minimize_maxiter})")
            try:
                sim.minimizeEnergy(maxIterations=int(self.options.preeq_minimize_maxiter))
            except Exception as e:
                print(f"[WARNING][EQ] Minimize raised: {e} -- trying one more minimize then continue")
                try:
                    sim.minimizeEnergy(maxIterations=int(self.options.preeq_minimize_maxiter))
                except Exception:
                    print("[ERROR][EQ] Minimization failed; aborting pre-equilibration")
                    raise
            
            # Helper to run MD safely
            def run_steps_safe(simulation, n_steps):
                if n_steps <= 0:
                    return True
                try:
                    simulation.step(int(n_steps))
                    return True
                except Exception as e:
                    print(f"[ERROR][EQ] MD step failed with exception: {e}")
                    return False
            
            # Stage 1: NVT (restrained or not depending on use_restraints)
            print(f"[DEBUG][EQ] Running NVT (stage1) for {self.options.preeq_nvt_steps} steps")
            ok = run_steps_safe(sim, self.options.preeq_nvt_steps)
            if not ok:
                print("[DEBUG][EQ] NVT failed — trying extra minimize + shorter test")
                try:
                    sim.minimizeEnergy(maxIterations=2000)
                except Exception:
                    print("[ERROR][EQ] Extra minimize failed after NVT failure; aborting")
                    raise
                tmp_integrator.setStepSize(0.5 * u.femtosecond)
                ok = run_steps_safe(sim, min(1000, self.options.preeq_nvt_steps))
                if not ok:
                    raise RuntimeError("Pre-equilibration NVT unstable even after recovery")
            
            # If we used restraints, reduce them before NPT (or remove if k_final==0)
            if restr_force is not None:
                print("[DEBUG][EQ] Adjusting restraint strength for next stage")
                try:
                    restr_force.setGlobalParameterDefaultValue(0, float(self.options.preeq_restraint_k_final))
                except Exception:
                # fallback: leave as-is
                    print("[WARNING][EQ] Could not set global parameter on restraint force; leaving it as-is")
            
            # Stage 2: NPT (unrestrained if user wanted) - if user wanted to remove restraints entirely do so
            if restr_force is not None and float(self.options.preeq_restraint_k_final) == 0.0:
            # Remove restraint force from system; must recreate Simulation when system mutates
            # Find its index and remove
                for i, f in enumerate(list(tmp_system.getForces())):
                    if isinstance(f, type(restr_force)) and f.getNumParticles() == restr_force.getNumParticles():
                        try:
                            tmp_system.removeForce(i)
                            print("[DEBUG][EQ] Removed restraint force from system before NPT")
                            restr_force = None
                            break
                        except Exception:
                            print("[WARNING][EQ] Failed to remove restraint; leaving with k=0")
                            try:
                                f.setGlobalParameterDefaultValue(0, 0.0)
                            except Exception:
                                pass
            # recreate Simulation with mutated system (barostat presence accounted for)
                print("[DEBUG][EQ] Creating a new integrator for the updated system")
                tmp_integrator_npt = mm.LangevinIntegrator(
                        self.options.default_temperature * u.kelvin, 1.0 / u.picosecond,
                        self.options.preeq_timestep_fs * u.femtosecond)
                
                sim = app.Simulation(prmtop.topology, tmp_system, tmp_integrator_npt, platform=None)
                sim.context.setPositions(pos.value_in_unit(u.nanometer) * u.nanometer) # reapply positions
                try:
                    sim.context.setPeriodicBoxVectors(*boxvecs)
                except Exception:
                    pass
                try:
                    sim.context.setVelocitiesToTemperature(self.options.default_temperature * u.kelvin)
                except Exception:
                    pass
            # Ensure a barostat exists if NPT requested
            if self.options.preeq_npt_steps > 0:
                has_barostat = any(isinstance(f, mm.MonteCarloBarostat) for f in tmp_system.getForces())
                if not has_barostat and self.options.enable_pressure_coupling:
                    try:
                        barostat = mm.MonteCarloBarostat(self.options.pressure * u.bar, self.options.default_temperature * u.kelvin)
                        tmp_system.addForce(barostat)
                        print("[DEBUG][EQ] Added MonteCarloBarostat to system for NPT stage")
                    # recreate sim to pick up the new force
                        sim = app.Simulation(prmtop.topology, tmp_system, tmp_integrator, platform=None)
                        sim.context.setPositions(crd.getPositions())
                        try:
                            sim.context.setPeriodicBoxVectors(*boxvecs)
                        except Exception:
                            pass
                        try:
                            sim.context.setVelocitiesToTemperature(self.options.default_temperature * u.kelvin)
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"[WARNING][EQ] Could not add barostat for NPT: {e} -- proceeding without explicit NPT")
                print(f"[DEBUG][EQ] Running NPT (stage2) for {self.options.preeq_npt_steps} steps")
                ok = run_steps_safe(sim, self.options.preeq_npt_steps)
                if not ok:
                    print("[DEBUG][EQ] NPT failed; attempting extra minimize then short run")
                    try:
                        sim.minimizeEnergy(maxIterations=2000)
                    except Exception:
                        print("[ERROR][EQ] Extra minimize failed during NPT error recovery; aborting")
                        raise
                    ok = run_steps_safe(sim, min(1000, self.options.preeq_npt_steps))
                    if not ok:
                        raise RuntimeError("Pre-equilibration NPT failed")
            
            # Final state extraction
            print("[DEBUG][EQ] Pre-equilibration completed successfully; extracting positions")
            state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
            pos_final = state.getPositions(asNumpy=True)
            try:
                box_final = state.getPeriodicBoxVectors()
            except Exception:
                box_final = None

            # Overwrite the Amber restart with equilibrated coordinates
            #print("[DEBUG][EQ] Writing equilibrated coordinates to system.mdcrd")
            #app.AmberInpcrdFile.writeFile(prmtop.topology, pos_final, boxVectors=box_final, file="system.mdcrd")
        # [EQ] ---------------------------------------------------------------------  


        print("[DEBUG] Exited util.in_temp_dir context manager")
        topology = prmtop.topology
        topology = _add_chains(topology, chains)
        print("[DEBUG] Topology chains added")

        print("[DEBUG] Creating openmm system")
        system, barostat = _create_openmm_system(
            prmtop,
            self.options.solvation,
            self.options.cutoff,
            self.options.use_big_timestep,
            self.options.use_bigger_timestep,
            self.options.implicit_solvent_model,
            self.options.enable_pme,
            self.options.pme_tolerance,
            self.options.enable_pressure_coupling,
            self.options.pressure,
            self.options.pressure_coupling_update_steps,
            self.options.remove_com,
            self.options.default_temperature,
            self.options.implicit_solvent_salt_conc,
            self.options.solute_dielectric,
            self.options.solvent_dielectric,
        )
        print("[DEBUG] openmm system created")

        if self.options.enable_amap:
            print("[DEBUG] Adding amap to system")
            amap.add_amap(
                system,
                topology,
                self.options.amap_alpha_bias,
                self.options.amap_beta_bias,
            )

        if self.options.enable_gamd:
            print("[DEBUG] enable_gamd is True, checking for gamd")
            assert has_gamd == True, "Couldn't find library integrator_factory. Please, install GaMD for OpenMM"
            allowed_modes = [
                "upper-dual",
                "lower-dual",
                "upper-total",
                "lower-total",
                "lower-dihedral",
                "upper-dihedral",
            ]
            if self.options.boost_type_str in allowed_modes:
                print(f"[DEBUG] Creating gamd integrator with mode: {self.options.boost_type_str}")
                integrator = _create_gamd_integrator(
                    self.options,
                    system,
                )
            else:
                raise Exception(
                    f"{self.options.boost_type_str} mode not supported. Check your boost_type_str option."
                )
        else:
            print("[DEBUG] Creating standard integrator")
            integrator = _create_integrator(
                self.options.default_temperature,
                self.options.use_big_timestep,
                self.options.use_bigger_timestep,
            )

        print("[DEBUG] Getting coordinates from AmberInpcrdFile")

        if self.options.pre_equilibrate:
            print("[DEBUG] Using equilibrated coordinates")
            coords = pos_final.value_in_unit(u.nanometer)
            vels = np.zeros_like(coords)  # Will be set properly by setVelocitiesToTemperature later
            print("[DEBUG] Will regenerate velocities at target temperature")
            # If you extracted box vectors during equilibration, use them too
            if box_final is not None:
                try:
                    box_a = box_final[0][0].value_in_unit(u.nanometer)
                    box_b = box_final[1][1].value_in_unit(u.nanometer) 
                    box_c = box_final[2][2].value_in_unit(u.nanometer)
                    box = np.array([box_a, box_b, box_c])
                except:
                # Fallback to original box if extraction fails
                    box = crd.getBoxVectors(asNumpy=True) if hasattr(crd, 'getBoxVectors') else None
        else:
            print("[DEBUG] Getting coordinates from AmberInpcrdFile")
            coords = crd.getPositions(asNumpy=True).value_in_unit(u.nanometer)


            #coords = crd.getPositions(asNumpy=True).value_in_unit(u.nanometer)
            try:
                print("[DEBUG] Getting velocities from AmberInpcrdFile")
                vels = crd.getVelocities(asNumpy=True)
            except AttributeError:
                print("[WARNING] No velocities found, setting to zero")
                vels = np.zeros_like(coords)
            try:
                print("[DEBUG] Getting box vectors from AmberInpcrdFile")
                box = crd.getBoxVectors(asNumpy=True)
                box_a = box[0][0].value_in_unit(u.nanometer)
                assert box[0][1] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
                assert box[0][1] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
                box_b = box[1][1].value_in_unit(u.nanometer)
                assert box[1][0] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
                assert box[1][2] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
                box_c = box[2][2].value_in_unit(u.nanometer)
                assert box[2][0] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
                assert box[2][1] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
                box = np.array([box_a, box_b, box_c])
            except AttributeError:
                print("[WARNING] No box vectors found")
                box = None

        print("[DEBUG] Returning SystemSpec from build_system")
        return SystemSpec(
            self.options.solvation,
            system,
            topology,
            integrator,
            barostat,
            coords,
            vels,
            box,
            {
                "solvation": self.options.solvation,
                "builder": "amber",
                "implicit_solvent_model": self.options.implicit_solvent_model,
            },
        )

    def _set_forcefield(self):
        print("[DEBUG] Setting forcefield")
        ff_dict = {
            "ff12sb": "leaprc.ff12SB",
            "ff14sb": "leaprc.protein.ff14SB",
            "ff14sbside": "leaprc.protein.ff14SBonlysc",
            "parmbsc1": "leaprc.DNA.bsc1",
            "OL15": "leaprc.DNA.OL15",
        }
        self._forcefield = ff_dict[self.options.forcefield]

    def _set_solvent_forcefield(self):
        print("[DEBUG] Setting solvent forcefield and box")
        ff_dict = {
            "spce": "leaprc.water.spce",
            "spceb": "leaprc.water.spceb",
            "opc": "leaprc.water.opc",
            "tip3p": "leaprc.water.tip3p",
            "tip4pew": "leaprc.water.tip4pew",
        }
        box_dict = {
            "spce": "SPCBOX",
            "spceb": "SPCBOX",
            "opc": "OPCBOX",
            "tip3p": "TIP3PBOX",
            "tip4pew": "TIP4PEWBOX",
        }

        self._solvent_forcefield = ff_dict[self.options.solvent_forcefield]
        self._solvent_box = box_dict[self.options.solvent_forcefield]

    def _generate_leap_header(self):
        print("[DEBUG] _generate_leap_header called")
        leap_cmds = []
        leap_cmds.append(f"set default PBradii {self.options.gb_radii}")
        leap_cmds.append(f"source {self._forcefield}")
        if self.options.solvation == "explicit":
            leap_cmds.append(f"source {self._solvent_forcefield}")
        return leap_cmds

    def _generate_solvent(self, mol_ids):
        print(f"[DEBUG] _generate_solvent called with {len(mol_ids)} mol_ids")
        leap_cmds = []
        list_of_mol_ids = ""
        for mol_id in mol_ids:
            list_of_mol_ids += f"{mol_id} "
        leap_cmds.append(f"solute = combine {{ {list_of_mol_ids} }}")
        leap_cmds.append(f"solvateBox solute {self._solvent_box} {self._solvent_dist}")
        if self.options.explicit_ions:
            leap_cmds.append(
                f"addIons solute {self.options.p_ion} {self.options.p_ioncount}"
            )
            leap_cmds.append(
                f"addIons solute {self.options.n_ion} {self.options.n_ioncount}"
            )
        return leap_cmds

    def _generate_leap_footer(self, mol_ids):
        print("[DEBUG] _generate_leap_footer called")
        leap_cmds = []
        list_of_mol_ids = ""
        for mol_id in mol_ids:
            list_of_mol_ids += f"{mol_id} "
        leap_cmds.append(f"sys = combine {{ {list_of_mol_ids} }}")
        leap_cmds.append("check sys")
        leap_cmds.append("saveAmberParm sys system.top system.mdcrd")
        leap_cmds.append("quit")
        return leap_cmds


def _create_openmm_system(
    parm_object,
    solvation_type,
    cutoff,
    use_big_timestep,
    use_bigger_timestep,
    implicit_solvent,
    enable_pme,
    pme_tolerance,
    enable_pressure_coupling,
    pressure,
    pressure_coupling_update_steps,
    remove_com,
    default_temperature,
    implicitSolventSaltConc,
    soluteDielectric,
    solventDielectric,
):
    print("[DEBUG] Entering _create_openmm_system")
    if solvation_type == "implicit":
        logger.info("Creating implicit solvent system")
        print("[DEBUG] Creating implicit solvent system")
        system = _create_openmm_system_implicit(
            parm_object,
            cutoff,
            use_big_timestep,
            use_bigger_timestep,
            implicit_solvent,
            remove_com,
            implicitSolventSaltConc,
            soluteDielectric,
            solventDielectric,
        )
        baro = None
    elif solvation_type == "explicit":
        logger.info("Creating explicit solvent system")
        print("[DEBUG] Creating explicit solvent system")
        system, baro = _create_openmm_system_explicit(
            parm_object,
            cutoff,
            use_big_timestep,
            use_bigger_timestep,
            enable_pme,
            pme_tolerance,
            enable_pressure_coupling,
            pressure,
            pressure_coupling_update_steps,
            remove_com,
            default_temperature,
        )
    else:
        raise ValueError(f"unknown value for solvation_type: {solvation_type}")

    print("[DEBUG] Exiting _create_openmm_system")
    return system, baro


def _get_hydrogen_mass_and_constraints(use_big_timestep, use_bigger_timestep):
    print("[DEBUG] Entering _get_hydrogen_mass_and_constraints")
    if use_big_timestep:
        logger.info("Enabling hydrogen mass=3, constraining all bonds")
        print("[DEBUG] Using hydrogen mass=3, constraining all bonds")
        constraint_type = ff.AllBonds
        hydrogen_mass = 3.0 * u.gram / u.mole
    elif use_bigger_timestep:
        logger.info("Enabling hydrogen mass=4, constraining all bonds")
        print("[DEBUG] Using hydrogen mass=4, constraining all bonds")
        constraint_type = ff.AllBonds
        hydrogen_mass = 4.0 * u.gram / u.mole
    else:
        logger.info("Enabling hydrogen mass=1, constraining bonds with hydrogen")
        print("[DEBUG] Using hydrogen mass=1, constraining bonds with hydrogen")
        constraint_type = ff.HBonds
        hydrogen_mass = None
    print("[DEBUG] Exiting _get_hydrogen_mass_and_constraints")
    return hydrogen_mass, constraint_type


def _create_openmm_system_implicit(
    parm_object,
    cutoff,
    use_big_timestep,
    use_bigger_timestep,
    implicit_solvent,
    remove_com,
    implicitSolventSaltConc,
    soluteDielectric,
    solventDielectric,
):
    print("[DEBUG] Entering _create_openmm_system_implicit")
    if cutoff is None:
        logger.info("Using no cutoff")
        print("[DEBUG] Using no cutoff")
        cutoff_type = ff.NoCutoff
        cutoff_dist = 999.0
    else:
        logger.info(f"Using a cutoff of {cutoff}")
        print(f"[DEBUG] Using a cutoff of {cutoff}")
        cutoff_type = ff.CutoffNonPeriodic
        cutoff_dist = cutoff

    hydrogen_mass, constraint_type = _get_hydrogen_mass_and_constraints(
        use_big_timestep, use_bigger_timestep
    )

    if implicit_solvent == "obc":
        logger.info('Using "OBC" implicit solvent')
        print('[DEBUG] Using "OBC" implicit solvent')
        implicit_type = app.OBC2
    elif implicit_solvent == "gbNeck":
        logger.info('Using "gbNeck" implicit solvent')
        print('[DEBUG] Using "gbNeck" implicit solvent')
        implicit_type = app.GBn
    elif implicit_solvent == "gbNeck2":
        logger.info('Using "gbNeck2" implicit solvent')
        print('[DEBUG] Using "gbNeck2" implicit solvent')
        implicit_type = app.GBn2
    elif implicit_solvent == "vacuum" or implicit_solvent is None:
        logger.info("Using vacuum instead of implicit solvent")
        print("[DEBUG] Using vacuum instead of implicit solvent")
        implicit_type = None
    else:
        RuntimeError("Should never get here")

    if implicitSolventSaltConc is None:
        implicitSolventSaltConc = 0.0
    if soluteDielectric is None:
        soluteDielectric = 1.0
    if solventDielectric is None:
        solventDielectric = 78.5

    print("[DEBUG] Creating implicit system with parm_object.createSystem")
    sys = parm_object.createSystem(
        nonbondedMethod=cutoff_type,
        nonbondedCutoff=cutoff_dist,
        constraints=constraint_type,
        implicitSolvent=implicit_type,
        removeCMMotion=remove_com,
        hydrogenMass=hydrogen_mass,
        implicitSolventSaltConc=implicitSolventSaltConc,
        soluteDielectric=soluteDielectric,
        solventDielectric=solventDielectric,
    )
    print("[DEBUG] Exiting _create_openmm_system_implicit")
    return sys


def _create_openmm_system_explicit(
    parm_object,
    cutoff,
    use_big_timestep,
    use_bigger_timestep,
    enable_pme,
    pme_tolerance,
    enable_pressure_coupling,
    pressure,
    pressure_couping_update_steps,
    remove_com,
    default_temperature,
):
    print("[DEBUG] Entering _create_openmm_system_explicit")
    if cutoff is None:
        raise ValueError("cutoff must be set for explicit solvent, but got None")
    else:
        if enable_pme:
            logger.info(f"Using PME with tolerance {pme_tolerance}")
            print(f"[DEBUG] Using PME with tolerance {pme_tolerance}")
            cutoff_type = ff.PME
        else:
            logger.info("Using reaction field")
            print("[DEBUG] Using reaction field")
            cutoff_type = ff.CutoffPeriodic

        logger.info(f"Using a cutoff of {cutoff}")
        print(f"[DEBUG] Using a cutoff of {cutoff}")
        cutoff_dist = cutoff

    hydrogen_mass, constraint_type = _get_hydrogen_mass_and_constraints(
        use_big_timestep, use_bigger_timestep
    )

    print("[DEBUG] Creating explicit system with parm_object.createSystem")
    s = parm_object.createSystem(
        nonbondedMethod=cutoff_type,
        nonbondedCutoff=cutoff_dist,
        constraints=constraint_type,
        implicitSolvent=None,
        removeCMMotion=remove_com,
        hydrogenMass=hydrogen_mass,
        rigidWater=True,
        ewaldErrorTolerance=pme_tolerance,
    )

    baro = None
    if enable_pressure_coupling:
        logger.info("Enabling pressure coupling")
        logger.info(f"Pressure is {pressure}")
        logger.info(
            f"Volume moves attempted every {pressure_couping_update_steps} steps"
        )
        print("[DEBUG] Enabling pressure coupling")
        baro = mm.MonteCarloBarostat(
            pressure, default_temperature, pressure_couping_update_steps
        )
        s.addForce(baro)

    print("[DEBUG] Exiting _create_openmm_system_explicit")
    return s, baro


def _create_integrator(temperature, use_big_timestep, use_bigger_timestep):
    print("[DEBUG] Entering _create_integrator")
    if use_big_timestep:
        logger.info("Creating integrator with 3.5 fs timestep")
        print("[DEBUG] Creating integrator with 3.5 fs timestep")
        timestep = 3.5 * u.femtosecond
    elif use_bigger_timestep:
        logger.info("Creating integrator with 4.5 fs timestep")
        print("[DEBUG] Creating integrator with 4.5 fs timestep")
        timestep = 4.5 * u.femtosecond
    else:
        logger.info("Creating integrator with 2.0 fs timestep")
        print("[DEBUG] Creating integrator with 2.0 fs timestep")
        timestep = 2.0 * u.femtosecond
    print("[DEBUG] Creating LangevinIntegrator")
    return mm.LangevinIntegrator(temperature * u.kelvin, 1.0 / u.picosecond, timestep)


def _create_gamd_integrator(options, system):
    print("[DEBUG] Entering _create_gamd_integrator")
    gamdIntegratorFactory = GamdIntegratorFactory()
    if options.use_big_timestep:
        logger.info("Creating custom integrator with 3.5 fs timestep")
        print("[DEBUG] Creating custom integrator with 3.5 fs timestep")
        timestep = 3.5 * u.femtosecond
    elif options.use_bigger_timestep:
        logger.info("Creating custom integrator with 4.5 fs timestep")
        print("[DEBUG] Creating custom integrator with 4.5 fs timestep")
        timestep = 4.5 * u.femtosecond
    else:
        logger.info("Creating custom integrator with 2.0 fs timestep")
        print("[DEBUG] Creating custom integrator with 2.0 fs timestep")
        timestep = 2.0 * u.femtosecond

    print("[DEBUG] Calling gamdIntegratorFactory.get_integrator")
    result = gamdIntegratorFactory.get_integrator(
        options.boost_type_str,
        system,
        options.default_temperature,
        timestep,
        options.conventional_md_prep,
        options.conventional_md,
        options.gamd_equilibration_prep,
        options.gamd_equilibration,
        options.total_simulation_length,
        options.averaging_window_interval,
        options.sigma0p,
        options.sigma0d,
    )
    [
        first_boost_group,
        second_boost_group,
        integrator,
        first_boost_type,
        second_boost_type,
    ] = result
    integrator.first_boost_group = first_boost_group
    integrator.second_boost_group = second_boost_group
    integrator.first_boost_type = first_boost_type
    integrator.second_boost_type = second_boost_type
    integrator.setRandomNumberSeed(options.random_seed)
    integrator.setFriction(options.friction_coefficient)
    print("[DEBUG] Exiting _create_gamd_integrator")
    return integrator


def _add_chains(topology, chain_list):
    print("[DEBUG] Entering _add_chains")
    # Verify that the input from Amber only has one chain
    assert len(list(topology.chains())) == 1

    newtop = app.Topology()

    # Add the chains to the new topology and
    # create a map between residues and chains.
    chain_map = {}
    for chain_info in chain_list:
        chain = newtop.addChain()
        for res in chain_info.residues.values():
            chain_map[res] = chain

    # Now we'll create a final chain for solvent, etc
    # and everything left to this last chain.
    last_chain = newtop.addChain()
    for res in topology.residues():
        if res.index not in chain_map:
            chain_map[res.index] = last_chain

    # Add all of the residues to the new topology, while
    # correcting the chain index.
    # Create a map between the old and new residues so
    # that we can add the atoms.
    residue_map = {}
    for residue in topology.residues():
        chain = chain_map[residue.index]
        new_residue = newtop.addResidue(residue.name, chain, residue.index)
        residue_map[residue] = new_residue

    # Now add back all of the atoms with tne new residues.
    # We keep a map between the old and new atoms so that
    # we can add the bonds.
    atom_map = {}
    for atom in topology.atoms():
        new_atom = newtop.addAtom(
            atom.name, atom.element, residue_map[atom.residue], atom.index
        )
        atom_map[atom] = new_atom

    # Now we add all of the bonds
    for bond in topology.bonds():
        newtop.addBond(atom_map[bond[0]], atom_map[bond[1]])

    print("[DEBUG] Exiting _add_chains")
    return newtop
