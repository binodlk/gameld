#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module to build AmberSubSystems from sequence or PDB file
"""

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List

import numpy as np  # type: ignore
from openmm import app  # type: ignore

from meld.system import indexing


class _AmberSubSystem(ABC):
    """
    Base class for other SubSystem classes.

    Provides functionality for translation/rotation and adding H-bonds.
    """

    def __init__(self):
        print("[DEBUG] _AmberSubSystem.__init__")
        self._translation_vector = np.zeros(3)
        self._rotatation_matrix = np.eye(3)
        self._disulfide_list = []
        self._general_bond = []
        self._prep_files = []
        self._frcmod_files = []
        self._lib_files = []
        self._info = []
        print("[DEBUG] _AmberSubSystem.__init__ complete")

    @abstractmethod
    def prepare_for_tleap(self, mol_id: str):
        """
        Prepare any inputs needed for tleap
        """
        pass

    @abstractmethod
    def generate_tleap_input(self, mol_id: str) -> List[str]:
        """
        Returns a list of tleap commands to run.
        """
        pass

    def set_translation(self, translation_vector: np.ndarray):
        print(f"[DEBUG] set_translation called: {translation_vector}")
        self._translation_vector = np.array(translation_vector)
        print(f"[DEBUG] set_translation: Vector is now {self._translation_vector}")

    def set_rotation(self, rotation_axis: np.ndarray, theta: float):
        print(f"[DEBUG] set_rotation called: axis={rotation_axis}, theta={theta}")
        theta = theta * 180 / math.pi
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        a = np.cos(theta / 2.0)
        b, c, d = -rotation_axis * np.sin(theta / 2.0)
        self._rotatation_matrix = np.array(
            [
                [
                    a * a + b * b - c * c - d * d,
                    2 * (b * c - a * d),
                    2 * (b * d + a * c),
                ],
                [
                    2 * (b * c + a * d),
                    a * a + c * c - b * b - d * d,
                    2 * (c * d - a * b),
                ],
                [
                    2 * (b * d - a * c),
                    2 * (c * d + a * b),
                    a * a + d * d - b * b - c * c,
                ],
            ]
        )
        print(f"[DEBUG] set_rotation: Rotation matrix is now\n{self._rotatation_matrix}")

    def add_bond(
        self,
        res_index_i: indexing.ResidueIndex,
        res_index_j: indexing.ResidueIndex,
        atom_name_i: str,
        atom_name_j: str,
        bond_type: str,
    ):
        print(f"[DEBUG] add_bond called: {res_index_i}, {res_index_j}, {atom_name_i}, {atom_name_j}, {bond_type}")
        assert isinstance(res_index_i, indexing.ResidueIndex)
        assert isinstance(res_index_j, indexing.ResidueIndex)
        self._general_bond.append(
            (int(res_index_i), int(res_index_j), atom_name_i, atom_name_j, bond_type)
        )
        print(f"[DEBUG] add_bond: Bond list is now {self._general_bond}")

    def add_disulfide(
        self, res_index_i: indexing.ResidueIndex, res_index_j: indexing.ResidueIndex
    ):
        print(f"[DEBUG] add_disulfide called: {res_index_i}, {res_index_j}")
        assert isinstance(res_index_i, indexing.ResidueIndex)
        assert isinstance(res_index_j, indexing.ResidueIndex)
        self._disulfide_list.append((int(res_index_i), int(res_index_j)))
        print(f"[DEBUG] add_disulfide: Disulfide list is now {self._disulfide_list}")

    def add_prep_file(self, fname: str):
        print(f"[DEBUG] add_prep_file called: {fname}")
        self._prep_files.append(fname)
        print(f"[DEBUG] add_prep_file: Prep list is now {self._prep_files}")

    def add_frcmod_file(self, fname: str):
        print(f"[DEBUG] add_frcmod_file called: {fname}")
        self._frcmod_files.append(fname)
        print(f"[DEBUG] add_frcmod_file: Frcmod list is now {self._frcmod_files}")

    def add_lib_file(self, fname: str):
        print(f"[DEBUG] add_lib_file called: {fname}")
        self._lib_files.append(fname)
        print(f"[DEBUG] add_lib_file: Lib list is now {self._lib_files}")

    def _gen_translation_string(self, mol_id: str) -> str:
        print(f"[DEBUG] _gen_translation_string called for mol_id={mol_id}")
        s = """translate {mol_id} {{ {x} {y} {z} }}""".format(
            mol_id=mol_id,
            x=self._translation_vector[0],
            y=self._translation_vector[1],
            z=self._translation_vector[2],
        )
        print(f"[DEBUG] _gen_translation_string returns: {s}")
        return s

    def _gen_rotation_string(self, mol_id: str) -> str:
        print(f"[DEBUG] _gen_rotation_string called for mol_id={mol_id}")
        # Current implementation is empty, for debugging just note the call
        return ""

    def _gen_bond_string(self, mol_id: str) -> List[str]:
        print(f"[DEBUG] _gen_bond_string called for mol_id={mol_id}")
        bond_strings = []
        for i, j, a, b, t in self._general_bond:
            d = f'bond {mol_id}.{i+1}.{a} {mol_id}.{j+1}.{b} "{t}"'
            bond_strings.append(d)
            print(f"[DEBUG] _gen_bond_string appending: {d}")
        return bond_strings

    def _gen_disulfide_string(self, mol_id: str) -> List[str]:
        print(f"[DEBUG] _gen_disulfide_string called for mol_id={mol_id}")
        disulfide_strings = []
        for i, j in self._disulfide_list:
            d = f"bond {mol_id}.{i+1}.SG {mol_id}.{j+1}.SG"
            disulfide_strings.append(d)
            print(f"[DEBUG] _gen_disulfide_string appending: {d}")
        return disulfide_strings

    def _gen_read_prep_string(self) -> List[str]:
        print("[DEBUG] _gen_read_prep_string called")
        prep_string = []
        for p in self._prep_files:
            s = f"loadAmberPrep {p}"
            prep_string.append(s)
            print(f"[DEBUG] _gen_read_prep_string appending: {s}")
        return prep_string

    def _gen_read_frcmod_string(self) -> List[str]:
        print("[DEBUG] _gen_read_frcmod_string called")
        frcmod_string = []
        for p in self._frcmod_files:
            s = f"loadAmberParams {p}"
            frcmod_string.append(s)
            print(f"[DEBUG] _gen_read_frcmod_string appending: {s}")
        return frcmod_string

    def _gen_read_lib_string(self) -> List[str]:
        print("[DEBUG] _gen_read_lib_string called")
        lib_string = []
        for p in self._lib_files:
            s = f"loadoff {p}"
            lib_string.append(s)
            print(f"[DEBUG] _gen_read_lib_string appending: {s}")
        return lib_string


class AmberSubSystemFromSequence(_AmberSubSystem):
    """
    Class to create a sub-system from sequence.
    """

    def __init__(self, sequence: str):
        print(f"[DEBUG] AmberSubSystemFromSequence.__init__ called with sequence: {sequence}")
        super(AmberSubSystemFromSequence, self).__init__()
        self._sequence = sequence
        sequence_len = len(sequence.split(" "))
        print(f"[DEBUG] sequence_len={sequence_len}")
        chain_info = indexing._ChainInfo({i: i for i in range(sequence_len)})
        self._info = indexing._SubSystemInfo(sequence_len, [chain_info])
        print("[DEBUG] AmberSubSystemFromSequence.__init__ complete")

    def prepare_for_tleap(self, mol_id: str):
        print(f"[DEBUG] AmberSubSystemFromSequence.prepare_for_tleap called for mol_id={mol_id}")
        # we don't need to do anything

    def generate_tleap_input(self, mol_id: str):
        print(f"[DEBUG] AmberSubSystemFromSequence.generate_tleap_input called for mol_id={mol_id}")
        leap_cmds = []
        leap_cmds.append("source leaprc.gaff")
        leap_cmds.extend(self._gen_read_frcmod_string())
        leap_cmds.extend(self._gen_read_prep_string())
        leap_cmds.extend(self._gen_read_lib_string())
        leap_cmd = f"{mol_id} = sequence {{ {self._sequence} }}"
        print(f"[DEBUG] generate_tleap_input appending: {leap_cmd}")
        leap_cmds.append(leap_cmd)
        leap_cmds.extend(self._gen_disulfide_string(mol_id))
        leap_cmds.extend(self._gen_bond_string(mol_id))
        rot_string = self._gen_rotation_string(mol_id)
        print(f"[DEBUG] generate_tleap_input appending rotation: {rot_string}")
        leap_cmds.append(rot_string)
        trans_string = self._gen_translation_string(mol_id)
        print(f"[DEBUG] generate_tleap_input appending translation: {trans_string}")
        leap_cmds.append(trans_string)
        print("[DEBUG] AmberSubSystemFromSequence.generate_tleap_input returning cmds")
        return leap_cmds


class AmberSubSystemFromPdbFile(_AmberSubSystem):
    """
    Create a new susbsystem from a pdb file.
    """

    def __init__(self, pdb_path: str):
        print(f"[DEBUG] AmberSubSystemFromPdbFile.__init__ called with pdb_path: {pdb_path}")
        super(AmberSubSystemFromPdbFile, self).__init__()

        print("[DEBUG] Opening pdb file for reading contents")
        with open(pdb_path) as pdb_file:
            self._pdb_contents = pdb_file.read()

        print("[DEBUG] Initializing app.PDBFile")
        pdb = app.PDBFile(pdb_path)
        topology = pdb.getTopology()
        residues = list(topology.residues())
        n_residues = len(residues)
        print(f"[DEBUG] PDB has {n_residues} residues")

        # get list of chainids
        chainids = []
        chain_to_res = defaultdict(list)
        for residue in residues:
            chainids.append(residue.chain.id)
            chain_to_res[residue.chain.id].append(residue.index)
        chainid_set = set(chainids)
        print(f"[DEBUG] chainids: {chainid_set}")

        # loop over the chainids in alphabetical order
        chains = []
        for chainid in sorted(chainid_set):
            chain_dict = {i: j for i, j in enumerate(chain_to_res[chainid])}
            print(f"[DEBUG] Chain {chainid} info: {chain_dict}")
            chain = indexing._ChainInfo(chain_dict)
            chains.append(chain)
        self._info = indexing._SubSystemInfo(n_residues, chains)
        print("[DEBUG] AmberSubSystemFromPdbFile.__init__ complete")

    def prepare_for_tleap(self, mol_id):
        print(f"[DEBUG] AmberSubSystemFromPdbFile.prepare_for_tleap called for mol_id={mol_id}")
        pdb_path = f"{mol_id}.pdb"
        print(f"[DEBUG] Writing PDB file to {pdb_path}")
        with open(pdb_path, "w") as pdb_file:
            pdb_file.write(self._pdb_contents)
        print(f"[DEBUG] PDB file written to {pdb_path}")

    def generate_tleap_input(self, mol_id):
        print(f"[DEBUG] AmberSubSystemFromPdbFile.generate_tleap_input called for mol_id={mol_id}")
        leap_cmds = []
        leap_cmds.append("source leaprc.gaff")
        leap_cmds.extend(self._gen_read_frcmod_string())
        leap_cmds.extend(self._gen_read_prep_string())
        leap_cmds.extend(self._gen_read_lib_string())
        leap_cmd = f"{mol_id} = loadPdb {mol_id}.pdb"
        print(f"[DEBUG] generate_tleap_input appending: {leap_cmd}")
        leap_cmds.append(leap_cmd)
        leap_cmds.extend(self._gen_bond_string(mol_id))
        leap_cmds.extend(self._gen_disulfide_string(mol_id))
        rot_string = self._gen_rotation_string(mol_id)
        print(f"[DEBUG] generate_tleap_input appending rotation: {rot_string}")
        leap_cmds.append(rot_string)
        trans_string = self._gen_translation_string(mol_id)
        print(f"[DEBUG] generate_tleap_input appending translation: {trans_string}")
        leap_cmds.append(trans_string)
        print("[DEBUG] AmberSubSystemFromPdbFile.generate_tleap_input returning cmds")
        return leap_cmds
