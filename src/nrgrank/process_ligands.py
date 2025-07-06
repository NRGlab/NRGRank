import os
import numpy as np
from nrgrank.general_functions import get_radius_number, load_rad_dict
import argparse
import pickle
import re


def load_atoms_mol2(filename, save_path, ligand_type='ligand'):
    coord_start = 0
    max_atoms = 0
    n_atoms = 0
    n_molecules = 0
    n_unique_molecules = 0
    same_molec_counter = 1
    molecule_name_list = []
    rad_dict = load_rad_dict()
    with open(filename) as f:
        lines = f.readlines()

    for counter, line in enumerate(lines):
        if line.startswith('@<TRIPOS>MOLECULE'):
            n_molecules += 1
            molecule_name = lines[counter+1][0:-1]
            if molecule_name != "\n":
                molec_suffix = "_0"
                if n_molecules > 1 and molecule_name_list[n_molecules-2].split("_")[0] == molecule_name:
                    molec_suffix = f"_{same_molec_counter}"
                    same_molec_counter += 1
                else:
                    same_molec_counter = 1
                molecule_name_list.append(lines[counter+1][0:-1] + molec_suffix)
                if molec_suffix == "_0":
                    n_unique_molecules += 1
            else:
                exit("Error when reading molecule name")
        elif line.startswith("@<TRIPOS>ATOM"):
            coord_start = 1
            if max_atoms < n_atoms:
                max_atoms = n_atoms
            n_atoms = 0
        if coord_start == 1:
            if line.startswith("@<TRIPOS>ATOM") is False:
                if line[0] == "@":
                    coord_start = 0
                elif line.split()[1][0] != "H":
                    n_atoms += 1
    if max_atoms < n_atoms:
        max_atoms = n_atoms

    n_atom_array = np.zeros(n_molecules, dtype=np.int32)
    atoms_xyz = np.full((n_molecules, max_atoms, 3), 9999, dtype=np.float32)
    atoms_type = np.full((n_molecules, max_atoms), -1, dtype=np.int32)

    molecule_counter = -1
    atom_counter = 0
    coord_start = 0
    atom_name_list = []
    temp_atom_name_list = []
    atoms_name_count = {}
    for counter, line in enumerate(lines):
        if line.startswith('@<TRIPOS>MOLECULE'):
            atoms_name_count = {}
            molecule_counter += 1
            if molecule_counter > 0:
                n_atom_array[molecule_counter-1] = atom_counter
                atom_counter = 0
                coord_start = 0
                atom_name_list.append(temp_atom_name_list)
                temp_atom_name_list = []
        if line.startswith('@<TRIPOS>ATOM') and line[0] != "\n":
            coord_start = 1
        if coord_start == 1 and line.startswith('@<TRIPOS>ATOM') is False:
            if line[0] != '@':
                line = line.split()
                if line[5][0] != 'H':
                    atoms_xyz[molecule_counter][atom_counter] = np.array([float(line[2]),
                                                                          float(line[3]),
                                                                          float(line[4])])
                    atom_type = line[5]
                    atoms_type_temp, _ = get_radius_number(atom_type, rad_dict)
                    atoms_type[molecule_counter][atom_counter] = atoms_type_temp
                    atm_name = atom_type.split(".")[0]
                    if atm_name in atoms_name_count:
                        atoms_name_count[atm_name] += 1
                    else:
                        atoms_name_count[atm_name] = 1
                    temp_atom_name_list.append(f"{atm_name}{atoms_name_count[atm_name]}")
                    atom_counter += 1
            else:
                coord_start = 0
    n_atom_array[-1] = atom_counter
    atom_name_list.append(temp_atom_name_list)
    np.save(os.path.join(save_path, f"{ligand_type}_atom_xyz"), atoms_xyz)
    np.save(os.path.join(save_path, f"{ligand_type}_atom_type"), atoms_type)
    with open(os.path.join(save_path, f"{ligand_type}_molecule_name.pkl"), 'wb') as f:
        pickle.dump(molecule_name_list, f)
    with open(os.path.join(save_path, f"{ligand_type}_atom_name.pkl"), 'wb') as f:
        pickle.dump(atom_name_list, f)
    np.save(os.path.join(save_path, f"{ligand_type}_atoms_num_per_ligand"), n_atom_array)
    if ligand_type != 'ligand':
        np.save(os.path.join(save_path, f"{ligand_type}_ligand_count"), np.array([n_unique_molecules]))


def get_suffix(conf_num):
    suffix = ""
    if conf_num != 0:
        suffix = f"_{conf_num}_conf"
    return suffix


def get_suffix_search_in_file_name(filepath):
    pattern = r'(_\d+_conf)'
    match = re.search(pattern, filepath)
    return match.group(1)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-l", '--ligand_path', type=str, help='Path to folder containing ligand folders')
    parser.add_argument("-t", '--ligand_type', type=str, help='Ligand type (not required)')
    parser.add_argument("-c", '--conformers_per_molecule', type=int, help='Number of conformers per molecule')
    parser.add_argument("-o", '--output_dir', type=str, help='Output directory')

    args = parser.parse_args()
    ligand_file_path = args.ligand_path
    ligand_type = args.ligand_type
    conformers_per_molecule = args.conformers_per_molecule
    output_dir = args.output_dir
    main(ligand_path=ligand_file_path, ligand_type=ligand_type, conformers_per_molecule=conformers_per_molecule, output_dir=output_dir)


def main(ligand_path, conformers_per_molecule, overwrite=False, ligand_type='ligand', output_dir=None):
    if os.path.isfile(ligand_path):
        if ligand_path.find('_conf') != -1:
            suffix = get_suffix_search_in_file_name(ligand_path)
        else:
            suffix = get_suffix(conformers_per_molecule)
        if output_dir is None:
            output_dir = os.path.dirname(ligand_path)
        output_folder = os.path.join(output_dir, f"preprocessed_ligands{suffix}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if os.listdir(output_folder) and not overwrite:
            print(f'{output_folder} is not empty... Skipping. \nUse overwrite=True to overwrite.')
        else:
            load_atoms_mol2(ligand_path, output_folder, ligand_type=ligand_type)
        return output_folder
    else:
        exit(f'Argument used for ligand_path is not a file: {ligand_path}')


if __name__ == "__main__":
    get_args()
