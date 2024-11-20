import os
import numpy as np
from general_functions import get_params_dict, get_radius_number, load_rad_dict
import concurrent.futures
from itertools import repeat
import argparse
from pathlib import Path
import pickle

def find_ligand_file_in_folder(folder_path):
    ligand_file_list = []
    filenames = next(os.walk(folder_path), (None, None, []))[2]
    for filename in filenames:
        if filename.endswith("ligands.mol2") or filename.endswith("final.mol2"):
            ligand_file_list.append(os.path.join(folder_path, filename))
    return ligand_file_list


def load_atoms_mol2(filename, rad_dict, save_path, ligand_type='ligand'):
    coord_start = 0
    max_atoms = 0
    n_atoms = 0
    n_molecules = 0
    n_unique_molecules = 0
    same_molec_counter = 1
    molecule_name_list = []
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
                if molec_suffix =="_0":
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


def preprocess_ligands_one_target(ligand_file_path, rad_dict, conf_num, ligand_type='ligand'):
    verbose = False
    suffix = get_suffix(conf_num)
    target_folder = os.path.dirname(ligand_file_path)
    output_folder = os.path.join(target_folder, f"preprocessed_ligands{suffix}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    load_atoms_mol2(ligand_file_path, rad_dict, output_folder, ligand_type=ligand_type)
    if verbose:
        print("Files saved to: ", output_folder)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-lp", '--ligand_path', type=str, help='Path to folder containing ligand folders')
    group.add_argument("-fp", '--folder_path', type=str, help='Path to folder containing ligand files')
    parser.add_argument("-lt", '--ligand_type', type=str, help='Ligand type (not required)')
    parser.add_argument("-sd", '--subdirectories', action='store_true',
                        help="Include subdirectories if target path is specified")

    args = parser.parse_args()
    if args.ligand_path and args.subdirectories:
        parser.error(
            "Argument -sd (subdirectories) can only be used with -tp (target path), not with -lp (ligand path).")
    if args.ligand_type and args.folder_path:
        parser.error(
            "Argument -lt (ligand type) can only be used with -lp (ligand path), not with -fp (folder path).")

    if args.folder_path:
        folder_path = args.folder_path
        subdirectories = args.subdirectories
        main(folder_path=folder_path, subdirectories=subdirectories)
    elif args.ligand_path:
        ligand_file_path = args.ligand_path
        ligand_type = args.ligand_type
        main(ligand_file_path=ligand_file_path, ligand_type=ligand_type)


def main(ligand_file_path=None, ligand_type='ligand', folder_path=None, subdirectories=None):
    root_software_path = Path(__file__).resolve().parents[1]
    os.chdir(root_software_path)
    deps_folder = os.path.join(root_software_path, 'deps')
    config_file = os.path.join(deps_folder, 'config.txt')
    params_dict = get_params_dict(config_file)
    conf_num = params_dict["CONFORMERS_PER_MOLECULE"]
    rad_dict = load_rad_dict(os.path.join(deps_folder, 'atom_type_radius.json'))

    if ligand_file_path:
        preprocess_ligands_one_target(ligand_file_path, rad_dict, conf_num, ligand_type=ligand_type)
    elif folder_path:
        if subdirectories:
            ligand_file_list = []
            folders = next(os.walk(folder_path))[1]
            for folder in folders:
                temp_ligand_file_list = find_ligand_file_in_folder(os.path.join(folder_path, folder))
                ligand_file_list.extend(temp_ligand_file_list)
        else:
            ligand_file_list = find_ligand_file_in_folder(folder_path)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(preprocess_ligands_one_target, ligand_file_list, repeat(rad_dict), repeat(conf_num))
        # preprocess_ligands_one_target(ligand_file_list[0], rad_dict, conf_num)


if __name__ == "__main__":
    get_args()