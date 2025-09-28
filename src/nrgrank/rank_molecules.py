import os
import numpy as np
from numba import njit
import timeit
import argparse
import math as m
import pickle
from nrgrank.general_functions import write_pdb

# def njit(njit):
#     return njit


@njit
def center_coords(ligand_atoms_xyz, list_size):
    length = list_size
    centered_coord = np.zeros((list_size, 3), dtype=np.float32)
    sum_x = np.sum(ligand_atoms_xyz[:, 0]) / length
    sum_y = np.sum(ligand_atoms_xyz[:, 1]) / length
    sum_z = np.sum(ligand_atoms_xyz[:, 2]) / length
    centroid_coords = np.array([sum_x, sum_y, sum_z])
    for i in range(len(ligand_atoms_xyz)):
        centered_coord[i] = ligand_atoms_xyz[i] - centroid_coords
    return centered_coord


def load_ligands(target_path, ligand_type, start, end, conf_num, path_to_ligands=None):
    if not path_to_ligands:
        if conf_num == 0:
            print('Ligands are in an old path and conf number is 0')
            path_to_ligands = os.path.join(target_path, f"preprocessed_ligands")
            if not os.path.isdir(path_to_ligands):
                exit('Could not find ligands folder')
        else:
            ligand_folder = f"preprocessed_ligands_{conf_num}_conf"
            path_to_ligands = os.path.join(target_path, ligand_folder)

    with open(os.path.join(path_to_ligands, f"{ligand_type}_atom_name.pkl"), 'rb') as f:
        atom_name = pickle.load(f)[start:end].copy()
    with open(os.path.join(path_to_ligands, f"{ligand_type}_molecule_name.pkl"), 'rb') as f:
        molecule_name = pickle.load(f)[start:end].copy()
    atom_type = np.load(os.path.join(path_to_ligands, f"{ligand_type}_atom_type.npy"), mmap_mode='r')[start:end].copy()
    atom_xyz = np.load(os.path.join(path_to_ligands, f"{ligand_type}_atom_xyz.npy"), mmap_mode='r')[start:end].copy()
    atoms_num_per_ligand = np.load(os.path.join(path_to_ligands, f"{ligand_type}_atoms_num_per_ligand.npy"),
                                   mmap_mode='r')[start:end].copy()
    ligand_count = len(atom_type)
    return atom_name, atom_type, atom_xyz, molecule_name, atoms_num_per_ligand, ligand_count


def Rx(theta):
    return np.matrix([[1, 0, 0], [0, m.cos(theta), -m.sin(theta)], [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)], [0, 1, 0], [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0], [m.sin(theta), m.cos(theta), 0], [0, 0, 1]])


def rotate_ligand(ligand_atoms_xyz, n_rotations):
    centered_ligand_atoms_xyz = center_coords(ligand_atoms_xyz, len(ligand_atoms_xyz))
    single_rotation = 360/n_rotations
    rotation_counter = 0
    rotated_ligand_coord_list = np.zeros((n_rotations**3, len(centered_ligand_atoms_xyz), 3), dtype=np.float32)
    for x_rot_count in range(n_rotations):
        for y_rot_count in range(n_rotations):
            for z_rot_count in range(n_rotations):
                x = np.deg2rad(x_rot_count*single_rotation)
                y = np.deg2rad(y_rot_count*single_rotation)
                z = np.deg2rad(z_rot_count*single_rotation)
                rotation_matrix = Rx(x) * Ry(y) * Rz(z)
                for i, coord in enumerate(centered_ligand_atoms_xyz):
                    rotated_ligand_coord_list[rotation_counter][i] = np.array(np.dot(rotation_matrix, coord))
                rotation_counter += 1
    rotated_ligand_coord_list_unique = np.unique(rotated_ligand_coord_list, axis=0)
    return rotated_ligand_coord_list_unique


@njit
def get_cf(lig_pose, point, cf_size_list, precalc_cf_list, ligand_atoms_types, default_cf, cell_width, min_xyz):
    cf = 0.0
    lig_pose = lig_pose + point
    x_index_array = ((lig_pose[:, 0] - min_xyz[0]) / cell_width).astype(np.int32)
    y_index_array = ((lig_pose[:, 1] - min_xyz[1]) / cell_width).astype(np.int32)
    z_index_array = ((lig_pose[:, 2] - min_xyz[2]) / cell_width).astype(np.int32)
    if np.min(x_index_array) < 0 or np.min(y_index_array) < 0 or np.min(z_index_array) < 0:
        cf = default_cf
    elif np.max(x_index_array) >= cf_size_list[0] or np.max(y_index_array) >= cf_size_list[1] or np.max(z_index_array) >= cf_size_list[2]:
        cf = default_cf
    else:
        for counter, _ in enumerate(x_index_array):
            temp_cf = precalc_cf_list[x_index_array[counter]][y_index_array[counter]][z_index_array[counter]][ligand_atoms_types[counter]-1]
            if temp_cf == default_cf:
                cf = default_cf
                break
            else:
                cf += temp_cf
    return cf


@njit
def get_cf_with_clash(lig_pose, point, load_range_list, grid_spacing, cf_size_list, load_cf_list, ligand_atoms_types,
                      default_cf, cell_width, min_xyz, clash_list, clash_list_size, num_atoms):
    ###### CHECK CLASH ######
    x_index_array = np.empty_like(lig_pose[:, 0])
    np.round(((lig_pose[:, 0] + point[0] - load_range_list[0][0]) / grid_spacing), 0, x_index_array)  # .astype(np.int32)
    y_index_array = np.empty_like(lig_pose[:, 1])
    np.round(((lig_pose[:, 1] + point[1] - load_range_list[1][0]) / grid_spacing), 0, y_index_array)  # .astype(np.int32)
    z_index_array = np.empty_like(lig_pose[:, 2])
    np.round(((lig_pose[:, 2] + point[2] - load_range_list[2][0]) / grid_spacing), 0, z_index_array)  # .astype(np.int32)
    x_index_array = x_index_array.astype(np.int32)
    y_index_array = y_index_array.astype(np.int32)
    z_index_array = z_index_array.astype(np.int32)
    x_index_array[x_index_array == cf_size_list[0]] -= 1
    y_index_array[y_index_array == cf_size_list[1]] -= 1
    z_index_array[z_index_array == cf_size_list[2]] -= 1
    if np.min(x_index_array) < 0 or np.min(y_index_array) < 0 or np.min(z_index_array) < 0:
        return default_cf
    elif np.max(x_index_array) >= clash_list_size[0] or np.max(y_index_array) >= clash_list_size[1] or np.max(z_index_array) >= clash_list_size[2]:
        return default_cf
    else:
        for number in np.arange(0, num_atoms, 1):
            clash_detect = clash_list[x_index_array[number]][y_index_array[number]][z_index_array[number]]
            if clash_detect:
                return default_cf
        else:
            cf = 0.0
            lig_pose = lig_pose + point
            x_index_array = ((lig_pose[:, 0] - min_xyz[0]) / cell_width).astype(np.int32)
            y_index_array = ((lig_pose[:, 1] - min_xyz[1]) / cell_width).astype(np.int32)
            z_index_array = ((lig_pose[:, 2] - min_xyz[2]) / cell_width).astype(np.int32)
            for counter, _ in enumerate(x_index_array):
                temp_cf = load_cf_list[x_index_array[counter]][y_index_array[counter]][z_index_array[counter]][ligand_atoms_types[counter]-1]
                if temp_cf == default_cf:
                    cf = default_cf
                    break
                else:
                    cf += temp_cf
            return cf


@njit
def get_cf_main(binding_site_grid, ligand_orientations, cf_size_list, n_cf_evals, load_cf_list, ligand_atoms_types,
                default_cf, cell_width, min_xyz):
    cfs_list = np.zeros((n_cf_evals, 3), dtype=np.float32)
    counter = 0
    for point_index, point in enumerate(binding_site_grid):
        for pose_index, lig_pose in enumerate(ligand_orientations):
            cf = get_cf(lig_pose, point, cf_size_list, load_cf_list, ligand_atoms_types, default_cf, cell_width, min_xyz)
            cfs_list[counter][0] = cf
            cfs_list[counter][1] = pose_index
            cfs_list[counter][2] = point_index
            counter += 1
    return cfs_list


@njit
def get_cf_main_clash(binding_site_grid, ligand_orientations, cf_size_list, n_cf_evals, load_cf_list,
                      ligand_atoms_types, default_cf, cell_width, min_xyz, load_range_list, preload_grid_distance,
                      clash_list, clash_list_size, num_atoms):
    cfs_list = np.zeros((n_cf_evals, 3), dtype=np.float32)
    counter = 0
    for point_index, point in enumerate(binding_site_grid):
        for pose_index, lig_pose in enumerate(ligand_orientations):
            cf = get_cf_with_clash(lig_pose, point, load_range_list, preload_grid_distance, cf_size_list, load_cf_list,
                                   ligand_atoms_types, default_cf, cell_width, min_xyz, clash_list, clash_list_size,
                                   num_atoms)
            cfs_list[counter][0] = cf
            cfs_list[counter][1] = pose_index
            cfs_list[counter][2] = point_index
            counter += 1
    return cfs_list


def main(target_name, preprocessed_target_path, preprocessed_ligand_path, result_folder_path,
         result_csv_and_pose_name=None, ligand_type='ligand', ligand_slice=None, write_info=True, write_file=True,
         file_separator=',', output_header=True, output_dictionary=True, unique_run_id=None, **user_config):
    """

    Parameters:
        target_name (str): The name of the target. Used for naming the output CSV file when csv_and_pose_name is not specified.
        preprocessed_target_path (str): Path to the preprocessed target folder from process_target().
        preprocessed_ligand_path (str): Path to the preprocessed ligand folder from process_ligand().
        result_folder_path (str): Directory path where the docking results will be stored.
        result_csv_and_pose_name (str, optional): Custom name for CSV and pose files. Default is None.
        ligand_type (str, optional): Type of ligand being processed (e.g., "ligand"). Default is 'ligand'. Useful when benchmarking with DUD-E
        ligand_slice (list[int] or None, optional): Slice range of ligands to process. Default is None.
        write_info (bool, optional): Write informational remarks or not. Default is True.
        write_file (bool, optional): Output a file or not. Default is True.
        output_dictionary (bool, optional): Output a dictionary or not. Default is True.
        file_separator (str, optional): Separator character for file. Default is ',', '\t' can be used to make a tsv.
        output_header (bool, optional): Flag to write header row in CSV file or not. Default is True.
        unique_run_id (str, optional): Unique identifier for the run to avoid file name conflictsand also used to name ligand poses. Default is None.
        **user_config: Arbitrary keyword arguments for overriding default docking parameters.

    Raises:
        FileNotFoundError: If an expected file is not found in the preprocessed target path.
        ValueError: If protein or ligand data integrity checks fail.
        TypeError: If input parameter types are incorrect.

    Returns:
        None

    """
    if not write_file and not output_dictionary:
        raise TypeError("At least one of write_file or output_dictionary must be specified. "
                        "Both can be True at the same time.")

    time_start = timeit.default_timer()
    save_time = False
    default_cf = 100000000
    info_lines = []
    output_lines = []

    params_dict_default = {
        'USE_CLASH': True,
        'LIGAND_ROTATIONS_PER_AXIS': 9,
        'LIGAND_TEST_DOT_SEPARATION': 1.5,
        'CLASH_DOT_DISTANCE': 0.25,
        'CONFORMERS_PER_MOLECULE': 1,
        'POSES_SAVED_PER_MOLECULE': 1,
        'WRITE_LIGAND_TEST_DOTS': False,
        'VERBOSE': False,
        'SAVE_TOTAL_TIME': False
    }
    params_dict = params_dict_default.copy()
    params_dict.update(user_config)

    test_dot_separation = params_dict['LIGAND_TEST_DOT_SEPARATION']
    conf_num = params_dict['CONFORMERS_PER_MOLECULE']
    if preprocessed_ligand_path.find('_conf') != -1:
        conf_num = int(preprocessed_ligand_path.split('_conf')[0].split('_')[-1])
    poses_saved_per_molecule = params_dict["POSES_SAVED_PER_MOLECULE"]
    use_clash = params_dict["USE_CLASH"]
    ligand_rotations_per_axis = params_dict["LIGAND_ROTATIONS_PER_AXIS"]
    clash_dot_distance = params_dict["CLASH_DOT_DISTANCE"]
    write_ligand_test_dots = params_dict["WRITE_LIGAND_TEST_DOTS"]
    if not result_csv_and_pose_name:
        ligand_pose_save_path = os.path.join(result_folder_path, 'ligand_poses')
    else:
        ligand_pose_save_path = os.path.join(result_folder_path, result_csv_and_pose_name)

    if not os.path.exists(preprocessed_target_path):
        raise FileNotFoundError(f'{preprocessed_target_path} does not exist')
    if not os.path.isdir(preprocessed_target_path):
        raise IsADirectoryError(f'{preprocessed_target_path} is not directory, expected a directory')
    if not os.path.exists(preprocessed_ligand_path):
        raise FileNotFoundError(f'{preprocessed_ligand_path} does not exist')
    if not os.path.isdir(preprocessed_ligand_path):
        raise IsADirectoryError(f'{preprocessed_ligand_path} is not a directory, expected a directory')

    if not os.path.isdir(result_folder_path):
        os.makedirs(result_folder_path)
    if not ligand_slice:
        ligand_slice = [0, None]
    start = ligand_slice[0]
    end = ligand_slice[1]
    if result_csv_and_pose_name:
        output_file_basename = result_csv_and_pose_name
    else:
        output_file_basename = target_name
        if conf_num > 1:
            output_file_basename += f"_{conf_num}_conf"
        if end:
            output_file_basename += f"_split_{start}_{end}"
        if unique_run_id:
            output_file_basename += f"_run_{unique_run_id}"
    if file_separator == ',':
        output_file_path = os.path.join(result_folder_path, f'{output_file_basename}.csv')
    elif file_separator == '\t':
        output_file_path = os.path.join(result_folder_path, f'{output_file_basename}.tsv')
    else:
        output_file_path = os.path.join(result_folder_path, f'{output_file_basename}.txt')
    counter = 1
    duplicate_file = False
    while os.path.isfile(output_file_path):
        counter += 1
        output_file_path = os.path.join(result_folder_path, f'{output_file_basename}_({counter}).csv')
        duplicate_file = True
    if duplicate_file:
        ligand_pose_save_path += f"_({counter})"

    binding_site_grid = np.load(os.path.join(preprocessed_target_path, f"ligand_test_dots_{test_dot_separation}.npy"))
    precalculated_cf_list = np.load(os.path.join(preprocessed_target_path, f"cf_list.npy"))

    if use_clash:
        load_range_list = np.load(os.path.join(preprocessed_target_path, "bd_site_cuboid_coord_range_array.npy"))
        clash_list = np.load(os.path.join(preprocessed_target_path, f"clash_list_{clash_dot_distance}.npy"))
        clash_list_size = clash_list.shape
    else:
        load_range_list = None
        clash_list = None
        clash_list_size = None

    if write_ligand_test_dots:
        write_pdb(binding_site_grid, "ligand_test_dots", ligand_pose_save_path, None, None)
    n_cf_evals = len(binding_site_grid) * ligand_rotations_per_axis**3
    atom_name_array, atom_type_array, atom_xyz_array, molecule_name_array, atoms_per_molecule_array, molecule_count_array \
        = load_ligands(preprocessed_target_path, ligand_type, start, end, conf_num, path_to_ligands=preprocessed_ligand_path)
    cfs_list_by_ligand = np.zeros(molecule_count_array, dtype=np.float32)
    if save_time:
        time_list = np.zeros(molecule_count_array, dtype=np.float32)

    cell_width = np.load(os.path.join(preprocessed_target_path, 'index_cube_cell_width.npy'))
    cf_size_list = np.array([np.size(precalculated_cf_list, axis=0), np.size(precalculated_cf_list, axis=1), np.size(precalculated_cf_list, axis=2)])

    info_lines.append(f"REMARK target folder: {preprocessed_target_path}")
    info_lines.append(f"REMARK software: {os.path.basename(__file__)}")
    info_lines.append(f"REMARK ligand type: {ligand_type}")
    info_lines.append(f"REMARK number of conformers: {conf_num}")
    info_lines.append(f"REMARK rotations per axis: {ligand_rotations_per_axis}")
    info_lines.append(f"REMARK dot separation: {test_dot_separation} A")
    info_lines.append(f"REMARK preloaded grid distance: {clash_dot_distance} A")
    info_lines.append(f"REMARK Total binding site grid dots: {len(binding_site_grid)}")
    info_lines.append(f"REMARK Total CF evaluations per ligand: {n_cf_evals}")
    info_lines.append(f"REMARK use clash: {use_clash}")
    info_lines.append(f"REMARK index cube width: {cell_width}")

    if params_dict['VERBOSE']:
        print("\n".join(info_lines))
    min_xyz = np.load(os.path.join(preprocessed_target_path, 'index_cube_min_xyz.npy'))

    for i, molecule in enumerate(atoms_per_molecule_array):
        time_molecule_start = timeit.default_timer()
        molecule_atom_count = atoms_per_molecule_array[i]
        molecule_atom_xyz = atom_xyz_array[i][0:molecule_atom_count]
        molecule_atom_types = atom_type_array[i][0:molecule_atom_count]
        molecule_rotations = rotate_ligand(molecule_atom_xyz, ligand_rotations_per_axis)
        num_atoms = len(molecule_rotations[0])
        if not use_clash:
            cfs_list = get_cf_main(binding_site_grid, molecule_rotations, cf_size_list, n_cf_evals, precalculated_cf_list,
                                   molecule_atom_types, default_cf, cell_width, min_xyz)
        else:
            cfs_list = get_cf_main_clash(binding_site_grid, molecule_rotations, cf_size_list, n_cf_evals,
                                         precalculated_cf_list, molecule_atom_types, default_cf, cell_width, min_xyz,
                                         load_range_list, clash_dot_distance, clash_list,
                                         clash_list_size, num_atoms)
        cfs_list_by_ligand[i] = np.min(cfs_list[:, 0])
        if poses_saved_per_molecule > 0:
            sorted_indices = np.argsort(cfs_list[:, 0])[:poses_saved_per_molecule]
            molecule_atoms_names = atom_name_array[i][0:molecule_atom_count]
            if poses_saved_per_molecule == 1:
                molecule_save_folder = ligand_pose_save_path
            else:
                molecule_save_folder = os.path.join(ligand_pose_save_path, molecule_name_array[i])
                if not os.path.isdir(molecule_save_folder):
                    os.makedirs(molecule_save_folder)
            for pose_number in range(0, poses_saved_per_molecule, 1):
                pose_index = sorted_indices[pose_number]
                pose_info = cfs_list[pose_index]
                pose_rotation_number = int(pose_info[1])
                translated_coords = np.zeros((len(molecule_rotations[pose_rotation_number]), 3), dtype=np.float32)
                for atom in range(len(molecule_rotations[pose_rotation_number])):
                    translated_coords[atom] = np.add(molecule_rotations[pose_rotation_number][atom],
                                                     binding_site_grid[int(pose_info[2])])
                pose_file_name = molecule_name_array[i]
                if conf_num == 1:
                    if pose_file_name.endswith('0'):
                        pose_file_name = pose_file_name.rsplit('_', 1)[0]
                pose_file_name += f'_{unique_run_id}'
                if poses_saved_per_molecule != 1:
                    pose_file_name += f'_pose_{pose_number+1}'
                extra_info = [
                              f"REMARK CF: {cfs_list[sorted_indices[pose_number]][0]:.2f}\n",
                              f"REMARK atom types: {np.array2string(molecule_atom_types, separator=' ', 
                                                                    max_line_width=2000).strip('[]')}\n"
                ]
                if unique_run_id:
                    extra_info.append(f"REMARK unique_run_ID: {unique_run_id}\n")
                write_pdb(translated_coords, pose_file_name, molecule_save_folder, molecule_atoms_names, extra_info)
        if save_time:
            time_list[i] = timeit.default_timer() - time_molecule_start

    if params_dict['SAVE_TOTAL_TIME']:
        info_lines.append(f"REMARK total screen time: {timeit.default_timer() - time_start:.3f} seconds")
        if params_dict['VERBOSE']:
            print(output_lines[-1])
    molecule_name_cleaned = [molecule_name.rsplit('_', 1)[0] for molecule_name in molecule_name_array]
    molecule_conformer_number = [molecule_name.rsplit('_', 1)[1] for molecule_name in molecule_name_array]
    if write_info:
        info_file_path = os.path.splitext(output_file_path)[0] + f'_info.txt'
        with open(info_file_path, "w") as f:
            f.writelines("\n".join(info_lines))
    if write_file:
        file_header = "Name,Score"
        if ligand_type != 'ligand':
            file_header += ',Type'
        if conf_num > 1:
            file_header += ",Conformer number"
        if save_time:
            file_header += ",Time"
        if not write_file:
            file_header += ",Binding site"
        if output_header:
            output_lines.append(file_header)
        for z, ligand in enumerate(atoms_per_molecule_array):
            output = f'"{molecule_name_cleaned[z]}",{cfs_list_by_ligand[z]:.0f}'
            if ligand_type != 'ligand':
                output += f',{ligand_type}'
            if conf_num > 1:
                output += f",{molecule_conformer_number[z]}"
            if save_time:
                output += f",{time_list[z]:.3f}"
            if not write_file:
                output += f",{target_name}"
            output_lines.append(output)
        if file_separator != ',':
            output_lines = [line.replace(',', file_separator) for line in output_lines]
        if write_file:
            with open(output_file_path, "w") as f:
                f.writelines("\n".join(output_lines))
                f.write("\n")
    else:
        output_file_path = None
    if params_dict['VERBOSE']:
        print("\n".join(output_lines))
    if output_dictionary:
        output_dictionary = {'Names': molecule_name_cleaned, 'Score': cfs_list_by_ligand, 'Type': ligand_type,
                             'Binding site': target_name}
        if conf_num > 1:
            output_dictionary['Conformer number'] = molecule_conformer_number
        if save_time:
            output_dictionary['Time'] = time_list
    else:
        output_dictionary = None
    return output_file_path, output_dictionary
