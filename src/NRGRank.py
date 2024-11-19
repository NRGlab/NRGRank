import os
import shutil
import numpy as np
from numba import njit
import timeit
from pathlib import Path
import argparse
import math as m
import pickle
from general_functions import get_params_dict, write_pdb

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


def load_ligands(target_path, ligand_type, start, end, conf_num, output_pose, path_to_ligands=None):
    if not path_to_ligands:
        if conf_num == 0:
            print('Ligands are in an old path and conf number is 0')
            path_to_ligands = os.path.join(target_path, f"preprocessed_ligands")
            if not os.path.isdir(path_to_ligands):
                exit('Could not find ligands folder')
        else:
            ligand_folder = f"preprocessed_ligands_{conf_num}_conf"
            path_to_ligands = os.path.join(target_path, ligand_folder)

    atom_name = []
    if output_pose:
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
def get_cf_main_clash(binding_site_grid, ligand_orientations, cf_size_list, n_cf_evals, load_cf_list, ligand_atoms_types,
                default_cf, cell_width, min_xyz, load_range_list, preload_grid_distance, clash_list,
                clash_list_size, num_atoms):
    cfs_list = np.zeros((n_cf_evals, 3), dtype=np.float32)
    counter = 0
    for point_index, point in enumerate(binding_site_grid):
        for pose_index, lig_pose in enumerate(ligand_orientations):
            cf = get_cf_with_clash(lig_pose, point, load_range_list, preload_grid_distance, cf_size_list,
                                       load_cf_list, ligand_atoms_types, default_cf, cell_width, min_xyz,
                                       clash_list, clash_list_size, num_atoms)
            cfs_list[counter][0] = cf
            cfs_list[counter][1] = pose_index
            cfs_list[counter][2] = point_index
            counter += 1
    return cfs_list


def main(path_to_target, ligand_type, ligand_slice, target, path_to_ligands, skip_info, skip_remark, create_folder=True, custom_deps_path=None):
    root_software_path = Path(__file__).resolve().parents[1]
    os.chdir(root_software_path)
    deps_folder = os.path.join(root_software_path, 'deps')
    if custom_deps_path:
        deps_folder = custom_deps_path
    config_file = os.path.join(deps_folder, "config.txt")

    time_start = timeit.default_timer()
    verbose = False
    time = False
    save_time = False
    output_lines = []
    params_dict = get_params_dict(config_file)
    dot_division = params_dict['LIGAND_TEST_DOT_SEPARATION']
    conf_num = params_dict['CONFORMERS_PER_MOLECULE']
    poses_per_molecule = params_dict["POSES_PER_MOLECULE"]
    use_clash = params_dict["USE_CLASH"]
    default_cf = params_dict['DEFAULT_CF']
    temp_path = os.path.join(root_software_path, 'temp')
    temp_ligand_poses_path = os.path.join(temp_path, 'ligand_poses', target)
    temp_output_path = os.path.join(temp_path, 'results', target)
    if create_folder:
        if os.path.exists(temp_output_path):
            shutil.rmtree(temp_output_path)
        os.makedirs(temp_output_path, exist_ok=True)
    else:
        if not os.path.exists(temp_output_path):
            exit('Folder to output result does not exist. You may want to avoid using the "-o" flag.')
    if ligand_slice:
        start = ligand_slice[0]
        end = ligand_slice[1]
        output_file_suffix = f"{start}_{end}.txt"
    else:
        start = 0
        end = None
        output_file_suffix = ".txt"
    output_file_path = os.path.join(temp_output_path, f'{ligand_type}_{output_file_suffix}')
    if path_to_ligands is not None:
        output_file_path = os.path.join(os.path.dirname(output_file_path), f'{'_'.join(os.path.basename(path_to_ligands).split('_')[1:])}_{output_file_suffix}')

    target_preprocessing_data_path = os.path.join(path_to_target, 'preprocessed_target')
    binding_site_grid = np.load(os.path.join(target_preprocessing_data_path, f"ligand_test_dots_{dot_division}.npy"))
    precalculated_cf_list = np.load(os.path.join(target_preprocessing_data_path, f"cf_list.npy"))

    if use_clash:
        load_range_list = np.load(os.path.join(target_preprocessing_data_path, "bd_site_cuboid_coord_range_array.npy"))
        clash_list = np.load(os.path.join(target_preprocessing_data_path, f"clash_list_{params_dict['CLASH_DOT_DISTANCE']}.npy"))
        clash_list_size = clash_list.shape
    else:
        load_range_list = None
        clash_list = None
        clash_list_size = None

    if params_dict["WRITE_LIGAND_TEST_DOTS"]:
        write_pdb(binding_site_grid, "ligand_test_dots", temp_ligand_poses_path, None, None)
    n_cf_evals = len(binding_site_grid) * params_dict["ROTATIONS_PER_AXIS"]**3
    ligands_atom_names, ligands_atom_types, ligands_atom_xyz, ligand_name_list, atom_num_per_ligand, ligand_count \
        = load_ligands(path_to_target, ligand_type, start, end, conf_num, params_dict["OUTPUT_POSE"], path_to_ligands=path_to_ligands)
    cfs_list_by_ligand = np.zeros(ligand_count, dtype=np.float32)
    if save_time:
        time_list = np.zeros(ligand_count, dtype=np.float32)

    cf_size_list = np.array([np.size(precalculated_cf_list, axis=0), np.size(precalculated_cf_list, axis=1), np.size(precalculated_cf_list, axis=2)])
    output_lines.append(f"REMARK target folder: {path_to_target}")
    output_lines.append(f"REMARK software: {os.path.basename(__file__)}")
    output_lines.append(f"REMARK ligand type: {ligand_type}")
    output_lines.append(f"REMARK number of conformers: {conf_num}")
    output_lines.append(f"REMARK rotations per axis: {params_dict['ROTATIONS_PER_AXIS']}")
    output_lines.append(f"REMARK dot separation: {params_dict['LIGAND_TEST_DOT_SEPARATION']} A")
    output_lines.append(f"REMARK water vdW radius: {params_dict['WATER_RADIUS']} A")
    output_lines.append(f"REMARK preloaded grid distance: {params_dict['CLASH_DOT_DISTANCE']} A")
    output_lines.append(f"REMARK Total binding site grid dots: {len(binding_site_grid)}")
    output_lines.append(f"REMARK Total CF evaluations per ligand: {n_cf_evals}")
    output_lines.append(f"REMARK use clash: {params_dict['USE_CLASH']}")
    if time:
        output_lines.append(f"REMARK It took {timeit.default_timer() - time_start:.3f} seconds to get setup")
    if verbose:
        print("\n".join(output_lines))
    min_xyz = np.load(os.path.join(target_preprocessing_data_path, 'index_cube_min_xyz.npy'))
    cell_width = np.load(os.path.join(target_preprocessing_data_path, 'index_cube_cell_width.npy'))

    for i, ligand in enumerate(atom_num_per_ligand):
        time_ligand_start = timeit.default_timer()
        ligand_atom_count = atom_num_per_ligand[i]
        ligand_atom_xyz = ligands_atom_xyz[i][0:ligand_atom_count]
        ligand_atom_types = ligands_atom_types[i][0:ligand_atom_count]
        ligand_rotations = rotate_ligand(ligand_atom_xyz, params_dict['ROTATIONS_PER_AXIS'])
        n_cf_evals = len(binding_site_grid) * len(ligand_rotations)
        num_atoms = len(ligand_rotations[0])
        if not use_clash:
            cfs_list = get_cf_main(binding_site_grid, ligand_rotations, cf_size_list, n_cf_evals, precalculated_cf_list,
                                   ligand_atom_types,default_cf, cell_width, min_xyz)
        else:
            cfs_list = get_cf_main_clash(binding_site_grid, ligand_rotations,cf_size_list, n_cf_evals,
                                         precalculated_cf_list, ligand_atom_types, default_cf, cell_width, min_xyz,
                                         load_range_list,params_dict['CLASH_DOT_DISTANCE'],clash_list, clash_list_size,
                                         num_atoms)

        sorted_indices = np.argsort(cfs_list[:, 0])[:poses_per_molecule]
        cfs_list_by_ligand[i] = cfs_list[sorted_indices[0]][0]
        if params_dict["OUTPUT_POSE"]:
            # TODO: put all conformer poses in 1 folder
            ligand_atoms_names = ligands_atom_names[i][0:ligand_atom_count]
            if ligand_name_list[i].startswith("*****"):
                ligand_name_list[i] = "lig_" + ligand_name_list[i].split("_")[-1]
            molec_output_folder = os.path.join(temp_ligand_poses_path, ligand_name_list[i])
            if not os.path.isdir(molec_output_folder):
                os.makedirs(molec_output_folder)
            for pdb_num in range(0, poses_per_molecule, 1):
                translated_coords = np.zeros((len(ligand_rotations[int(cfs_list[sorted_indices[pdb_num]][1])]), 3), dtype=np.float32)
                for atom in range(len(ligand_rotations[int(cfs_list[sorted_indices[pdb_num]][1])])):
                    translated_coords[atom] = np.add(
                        ligand_rotations[int(cfs_list[sorted_indices[pdb_num]][1])][atom],
                        binding_site_grid[int(cfs_list[sorted_indices[pdb_num]][2])])
                write_pdb(translated_coords, f"{ligand_name_list[i]}_pose_{pdb_num+1}", molec_output_folder,
                           ligand_atoms_names, [f"REMARK CF {cfs_list[sorted_indices[pdb_num]][0]:.2f}\n",
                                                f"REMARK types: {np.array2string(ligand_atom_types, separator=' ', max_line_width=2000).strip('[]')}\n"])
        if save_time:
            time_list[i] = timeit.default_timer() - time_ligand_start

    if time:
        output_lines.append(f"REMARK total screen time: {timeit.default_timer() - time_start:.3f} seconds")
        if verbose:
            print(output_lines[-1])

    if not skip_info:
        info_file_path = os.path.join(os.path.dirname(output_file_path), 'info.txt')
        with open(info_file_path, "w") as f:
            f.writelines("\n".join(output_lines))

    output_header = "HEADER,Name,CF"
    if ligand_type != 'ligand':
        output_header += ',Type'
    if conf_num > 1:
        output_header += ",Conformer_number"
    if save_time:
        output_header += ",Time"
    if skip_remark:
        output_lines = [output_header[7:]]
    else:
        output_lines.append(output_header)

    if verbose:
        print(output_header)
    for z, ligand in enumerate(atom_num_per_ligand):
        output = f"RESULT,{ligand_name_list[z].rsplit('_', 1)[0]},{cfs_list_by_ligand[z]:.0f}"
        if skip_info or skip_remark:
            output = output[7:]
        if ligand_type != 'ligand':
            output += f',{ligand_type}'
        if conf_num > 1:
            try:
                output += "," + ligand_name_list[z].rsplit('_', 1)[1]
            except:
                output += f",{str(0)}"
        if save_time:
            output += f",{time_list[z]:.3f}"
        output_lines.append(output)
        if verbose:
            print(output)
    with open(output_file_path, "w") as f:
        f.writelines("\n".join(output_lines))
        f.write("\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--path_to_target', required=True, type=str, help='Path to target folder')
    parser.add_argument("-s", '--ligand_slice', type=str, help='Subset of ligands to screen. 2 integers separated by a comma (last one not considered): 0,100')
    parser.add_argument('-l', '--ligand_path', default=None, type=str, help='Custom ligand path')
    parser.add_argument('-lt', '--ligand_type', type=str, default='ligand',help='Specify if ligand is of a special type')
    parser.add_argument('-si', '--skip_info', action='store_true', help='Skips writing a target info file')
    parser.add_argument('-sr', '--skip_remark', action='store_true', help='Skips REMARK at start of result file')
    parser.add_argument('-o', '--create_folder', action='store_false', help='Prevents NRGRank from making folders to store output')

    args = parser.parse_args()
    path_to_target = args.path_to_target
    category = args.ligand_type
    ligand_slice = args.ligand_slice
    if ligand_slice:
        ligand_slice = [int(x) for x in ligand_slice.split(',')]
    target = os.path.basename(path_to_target)
    path_to_ligands = args.ligand_path
    skip_info = args.skip_info
    skip_remark = args.skip_remark
    create_folder = args.create_folder

    main(path_to_target, category, ligand_slice, target, path_to_ligands, skip_info, skip_remark, create_folder)


if __name__ == "__main__":
    get_args()
