import os
import numpy as np
from numba import njit
from general_functions import get_params_dict, load_rad_dict, get_radius_number, write_pdb
import shutil
import timeit
import concurrent.futures
import argparse
from pathlib import Path
import traceback


def load_atoms_mol2(filename, rad_dict):
    coord_start = 0
    atoms_xyz = []
    atoms_numbers = []
    atoms_types = []
    atoms_radius = []
    with open(filename) as f:
        for line in f:
            if line.strip():
                line = line.split()
                if line[0] == '@<TRIPOS>ATOM':
                    coord_start = 1
                if line[0][0] == '@' and coord_start == 1 and line[0] != '@<TRIPOS>ATOM':
                    break
                if coord_start == 1 and line[0] != '@<TRIPOS>ATOM' and line[5].split(".")[0] != 'H':
                    atoms_xyz.append([float(line[2]), float(line[3]), float(line[4])])
                    atoms_numbers.append(int(line[0]))
                    atoms_type, atom_radius = get_radius_number(line[5], rad_dict)
                    atoms_types.append(atoms_type)
                    atoms_radius.append(atom_radius)
    atoms_xyz = np.array(atoms_xyz, dtype=np.float32)
    atoms_radius = np.array(atoms_radius, dtype=np.float32)
    atoms_numbers = np.array(atoms_numbers, dtype=np.int32)
    atoms_types = np.array(atoms_types, dtype=np.int32)
    atoms_types_sorted = atoms_types[atoms_numbers.argsort()]

    return atoms_xyz, atoms_types_sorted, atoms_radius


def find_cleft_file_simple(target_folder):
    get_cleft_folder = os.path.join(target_folder, "get_cleft")
    if not os.path.exists(get_cleft_folder):
        exit(f'Folder does not exist: {get_cleft_folder}')
    binding_site_files = sorted([file for file in os.listdir(get_cleft_folder) if file.find('_sph_') != -1 or file.find('bd_site') != -1])
    if len(binding_site_files) > 0:
        return os.path.join(get_cleft_folder, binding_site_files[0])
    else:
        exit('Could not find binding site file with _sph_ in the name in: ' + get_cleft_folder)


def load_binding_site_pdb(binding_site):
    atom_coord_type_list = []
    with open(binding_site) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                line = line.strip()
                temp_array = [float(line[30:38]), float(line[38:46]), float(line[46:54]), float(line[60:66])]  # sphere radius
                atom_coord_type_list.append(temp_array)
    return atom_coord_type_list


def make_binding_site_cuboid(dot_division, a, padding, preprocessed_file_path):
    # find min and max coords for box as well as remove or add sphere radius
    x = [np.min(a[:, 0]) - a[np.argmin(a[:, 0]), 3] - padding, np.max(a[:, 0]) + a[np.argmax(a[:, 0]), 3] + padding]
    y = [np.min(a[:, 1]) - a[np.argmin(a[:, 1]), 3] - padding, np.max(a[:, 1]) + a[np.argmax(a[:, 1]), 3] + padding]
    z = [np.min(a[:, 2]) - a[np.argmin(a[:, 2]), 3] - padding, np.max(a[:, 2]) + a[np.argmax(a[:, 2]), 3] + padding]
    bd_site_box = []
    for x_coord in x:
        for y_coord in y:
            for z_coord in z:
                bd_site_box.append([x_coord, y_coord, z_coord])
    # write_pdb(bd_site_box, f"{os.path.split(target_path)[-1]}_bd_site_box", f'./temp/ligand_poses', None, None)
    x_range = np.arange(x[0], x[1], dot_division)
    y_range = np.arange(y[0], y[1], dot_division)
    z_range = np.arange(z[0], z[1], dot_division)
    np.save(os.path.join(preprocessed_file_path, "bd_site_cuboid_coord_range_array"), np.stack((x, y, z)))
    return x_range, y_range, z_range


def build_index_cubes(params, target_atoms_xyz, atoms_radius, preprocessed_file_path, cw_factor=1):
    water_radius = params['WATER_RADIUS']
    grid_placeholder = -1
    max_rad = np.amax(atoms_radius, axis=0)
    cell_width = 2 * (max_rad + water_radius)
    max_xyz = np.zeros(3)
    max_xyz[0] = np.max(target_atoms_xyz[:, 0]) + cell_width*cw_factor
    max_xyz[1] = np.max(target_atoms_xyz[:, 1]) + cell_width*cw_factor
    max_xyz[2] = np.max(target_atoms_xyz[:, 2]) + cell_width*cw_factor
    min_xyz = np.zeros(3)
    min_xyz[0] = np.min(target_atoms_xyz[:, 0]) - cell_width*cw_factor
    min_xyz[1] = np.min(target_atoms_xyz[:, 1]) - cell_width*cw_factor
    min_xyz[2] = np.min(target_atoms_xyz[:, 2]) - cell_width*cw_factor
    lengths = ((max_xyz - min_xyz) / cell_width).astype(np.int32) + 1
    temp_grid = []
    for i in range(lengths[0]):
        temp_grid.append([])
        for j in range(lengths[1]):
            temp_grid[i].append([])
            for k in range(lengths[2]):
                temp_grid[i][j].append([])
    for i, row in enumerate(target_atoms_xyz):
        grid_indices = ((row[:3] - min_xyz) / cell_width).astype(np.int32)
        temp_grid[grid_indices[0]][grid_indices[1]][grid_indices[2]].append(i)
    max_cell_len = 0
    for row in temp_grid:
        for col in row:
            for cell in col:
                n = len(cell)
                if n > max_cell_len:
                    max_cell_len = n
    grid = np.full((lengths[0], lengths[1], lengths[2], max_cell_len), grid_placeholder, dtype=np.int32)
    for i in range(lengths[0]):
        for j in range(lengths[1]):
            for k in range(lengths[2]):
                for x, v in enumerate(temp_grid[i][j][k]):
                    grid[i][j][k][x] = v
    np.save(os.path.join(preprocessed_file_path, f"index_cube_min_xyz"), min_xyz)
    np.save(os.path.join(preprocessed_file_path, f"index_cube_cell_width"), cell_width)
    return grid, min_xyz, cell_width, max_xyz


def prepare_preprocess_output(path_to_target, params_dict, original_config_path, overwrite):
    numpy_output_path = os.path.join(path_to_target, 'preprocessed_target')
    os.makedirs(numpy_output_path, exist_ok=True)

    config_file_name = f"config_{params_dict['CLASH_DOT_DISTANCE']}"
    config_output = os.path.join(numpy_output_path, config_file_name + ".txt")
    if os.path.exists(config_output):
        if overwrite:
            os.remove(config_output)
            shutil.copyfile(original_config_path, config_output)
        else:
            print(f"(WARNING) Config file already exists and overwrite flag is missing: {config_output}")
            return numpy_output_path
    shutil.copyfile(original_config_path, config_output)
    return numpy_output_path


def load_ligand_test_dots(params, binding_site_spheres):
    """ This function uses the binding site spheres to make dots on which the ligand will be centered for testing poses"""
    dot_division = params['LIGAND_TEST_DOT_SEPARATION']
    grid_coords = []
    a = np.array(binding_site_spheres)

    x = [round(np.min(a[:, 0]) - a[np.argmin(a[:, 0]), 3], 3), round(np.max(a[:, 0]) + a[np.argmax(a[:, 0]), 3], 3)]
    y = [round(np.min(a[:, 1]) - a[np.argmin(a[:, 1]), 3], 3), round(np.max(a[:, 1]) + a[np.argmax(a[:, 1]), 3], 3)]
    z = [round(np.min(a[:, 2]) - a[np.argmin(a[:, 2]), 3], 3), round(np.max(a[:, 2]) + a[np.argmax(a[:, 2]), 3], 3)]

    for dot_x in np.arange(x[0], x[1], dot_division):
        for dot_y in np.arange(y[0], y[1], dot_division):
            for dot_z in np.arange(z[0], z[1], dot_division):
                coords = np.array([round(dot_x, 3), round(dot_y, 3), round(dot_z, 3)])
                for row in a:
                    distance = np.linalg.norm(coords - row[:3])
                    if distance < row[3]:
                        grid_coords.append(coords)
                        break
    return grid_coords


def clean_binding_site_grid(target_grid, binding_site_grid, min_xyz, cell_width, target_atoms_xyz, ligand_test_dot_file_path):
    index = []
    for a, point in enumerate(binding_site_grid):
        grid_index = ((point - min_xyz) / cell_width).astype(np.int32)
        for i_offset in [-1, 0, 1]:
            for j_offset in [-1, 0, 1]:
                for k_offset in [-1, 0, 1]:
                    i = i_offset + grid_index[0]
                    j = j_offset + grid_index[1]
                    k = k_offset + grid_index[2]
                    if i < len(target_grid) and j < len(target_grid[0]) and k < len(target_grid[0][0]):
                        for neighbour in target_grid[i][j][k]:
                            if neighbour == -1:
                                break
                            else:
                                dist = np.sqrt((target_atoms_xyz[neighbour][0] - point[0]) ** 2 +
                                               (target_atoms_xyz[neighbour][1] - point[1]) ** 2 +
                                               (target_atoms_xyz[neighbour][2] - point[2]) ** 2)
                                if dist <= 2.0:
                                    index.append(a)

                                    break
    cleaned_binding_site_grid = np.delete(binding_site_grid, index, 0)
    # write_pdb(cleaned_binding_site_grid, "cleaned grid", f'./temp/ligand_poses/', None, None)
    np.save(ligand_test_dot_file_path, cleaned_binding_site_grid)


@njit
def get_cf_list(target_grid, atom_type_range, target_atom_types, energy_matrix, number_types):
    target_grid_x = len(target_grid)
    target_grid_y = len(target_grid[0])
    target_grid_z = len(target_grid[0][0])
    result_array = np.zeros((target_grid_x, target_grid_y, target_grid_z, number_types))
    for x in np.arange(0, target_grid_x):
        for y in np.arange(0, target_grid_y):
            for z in np.arange(0, target_grid_z):
                grid_index = [x, y, z]
                for counter, atom_type in enumerate(atom_type_range):
                    cf = 0.0
                    for i_offset in [-1, 0, 1]:
                        for j_offset in [-1, 0, 1]:
                            for k_offset in [-1, 0, 1]:
                                i = i_offset + grid_index[0]
                                j = j_offset + grid_index[1]
                                k = k_offset + grid_index[2]
                                if 0 < i < len(target_grid) and 0 < j < len(target_grid[0]) and 0 < k < len(target_grid[0][0]):
                                    if target_grid[i][j][k][0] != -1:
                                        for neighbour in target_grid[i][j][k]:
                                            if neighbour == -1:
                                                break
                                            else:
                                                # Normal program:
                                                type_1 = atom_type
                                                type_2 = target_atom_types[neighbour]
                                                # Randomly chosen atom type
                                                # type_1 = np.random.randint(1, 41)
                                                # type_2 = np.random.randint(1, 41)
                                                energy_value = energy_matrix[type_1][type_2]
                                                cf += energy_value
                    result_array[x, y, z, counter] = cf
    return result_array


@njit
def get_clash_per_dot(x_range, y_range, z_range, target_grid, min_xyz, cell_width, target_atoms_xyz, max_size_array):
    clash_list = np.zeros((max_size_array[0], max_size_array[1], max_size_array[2]), dtype=np.bool_)
    for a, x_value in enumerate(x_range):
        for b, y_value in enumerate(y_range):
            for c, z_value in enumerate(z_range):
                ligand_atom = np.array([x_value, y_value, z_value], dtype=np.float32)
                clash_list[a][b][c] = get_clash_for_dot(ligand_atom, target_grid, min_xyz, cell_width, target_atoms_xyz)
    return clash_list

@njit
def get_clash_for_dot(ligand_atom_coord, target_grid, min_xyz, cell_width, target_atoms_xyz):
    # TODO: test if clashes per radius associated to each type is better
    clash = False
    grid_index = ((ligand_atom_coord - min_xyz) / cell_width).astype(np.int32)
    for i_offset in [-1, 0, 1]:
        for j_offset in [-1, 0, 1]:
            for k_offset in [-1, 0, 1]:
                i = i_offset + grid_index[0]
                j = j_offset + grid_index[1]
                k = k_offset + grid_index[2]
                if 0 <= i < len(target_grid) and 0 <= j < len(target_grid[0]) and 0 <= k < len(target_grid[0][0]):
                    if target_grid[i][j][k][0] != -1:
                        for neighbour in target_grid[i][j][k]:
                            if neighbour == -1:
                                break
                            else:
                                dist = np.linalg.norm(target_atoms_xyz[neighbour] - ligand_atom_coord)
                                if dist <= 2.0:
                                    clash = True
                                    return clash
    return clash


def preprocess_one_target(target, main_path, params_dict, energy_matrix, time_start, overwrite, verbose=True):
    root_software_path = Path(__file__).resolve().parents[1]
    deps_folder = os.path.join(root_software_path, 'deps')
    config_file_path = os.path.join(deps_folder, 'config.txt')
    target_path = str(os.path.join(main_path, target))
    receptor_path = os.path.join(target_path, "receptor.mol2")
    use_clash = params_dict["USE_CLASH"]
    if not os.path.isfile(receptor_path):
        exit(f"Could not find receptor.mol2 file at path: {receptor_path}")

    # ####################### DEFINE CUBOID AROUND BINDING SITE #######################

    rad_dict = load_rad_dict(os.path.join(deps_folder, "atom_type_radius.json"))
    clash_grid_distance = params_dict['CLASH_DOT_DISTANCE']
    bd_site_cuboid_padding = params_dict["BD_SITE_CUBOID_PADDING"]
    dot_division = params_dict['LIGAND_TEST_DOT_SEPARATION']
    number_of_atom_types = 39

    binding_site = find_cleft_file_simple(target_path)
    target_atoms_xyz, target_atoms_types, atoms_radius = load_atoms_mol2(receptor_path, rad_dict)
    preprocessed_file_path = prepare_preprocess_output(target_path, params_dict, config_file_path, overwrite)
    index_cubes, min_xyz, cell_width, max_xyz = build_index_cubes(params_dict, target_atoms_xyz, atoms_radius, preprocessed_file_path)
    binding_site_spheres = load_binding_site_pdb(binding_site)
    binding_site_x_range, binding_site_y_range, binding_site_z_range = make_binding_site_cuboid(clash_grid_distance,
                                                                                                np.array(binding_site_spheres),
                                                                                                bd_site_cuboid_padding,
                                                                                                preprocessed_file_path)

    # ####################### PRECALCULATE CF #######################
    cf_array_path = os.path.join(preprocessed_file_path, "cf_list.npy")
    clash_file_path = os.path.join(preprocessed_file_path, f"clash_list_{clash_grid_distance}.npy")
    if not os.path.isfile(cf_array_path) or overwrite:
        max_size_array = np.array([len(binding_site_x_range), len(binding_site_y_range), len(binding_site_z_range), number_of_atom_types], dtype=np.int32)
        atom_type_range = np.arange(1, number_of_atom_types+1)
        if use_clash:
            if verbose:
                print('Getting clashes')
            clash_list = get_clash_per_dot(binding_site_x_range, binding_site_y_range, binding_site_z_range,
                                           index_cubes, min_xyz, cell_width, target_atoms_xyz, max_size_array)
            np.save(clash_file_path, clash_list)
        if verbose:
            print('Precalculating CF')
        cfs_list = get_cf_list(index_cubes, atom_type_range, target_atoms_types, energy_matrix, number_of_atom_types)
        np.save(cf_array_path, cfs_list)
    else:
        print(f"CF already precalculated. Use -o flag if you wish to overwrite.")

    # ####################### GENERATE AND CLEAN LIGAND TEST DOTS #######################

    ligand_test_dot_file_path = os.path.join(preprocessed_file_path, f"ligand_test_dots_{dot_division}.npy")
    if not os.path.isfile(ligand_test_dot_file_path) or overwrite:
        original_grid = load_ligand_test_dots(params_dict, binding_site_spheres)
        clean_binding_site_grid(index_cubes, original_grid, min_xyz, cell_width, target_atoms_xyz,
                                ligand_test_dot_file_path)
    else:
        print(f"The file for binding site dots at {dot_division} A distance already exists")
    if verbose:
        total_run_time = timeit.default_timer() - time_start
        if total_run_time > 60.0:
            total_run_time /= 60
            print(f"{target}: {total_run_time} minutes to run")
        else:
            print(f"{target}: {total_run_time} seconds to run")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--path_to_targets', required=True, type=str,
                        help='Path to folder containing target folders')
    parser.add_argument('-t', '--specific_target', type=str,
                        help='Specify target(s) to analyse. If multiple: separate with comma no space')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='specify if you want to overwrite pre existing files')

    args = parser.parse_args()
    path_to_targets = os.path.abspath(args.path_to_targets)
    if args.specific_target is None:
        target_list = next(os.walk(path_to_targets))[1]
    else:
        target_list = args.specific_target.split(',')
    target_list = sorted(target_list)
    main(path_to_targets, target_list, overwrite=args.overwrite)


def main(path_to_targets, target_list, overwrite=False):
    root_software_path = Path(__file__).resolve().parents[1]
    os.chdir(root_software_path)
    time_start = timeit.default_timer()
    config_file_path = os.path.join(root_software_path, 'deps', 'config.txt')
    params_dict = get_params_dict(config_file_path)
    use_clash = params_dict["USE_CLASH"]
    if use_clash:
        print('Running in clean mode (ignoring poses with clashes)')
    matrix_name = params_dict['MATRIX_NAME']
    matrix_path = os.path.join(root_software_path, 'deps', 'matrix', f'{matrix_name}.npy')
    energy_matrix = np.load(matrix_path)
    verbose = True
    if len(target_list) > 1:
        verbose = False
        print('Preprocessing targets: ', target_list)
    else:
        print('Preprocessing target: ', target_list[0])
    # preprocess_one_target(target_list[0], path_to_targets, params_dict, energy_matrix, time_start, overwrite, verbose)

    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        futures = []
        for target in target_list:
            futures.append(
                executor.submit(preprocess_one_target, target, path_to_targets, params_dict, energy_matrix, time_start,
                                overwrite, verbose))
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing task: {e}")
                print("Traceback:")
                traceback.print_exc()

if __name__ == "__main__":
    get_args()
