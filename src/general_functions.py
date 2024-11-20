import os
import json


def get_params_dict(file_path):
    config = {}
    bool_dict = {"True": True, "False": False}

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("-") == 0:
                line = line.strip()
                if line:
                    key, value = line.split()
                    if value in bool_dict:
                        value = bool_dict[value]
                    else:
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass
                    config[key] = value
    return config

def write_pdb(coord_list, name, path, ligand_names, extra_info):
    if not os.path.isdir(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    if not os.path.isdir(path):
        os.mkdir(path)
    textfile = open(os.path.join(path, name + ".pdb"), 'w')
    counter = 0
    if extra_info is not None:
        for line in extra_info:
            textfile.write(line)
    for line in coord_list:
        if ligand_names is not None:
            textfile.write("HETATM {:>4}  {:<4}LIG L   1{:>12} {:>7} {:>7}  1.00  0.10 \n".format(str(counter),
                                                                                                    ligand_names[counter],
                                                                                                    str(round(line[0], 3)),
                                                                                                    str(round(line[1], 3)),
                                                                                                    str(round(line[2], 3))))
        else:
            textfile.write("HETATM {:>4}  C   DOT X   1{:>12} {:>7} {:>7}  1.00  0.10 \n".format(str(counter),
                                                                                                  str(round(line[0], 3)),
                                                                                                  str(round(line[1], 3)),
                                                                                                  str(round(line[2], 3))))
        counter += 1
    textfile.close()

def load_rad_dict(filepath):
    with open(filepath, 'r') as file:
        loaded_atom_data = json.load(file)
    return loaded_atom_data


def get_radius_number(letter_type, rad_dict):
    letter_type = letter_type.upper().replace(' ', '')
    try:
        atm_info = [rad_dict[letter_type]['type_number'], rad_dict[letter_type]['radius']]
    except KeyError:
        atm_info = [39, 2.00]
    atm_type_num = atm_info[0]
    atm_rad = atm_info[1]
    return atm_type_num, atm_rad