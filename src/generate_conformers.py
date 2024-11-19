import concurrent.futures
import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdForceFieldHelpers, rdDistGeom
from general_functions import get_params_dict, load_rad_dict
from itertools import repeat
from datetime import datetime
from process_ligands import preprocess_ligands_one_target
import subprocess
from pathlib import Path


def generate_conformers(smiles_line, no_conformers, name_position):
    etkdg = rdDistGeom.ETKDGv3()
    # === optional settings ===
    # etkdg.maxAttempts = 10
    # etkdg.pruneRmsThresh = 0.5
    # etkdg.numThreads = 10
    # https://greglandrum.github.io/rdkit-blog/posts/2023-03-02-clustering-conformers.html
    etkdg.randomSeed = 0xa700f
    etkdg.verbose = False
    etkdg.useRandomCoords = True  # Start with random coordinates
    split_smiles_line = smiles_line.split()
    smiles = split_smiles_line[0]
    name = split_smiles_line[name_position]
    molecule = Chem.MolFromSmiles(smiles)
    print('smiles: ', smiles)
    print('name: ', name)

    try:
        frags = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
    except:
        print('Error getting fragment for: ', smiles_line)
        frags = molecule
        if frags is None:
            return None
    molecule = max(frags, key=lambda frag: frag.GetNumAtoms())

    # print('molecule: ', molecule)
    # mol_weight = rdMolDescriptors.CalcExactMolWt(molecule)
    # print(f"Molecular weight of the molecule in Daltons: {mol_weight:.2f} Da")
    # num_heavy_atoms = molecule.GetNumHeavyAtoms()
    # print('num_heavy_atoms: ', num_heavy_atoms)
    # if mol_weight > 1800 or num_heavy_atoms <= 3:
    #     print("Molecular weight is over 1800 Da or the molecule has fewer than 3 heavy atoms: Ignoring")
    #     return None
    # else:
    mol = Chem.AddHs(molecule, addCoords=True)
    if no_conformers == 1:
        try:
            AllChem.EmbedMolecule(mol, params=etkdg)
        except Exception as e:
            print('=====================================')
            print(f'Could not generate conformer for: {e}\n Molecule: {smiles_line}\n')
            print('=====================================')
            return None
    else:
        AllChem.EmbedMultipleConfs(mol, no_conformers, params=etkdg)
    mol.SetProp("_Name", name)
    return mol


def read_args():
    smiles_path = sys.argv[1]
    custom_output_folder_path = sys.argv[2]
    name_column_id = sys.argv[3] # Number of column. Example: if name is in last column arg should be -1. Starts at 0
    optimize = False
    convert = False
    preprocess = False
    main(smiles_path, optimize, custom_output_folder_path, preprocess, convert, name_column_id)


def main(smiles_path, optimize, custom_output_folder, preprocess, convert, name_column_id):
    print("Started generating conformers @ ", datetime.now())
    print("Path to smiles: ", smiles_path)
    root_software_path = Path(__file__).resolve().parents[1]
    os.chdir(root_software_path)
    deps_folder_path = os.path.join(root_software_path, 'deps')
    params_dict = get_params_dict(os.path.join(deps_folder_path, "config.txt"))
    conf_num = params_dict["CONFORMERS_PER_MOLECULE"]
    if conf_num <= 0:
        exit("Number of conformers is 0")

    output_folder = os.path.join(os.path.dirname(smiles_path), f"{os.path.basename(smiles_path).split('.')[0]}_conformers")
    if custom_output_folder != "False":
        output_folder = custom_output_folder

    end = ""
    if optimize == "yes":
        end = "_optimized"
    sdf_output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(smiles_path))[0]}_{conf_num}_conf{end}.sdf")
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    mol2_output_file = os.path.splitext(sdf_output_file)[0] + '.mol2'

    writer = AllChem.SDWriter(sdf_output_file)
    with open(smiles_path) as f:
        lines = f.readlines()
        if lines[0].startswith('smiles'):
            lines = lines[1:]

    # for line_counter, line in enumerate(lines):
    #     print(f"{line_counter+1}/{len(lines)}")
    #     mol = generate_conformers(line, conf_num, name_position)
    #     if mol is not None:
    #         for cid in range(mol.GetNumConformers()):
    #             if optimize == "yes":
    #                 Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, cid)
    #             mol = Chem.RemoveHs(mol)
    #             writer.write(mol, cid)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for mol in executor.map(generate_conformers, lines, repeat(conf_num), repeat(name_column_id)):
            if mol is not None:
                for cid in range(mol.GetNumConformers()):
                    if optimize == "yes":
                        Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, cid)
                    mol = Chem.RemoveHs(mol)
                    writer.write(mol, cid)
    AllChem.SDWriter.close(writer)
    print("Finished generating conformers @ ", datetime.now())

    if convert:
        open_babel_command = f"obabel \"{sdf_output_file}\" -O \"{mol2_output_file}\" ---errorlevel 1"
        print("converting to mol2")
        print(f'obabel command: {open_babel_command}')
        subprocess.run(open_babel_command, shell=True, check=True)
        os.remove(sdf_output_file)

    if preprocess:
        rad_dict = load_rad_dict(os.path.join(deps_folder_path, "atom_type_radius.json"))
        preprocess_ligands_one_target(rad_dict, conf_num, output_folder,'single_file', mol2_output_file,None)


if __name__ == '__main__':
    read_args()
