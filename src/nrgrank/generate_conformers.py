import concurrent.futures
import os
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors, rdForceFieldHelpers, rdDistGeom
    HAS_RDKIT = True
except Exception:
    rdkit = None  # type: ignore
    HAS_RDKIT = False
from itertools import repeat
from datetime import datetime
from nrgrank import process_ligands
import subprocess
import csv
import argparse


def require_rdkit(feature: str = "this feature") -> None:
    if not HAS_RDKIT:
        raise ImportError(
            f"RDKit is required for {feature}. Install with 'pip install nrgrank[rdkit]' "
            f"or ensure RDKit is available in your environment."
        )


def get_delimiter(file_path, bytes_to_read=4096):
    sniffer = csv.Sniffer()
    data = open(file_path, "r").read(bytes_to_read)
    delimiter = sniffer.sniff(data).delimiter
    return delimiter


def read_column_from_csv(file_path, column_number, delimiter, has_header=True):
    column_values = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) > column_number:
                column_values.append(row[column_number])

    return column_values


def generate_conformers(molecule_smile, molecule_name, no_conformers, mol_weight_max=None, heavy_atoms_min=None):
    etkdg = rdDistGeom.ETKDGv3()
    # === optional settings ===
    # etkdg.maxAttempts = 10
    # etkdg.pruneRmsThresh = 0.5
    # etkdg.numThreads = 10
    # https://greglandrum.github.io/rdkit-blog/posts/2023-03-02-clustering-conformers.html
    etkdg.randomSeed = 0xa700f
    etkdg.verbose = False
    etkdg.useRandomCoords = True
    molecule = Chem.MolFromSmiles(molecule_smile)
    try:
        frags = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
    except:
        print('Error getting fragment for: ', molecule_name)
        frags = molecule
        if frags is None:
            return None
    molecule = max(frags, key=lambda frag: frag.GetNumAtoms())
    if mol_weight_max:
        mol_weight = rdMolDescriptors.CalcExactMolWt(molecule)
        if mol_weight > mol_weight_max:
            return None
    if heavy_atoms_min:
        num_heavy_atoms = molecule.GetNumHeavyAtoms()
        if num_heavy_atoms <= heavy_atoms_min:
            return None

    mol = Chem.AddHs(molecule, addCoords=True)
    if no_conformers == 1:
        try:
            AllChem.EmbedMolecule(mol, params=etkdg)
        except Exception as e:
            print('=====================================')
            print(f'Error: {e}\n Molecule: {molecule_name}\n')
            print('=====================================')
            return None
    else:
        AllChem.EmbedMultipleConfs(mol, no_conformers, params=etkdg)
    mol.SetProp("_Name", molecule_name)

    return mol


def read_args():
    parser = argparse.ArgumentParser(description="Process and convert chemical data.")

    parser.add_argument(
        "-s",
        "--smiles_path",
        type=str,
        help="Path to the SMILES file."
    )
    parser.add_argument(
        "-sc",
        "--smiles_column_number",
        type=int,
        required=True,
        help="Number of the column containing smiles (Starts at 0). Example: -1 for last column.",
    )
    parser.add_argument(
        "-nc",
        "--name_column_number",
        type=int,
        required=True,
        help="Number of the column containing names (Starts at 0). Example: -1 for last column.",
    )
    parser.add_argument(
        "-o",
        "--output_folder_path",
        type=str,
        default=None,
        help="Path to the custom output folder. Use 'None' if not specified.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable optimization. Defaults to False.",
    )
    parser.add_argument(
        "-dc",
        "--no_convert",
        action="store_false",
        dest="convert",
        help="Disable conversion from SDF to MOL2. Defaults to True.",
    )
    parser.add_argument(
        "-p",
        "--no_preprocess",
        action="store_false",
        dest="preprocess",
        help="Disable preprocessing. Defaults to True.",
    )
    parser.add_argument(
        "-mw",
        "--molecular_weight_max",
        type=int,
        default=0,
        help="Maximum molecular weight. Defaults to 0 which will not filter for MW",
    )
    parser.add_argument(
        "-ha",
        "--heavy_atoms_min",
        type=int,
        default=0,
        help="Minimum number of heavy atoms. Defaults to 0 which will not filter for this setting",
    )

    args = parser.parse_args()

    main(
        smiles_path_or_list=args.smiles_path,
        smiles_column_number=args.smiles_column_number,
        name_column_number=args.name_column_number,
        output_folder_path=args.output_folder_path,
        optimize=args.optimize,
        convert=args.convert,
        preprocess=args.preprocess,
        molecular_weight_max=args.molecular_weight_max,
        heavy_atoms_min=args.heavy_atoms_min
    )


def main(smiles_path_or_dict, output_folder_path, smiles_column_number=None, name_column_number=None,
         conformers_per_molecule=1, optimize=False, convert=True, preprocess=False, molecular_weight_max=None,
         heavy_atoms_min=None):
    require_rdkit()
    if isinstance(smiles_path_or_dict, dict):
        molecule_smiles_list = list(smiles_path_or_dict['Smiles'])
        molecule_name_list = list(smiles_path_or_dict['Name'])
    else:
        delimiter = get_delimiter(smiles_path_or_dict, bytes_to_read=4096)
        molecule_smiles_list = read_column_from_csv(smiles_path_or_dict, smiles_column_number, delimiter, has_header=True)
        molecule_name_list = read_column_from_csv(smiles_path_or_dict, name_column_number, delimiter, has_header=True)

    if conformers_per_molecule <= 0:
        exit("Number of conformers must be greater than 0.")

    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    sdf_output_file = os.path.join(output_folder_path, "conformer")
    if conformers_per_molecule > 1:
        sdf_output_file += f"_{conformers_per_molecule}_conf"
    if optimize:
        sdf_output_file += "_optimized"
    sdf_output_file += '.sdf'
    mol2_output_file = os.path.splitext(sdf_output_file)[0] + '.mol2'

    writer = AllChem.SDWriter(sdf_output_file)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for mol in executor.map(generate_conformers, molecule_smiles_list, molecule_name_list,
                                repeat(conformers_per_molecule), repeat(molecular_weight_max), repeat(heavy_atoms_min)):
            if mol is not None:
                for cid in range(mol.GetNumConformers()):
                    if optimize:
                        Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, cid)
                    mol = Chem.RemoveHs(mol)
                    writer.write(mol, cid)
    AllChem.SDWriter.close(writer)
    print("Finished generating conformers @ ", datetime.now())

    if convert:
        print("converting to mol2")
        open_babel_command = f"obabel \"{sdf_output_file}\" -O \"{mol2_output_file}\" ---errorlevel 1"
        print(f'obabel command: {open_babel_command}')
        subprocess.run(open_babel_command, shell=True, check=True)
        os.remove(sdf_output_file)

    if preprocess:
        process_ligands(ligand_path=mol2_output_file)


if __name__ == '__main__':
    read_args()