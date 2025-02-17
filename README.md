# Installation (Requires at least Python version 3.9 )

Use the following command in terminal (Mac/Linux):

```
git clone https://github.com/NRGlab/NRGRank
cd NRGRank
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Preparatory steps

## Preprocessing the target

   1) Prepare target folder:
   
      A folder with the target file called receptor.mol2 (must be in mol2 format; can be done in PyMol).
      
      A folder called get_cleft in the target folder with the desired binding site file (made with GetCleft). The name must contain "_sph_".
         
      Example:
      ```
      TARGET_FOLDER
          ├── get_cleft
          │   ├── receptor_sph_1.pdb
          ├── receptor.mol2
      ```

   2) Preparing the target with process_target.py:
   
       ### Required Flags
      | Flag | Description                                    | Possible value(s) |
      |------|------------------------------------------------|-------------------|
      | `-p` | Path to parent directory of the targets folder | Absolute path     |
   
      ### Optional Flags
      | Flag | Description                                                    | Possible value(s) |
      |------|----------------------------------------------------------------|-------------------|
      | `-t` | Name of target folder if you want to process a specific target | Name of target    |
      | `-o` | Allows overwriting existing files                              | N/A               |
   
      ### Example command to preprocess the aa2ar target in /foo/bar/targets

      ```
      python src/process_target.py -p /foo/bar/targets -t aa2ar -o
      ```

## Preparing the ligands
   
All ligands for 1 screen must be in the same mol2 file. However, multiple ligand files can be prepared with one command.
These can either be in one folder of multiple subdirectories of a given folder.
      
   ### Required Flags
   
   For running on a single mol2 file:

   | Flag  | Description              | Possible value(s) |
   |-------|--------------------------|-------------------|
   | `-lp` | Path to ligand mol2 file | Absolute path     |

   For running on a multiple mol2 file:

   | Flag  | Description                                | Possible value(s) |
   |-------|--------------------------------------------|-------------------|
   | `-fp` | Path to parent directory of ligand file(s) | Absolute path     |
   
   ### Optional Flags for running on multiple mol2 file in subdirectories

   | Flag  | Description                                                     | Possible value(s) |
   |-------|-----------------------------------------------------------------|-------------------|
   | `-sd` | For when your ligand files are in multiple subdirectories of fp | N/A               |
   
   ### Example commands:
      
   To preprocess the ligand file __/foo/bar/ligand.mol2__ file

   ```
   python src/process_ligands.py -lp /foo/bar/ligand.mol2
   ```
# NRGRank

You should now have a folder structure as follows and are ready to run NRGRank:
```
TARGET_FOLDER
    ├── get_cleft
    │   ├── receptor_sph_1.pdb
    ├── preprocessed_target
    ├── preprocessed_ligands
    ├── receptor.mol2
```

### Example commands:
      
   Running NRGRank on the first 1000 ligands for the target stored at /foo/bar/target:

   ```
   python src/NRGRank.py -p /foo/bar/target -s 0,1001
   ```

### Required Flags
    
| Flag  | Description             | Possible value(s) |
|-------|-------------------------|-------------------|
| `-p`  | Path to target folder   | Absolute path     |

### Optional Flags
    
| Flag  | Description                                | Possible value(s)          |
|-------|--------------------------------------------|----------------------------|
| `-s`  | Only screen subset of ligands              | int,int (last is excluded) |
| `-l`  | Path to preprocessed ligand folder         | Absolute folder            |
| `-lt` | Specify if ligand is of a special type     | str                        |
| `-si` | Skips writing the target info file         | N/A                        |
| `-sr` | Skips writing REMARKS before result        | N/A                        |
| `-o`  | Prevents NRGRank from creating new folders | N/A                        |

---

# Example

For this example, we will be using a target and ligands from the DUD-E database. 
    
1) Download the target file here: <https://dude.docking.org//targets/aa2ar/receptor.pdb>
2) Convert the downloaded file to mol2 using a software such as **PyMOL** or an online converter such as the one available here: <https://www.cheminfo.org/Chemistry/Cheminformatics/FormatConverter/index.html>
3) Rename the target file to receptor.mol2
4) Download and extract the set of known binders here: <https://dude.docking.org//targets/aa2ar/actives_final.mol2.gz>
5) Make a folder (AA2AR) with the target file using the following structure:

    ```
    AA2AR
    ├── actives_final.mol2
    ├── receptor.mol2
    ```

6) Generate a binding site using GetCleft (1.75 minimum sphere radii / 3.75 maximum sphere radii). The easiest way is by using the NRGSuite-Qt PyMOL plugin available here: <https://nrg-qt.readthedocs.io/en/latest/>
7) Place the binding site file in a folder called **get_cleft** to obtain the following folder structure:
    ```
    AA2AR
      ├── get_cleft
      │   ├── receptor_sph_1.pdb
      ├── receptor.mol2
      ├── actives_final.mol2
    ```
8) Prepare the target using the following command (replace the path to the parent directory of the target): 
    ```
    python src/process_target.py -p /foo/bar/targets -t AA2AR -o
    ```
9) Prepare the ligand file
    ```
    python src/process_ligands.py -lp /foo/bar/AA2AR/actives_final.mol2
    ```
10) Run NRGRank:
    ```
    python src/NRGRank.py -p /foo/bar/AA2AR
    ```




