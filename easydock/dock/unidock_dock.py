#!/usr/bin/env python3

import argparse
import logging
import os
import re
import tempfile
import timeit
import subprocess
import yaml
from pathlib import Path

from easydock.auxiliary import expand_path, resolve_path
from easydock.dock.preparation_for_docking import ligand_preparation, pdbqt2molblock


class RawTextArgumentDefaultsHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def __get_pdbqt_and_score(ligand_out_fname):
    with open(ligand_out_fname) as f:
        pdbqt_out = f.read()
    match = re.search(r'REMARK VINA RESULT:\s+(-?[\d.]+)', pdbqt_out)
    if match:
        score = round(float(match.group(1)), 3)
    else:
        score = None

    return score, pdbqt_out


def mol_dock(mol_or_mols, config, ring_sample=False):
    """

    :param mol_or_mols: RDKit Mol or list of Mols of ligands with titles
    :param config: yml-file with docking settings
    :param ring_sample: whether to sample saturated rings and dock multiple starting conformers
    :return:
    """
    config = __parse_config(config)
    is_list = isinstance(mol_or_mols, list)
    mols = mol_or_mols if is_list else [mol_or_mols]

    start_time = timeit.default_timer()

    # Prepare all ligands
    prepared_ligands = []
    for mol in mols:
        mol_id = mol.GetProp('_Name')
        ligand_pdbqt_list = ligand_preparation(mol, boron_replacement=True, ring_sample=ring_sample)
        if ligand_pdbqt_list:
            prepared_ligands.append((mol, mol_id, ligand_pdbqt_list))

    if not prepared_ligands:
        return [(mol.GetProp('_Name'), None) for mol in mols] if is_list else (mols[0].GetProp('_Name'), None)

    out_dir = tempfile.mkdtemp(suffix='_unidock_out')
    index_fd, index_fname = tempfile.mkstemp(suffix='_index.txt', text=True)
    os.close(index_fd)

    ligand_files = []
    ligand_mapping = {}  # To keep track of which file belongs to which mol_id and conformer
    results = []

    try:
        with open(index_fname, 'wt') as index_f:
            ligand_idx = 0
            for mol, mol_id, ligand_pdbqt_list in prepared_ligands:
                mol_ligand_files = []
                for i, ligand_pdbqt in enumerate(ligand_pdbqt_list):
                    # Do not use mol_id in the filename since SMILES strings contain '/' and '\'
                    ligand_fd, ligand_fname = tempfile.mkstemp(suffix=f'_ligand_{ligand_idx}.pdbqt', text=True)
                    os.close(ligand_fd)
                    with open(ligand_fname, 'wt') as f:
                        f.write(ligand_pdbqt)
                    ligand_files.append(ligand_fname)
                    mol_ligand_files.append((ligand_fname, i))
                    index_f.write(ligand_fname + '\n')
                    ligand_idx += 1
                ligand_mapping[mol_id] = {'mol': mol, 'files': mol_ligand_files}

        cmd = [
            config["script_file"],
            "--receptor", config["protein"],
            "--ligand_index", index_fname,
            "--dir", out_dir,
        ]

        # Add predefined mapping
        if "search_mode" in config:
            cmd += ["--search_mode", config["search_mode"]]
        if "exhaustiveness" in config:
            cmd += ["--exhaustiveness", config["exhaustiveness"]]
        if "n_poses" in config:
            cmd += ["--num_modes", config["n_poses"]]
        if "seed" in config:
            cmd += ["--seed", config["seed"]]
        if "ncpu" in config:
            cmd += ["--cpu", config["ncpu"]]
        if "protein_setup" in config:
            cmd += ["--config", config["protein_setup"]]

        # Add all other arguments
        reserved_keys = ['protein', 'protein_setup', 'script_file', 'search_mode', 'exhaustiveness', 'n_poses', 'seed', 'ncpu', 'num_modes', 'receptor', 'ligand_index', 'dir', 'config']
        for key, value in config.items():
            if key not in reserved_keys:
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd += [f"--{key}", str(value)]

        cmd = ' '.join(map(str, cmd))

        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

        for mol_id, data in ligand_mapping.items():
            dock_output_conformer_list = []
            mol = data['mol']
            for ligand_fname, i in data['files']:
                ligand_basename = Path(ligand_fname).stem
                output_fname = os.path.join(out_dir, f"{ligand_basename}_out.pdbqt")
                if os.path.isfile(output_fname):
                    score, pdbqt_out = __get_pdbqt_and_score(output_fname)
                    if score is not None:
                        mol_block = pdbqt2molblock(pdbqt_out.split('MODEL')[1], mol, mol_id)
                        dock_output = {'docking_score': score,
                                       'raw_block': pdbqt_out,
                                       'mol_block': mol_block}
                        dock_output_conformer_list.append(dock_output)

            if dock_output_conformer_list:
                output = min(dock_output_conformer_list, key=lambda x: x['docking_score'])
            else:
                output = None
            results.append((mol_id, output))

    except subprocess.CalledProcessError as e:
        logging.warning(f'(unidock) Error caused by docking of batch\n'
                        f'{str(e)}\n')
        results = [(mol.GetProp('_Name'), None) for mol in mols]
    except Exception as e:
        logging.warning(f'(unidock) Unexpected error caused by docking of batch\n'
                        f'{str(e)}\n')
        results = [(mol.GetProp('_Name'), None) for mol in mols]

    finally:
        if os.path.exists(index_fname):
            os.remove(index_fname)
        for fname in ligand_files:
            if os.path.exists(fname):
                os.remove(fname)
        if os.path.exists(out_dir):
            for f in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, f))
            os.rmdir(out_dir)

    dock_time = round(timeit.default_timer() - start_time, 1)

    # Calculate docking time per molecule
    valid_results = [r for r in results if r[1] is not None]
    if valid_results:
        time_per_mol = dock_time / len(valid_results)
        for mol_id, output in valid_results:
            output['dock_time'] = round(time_per_mol, 1)

    # Make sure all submitted molecules have an entry
    final_results = []
    result_dict = dict(results)
    for mol in mols:
        mol_id = mol.GetProp('_Name')
        if mol_id in result_dict:
            final_results.append((mol_id, result_dict[mol_id]))
        else:
            final_results.append((mol_id, None))

    if is_list:
        return final_results
    else:
        return final_results[0]


def __parse_config(config_fname):
    with open(config_fname) as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(config_fname))
    for arg in ['protein', 'protein_setup', 'script_file']:
        config[arg] = resolve_path(config[arg], config_dir)

    return config
