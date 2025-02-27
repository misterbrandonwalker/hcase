# Author: Gergely Zahoranszky-Kohalmi, PhD
#
#
# Email: gergely.zahoranszky-kohalmi@nih.gov
#
# Organization: National Center for Advancing Translational Sciences (NCATS/NIH)
#
#
#
#
# References
#
#
# Ref: https://chartio.com/resources/tutorials/how-to-save-a-plot-to-a-file-using-matplotlib/
# Ref: https://engineering.hexacta.com/pandas-by-example-columns-547696ff78dd
# Ref: https://github.com/matplotlib/matplotlib/issues/3466/
# Ref: https://maxpowerwastaken.github.io/blog/pandas_view_vs_copy/
# Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
# Ref: https://pubs.acs.org/doi/10.1021/ci5001983
# Ref: https://pypi.org/project/hilbertcurve/
# Ref: https://realpython.com/python-rounding/
# Ref: https://stackoverflow.com/questions/38862293/how-to-add-incremental-numbers-to-a-new-column-using-pandas/38862389
# Ref: https://towardsdatascience.com/dockerizing-jupyter-projects-39aad547484a
# Ref: https://www.dataquest.io/blog/settingwithcopywarning/
# Ref: https://www.geeksforgeeks.org/log-functions-python/
# Ref: https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/
# Ref: https://www.w3schools.com/python/ref_math_ceil.asp
# ChatGPT 4.0 Palantir Instance
# ChatGPT 4o www.openai.com


"""
By ChatGPT 4.0 Palantir Instance
"""



import math
import pandas as pd
from typing import Optional
import numpy as np
import time

from rdkit import RDLogger

from hilbertcurve.hilbertcurve import HilbertCurve
from hcase.scaffold_keys import smiles2bmscaffold, smiles2scaffoldkey, sk_distance, onestring

from logging import getLogger

from pandarallel import pandarallel
import cupy as cp

def initialize_pandarallel(n_cores: int = 1, progress_bar: bool = True):
    """
    Initializes pandarallel with a specified number of cores.

    Args:
        n_cores (int, optional): Number of CPU cores to use. Defaults to all available cores.
        progress_bar (bool): Whether to show a progress bar. Defaults to True.
    """
    pandarallel.initialize(nb_workers=n_cores, progress_bar=progress_bar)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

logger = getLogger()

def order_scaffolds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orders scaffolds based on structure and scaffold key.

    Args:
        df: Pandas DataFrame with the following columns:
            - pattern_id: includes the pattern type and a unique ID separated by a "dot"
              example: scaffold.4
            - structure: SMILES structure
            - ptype: type of scaffold, example: scaffold
            - hash: InChIKey, example: NOWKCMXCCJGMRR-UHFFFAOYSA-N

    Returns:
        A DataFrame ordered by scaffold keys with structure, order, scaffold_id, and scaffold_key.
    """
    logger.info('[*] Ordering reference scaffolds ..')

    # Filter the dataframe to include only rows where ptype is 'scaffold'
    df = df.query("ptype == 'scaffold'")

    nr_orig_scaffolds = df.shape[0]


    # Apply sk function (structure key) using pandarallel
    df['sk'] = df['structure'].astype(str).parallel_apply(sk)


    # Filter out rows where the scaffold key is 'NA'
    df = df[df['sk'].ne('NA')]



    # Apply sk_one function (structure key variant) using pandarallel
    df['sk_one'] = df['sk'].parallel_apply(sk_one)


    # Sort by 'sk_one'
    df = df.sort_values(['sk_one'])

    # Group by 'sk_one' and take the first occurrence of 'sk', 'pattern_id', and 'structure'
    df = df.groupby('sk_one', as_index=False).nth(0)


    # Drop the 'sk_one' column and rename columns for clarity
    df = df.drop(columns=['sk_one']).rename(columns={
        'sk': 'scaffold_key',
        'pattern_id': 'scaffold_id'
    })

    # Reset the index and create a new 'order' column
    df = df.reset_index(drop=True)
    df['order'] = df.index + 1

    # Reorganize the DataFrame to include only the necessary columns
    df = df[['structure', 'order', 'scaffold_id', 'scaffold_key']].copy()

    # Print information about the number of scaffolds
    logger.info('[*] Number of scaffolds in input:')
    logger.info(nr_orig_scaffolds)

    logger.info('[*] Number of unique reference scaffolds:')
    logger.info(df.shape[0])

    logger.info('[*] Done.')

    return df


def define_reference_scaffolds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defines the reference scaffolds by extracting relevant columns.

    Args:
        df: Pandas DataFrame with the following columns:
            - pattern_id: includes the pattern type and a unique ID separated by a "dot"
              example: scaffold.4
            - structure: SMILES structure
            - ptype: type of scaffold, example: scaffold
            - hash: InChIKey, example: NOWKCMXCCJGMRR-UHFFFAOYSA-N

    Returns:
        A new DataFrame containing the 'pattern_id', 'structure', 'ptype', and 'hash' columns.
    """
    # Create a reference DataFrame by selecting the relevant columns
    df_ref = df[['pattern_id', 'structure', 'ptype', 'hash']].copy()

    return df_ref


def sk(smiles: str, trailing_inchikey: bool = True) -> Optional[str]:
    """
    Converts a SMILES string into a scaffold key.

    Args:
        smiles: The SMILES string representing a chemical structure.
        trailing_inchikey: If True, includes trailing InChIKey in the scaffold key.

    Returns:
        The scaffold key as a string, or None if the conversion fails.
    """
    sk = smiles2scaffoldkey(smiles, trailing_inchikey)
    return sk


def sk_one(sk: str, has_inchikey: bool = True) -> Optional[str]:
    """
    Converts a scaffold key into a "one string" representation.

    Args:
        sk: The scaffold key.
        has_inchikey: If True, indicates that the scaffold key includes an InChIKey.

    Returns:
        The "one string" representation of the scaffold key, or None if the conversion fails.
    """
    sk_one_str = onestring(sk, has_inchikey)
    return sk_one_str


def tr_expand_coords(df: pd.DataFrame, source_col: str, id_col: str, delimiter: str) -> pd.DataFrame:
    """
    Expands coordinates from a delimited string column into multiple dimensions.

    Args:
        df: Pandas DataFrame containing the data.
        source_col: The column containing delimited coordinate strings.
        id_col: The column containing unique identifiers (currently not used in the function).
        delimiter: The delimiter used to split the coordinate string.

    Returns:
        A DataFrame with expanded coordinates in separate columns.
    """
    df_orig = df.copy()
    df_expanded = df[source_col].str.split(delimiter, expand=True)

    # Rename expanded columns
    nr_cols = len(df_expanded.columns)
    columns = [f'Dim_{i+1}' for i in range(nr_cols)]
    df_expanded.columns = columns
    df_expanded = df_expanded.astype('int32')

    # Concatenate original dataframe with expanded coordinates
    df_result = pd.concat([df_orig, df_expanded], axis=1)

    return df_result


def closest_scaffold(sk_struct: str, df_space: pd.DataFrame, scaffold_keys_arr: np.ndarray, weights: np.ndarray, use_cupy: bool = False) -> int:
    """
    Finds the closest reference scaffold for a given structure using preprocessed scaffold keys.
    
    Args:
        sk_struct: The scaffold structure (a string).
        df_space: DataFrame containing scaffold keys and related information.
        scaffold_keys_arr: Preprocessed scaffold keys.
        weights: Precomputed weight array (1 / np.arange(1, max_features + 1))
        row_based: Boolean to toggle between row-based (True) and matrix-based (False) computation
        use_cupy: Boolean to toggle between using CuPy (True) for GPU or NumPy (False) for CPU
        
    Returns:
        The order of the closest scaffold as an integer.
    """
    # Fast conversion of space-separated string to NumPy array
    sk_struct_arr = np.fromstring(sk_struct, sep=' ', dtype=np.float32)
    scaffold_keys_arr = scaffold_keys_arr.astype(float)

    if use_cupy:
        # Using CuPy for GPU-accelerated matrix-based computation

        
        # Transfer to GPU
        sk_struct_arr_cp = cp.asarray(sk_struct_arr)
        scaffold_keys_arr_cp = cp.asarray(scaffold_keys_arr, dtype=cp.float32)
        weights_cp = cp.asarray(weights)

        # Compute the diff and distance on the GPU
        diff = cp.abs(scaffold_keys_arr_cp - sk_struct_arr_cp[:, cp.newaxis]) ** 1.5
        distance = cp.einsum('ij,j->i', diff, weights_cp)  # Efficient weighted sum
        closest_index = cp.argmin(distance).get()  # Transfer result back to CPU for indexing
    else:
        # Normal NumPy computation (CPU)
        diff = np.abs(scaffold_keys_arr - sk_struct_arr) ** 1.5
        distance = np.einsum('ij,j->i', diff, weights)  # Efficient weighted sum
        closest_index = np.argmin(distance)

    return int(df_space.iloc[closest_index]['order'])



def get_bucket_id(closest_order: int, bucket_size: int) -> int:
    """
    Calculates the bucket ID based on the closest scaffold order and bucket size.

    Args:
        closest_order: The order of the closest scaffold.
        bucket_size: The size of each bucket.

    Returns:
        The bucket ID as an integer.
    """
    bucket_id = int(round(closest_order / bucket_size)) + 1
    return bucket_id


def get_hilbert_coordinates(hc: HilbertCurve, bucket_id: int) -> str:
    """
    Retrieves Hilbert curve coordinates based on the bucket ID.

    Args:
        hc: The HilbertCurve object.
        bucket_id: The ID of the bucket.

    Returns:
        A string representing the Hilbert coordinates, separated by semicolons.
    """
    coordinates = hc.point_from_distance(bucket_id - 1)
    coordinate_str = ';'.join(str(coord) for coord in coordinates)

    return coordinate_str


def train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trains the model by ordering scaffolds and preparing the reference scaffold set.

    Args:
        df: Pandas DataFrame with the following columns:
            - pattern_id: includes the pattern type and a unique ID separated by a "dot"
              example: scaffold.4
            - structure: SMILES structure
            - ptype: type of scaffold, example: scaffold
            - hash: InChIKey, example: NOWKCMXCCJGMRR-UHFFFAOYSA-N

    Returns:
        A DataFrame representing the ordered reference scaffolds set (df_space).
    """
    # Define reference scaffolds
    df = define_reference_scaffolds(df)

    # Order the reference scaffolds by their Scaffold Keys (SKs)
    df = order_scaffolds(df)

    df_space = df
    return df_space


def compute_max_phc_order(df_space: pd.DataFrame) -> int:
    """
    Computes the maximum PHC order based on the number of reference scaffolds.

    Args:
        df_space: DataFrame containing the reference scaffolds.

    Returns:
        The maximum PHC order (int).
    """
    log_base = 4

    # Number of reference scaffolds
    M = df_space.shape[0]

    # Compute the maximum PHC order using log base 4
    max_z = math.ceil(math.log(M, log_base))

    return int(max_z)


def preprocess_scaffolds(df_space: pd.DataFrame):
    """
    Preprocesses scaffold keys to remove the inchikey part and split them into components.
    This avoids the cost of splitting the string on every function call.
    
    Args:
        df_space: DataFrame containing scaffold keys.
        
    Returns:
        numpy array: Preprocessed scaffold keys for efficient distance computation.
    """
    # Split scaffold keys and remove inchikey (removes last element of split)
    scaffold_keys_split = df_space['scaffold_key'].str.split(' ').apply(lambda x: x[:-1])  # Remove inchikey
    scaffold_keys_arr = np.array(scaffold_keys_split.tolist(), dtype=object)
    return scaffold_keys_arr

def preprocess_scaffolds_vectorized(df_structures: pd.DataFrame) -> np.ndarray:
    """Convert all sk_struct strings to NumPy arrays at once."""
    return np.vstack(df_structures['sk_struct'].apply(lambda x: np.fromstring(x, sep=' ', dtype=np.float32)))


def compute_distances_vectorized(sk_struct_arr: np.ndarray, scaffold_keys_arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Vectorized distance computation."""
    diff = np.abs(scaffold_keys_arr[:, None, :] - sk_struct_arr[None, :, :]) ** 1.5  # Shape (num_scaffolds, num_structures, num_features)
    distances = np.einsum('ijk,k->ij', diff, weights)  # Shape (num_scaffolds, num_structures)
    
    return distances

def batch_process_find_closest_scaffolds(sk_struct_arr, scaffold_keys_arr, weights, batch_size, use_cupy=False):
    """Batch processing for finding closest scaffolds."""
    num_structures = sk_struct_arr.shape[0]
    all_distances = []  # We'll store all the distance computations here

    # Iterate through the data in batches
    for i in range(0, num_structures, batch_size):
        sk_struct_batch = sk_struct_arr[i:i + batch_size]
        
        if use_cupy:  # GPU-based computation with CuPy
            sk_struct_batch_cp = cp.asarray(sk_struct_batch, dtype=cp.float32)
            distances_batch = compute_distances_vectorized(
                sk_struct_batch_cp, cp.asarray(scaffold_keys_arr), cp.asarray(weights)
            )
            all_distances.append(distances_batch.get())  # Transfer result back to CPU
        else:  # CPU-based computation with NumPy
            distances_batch = compute_distances_vectorized(
                sk_struct_batch, scaffold_keys_arr, weights
            )
            all_distances.append(distances_batch)

    # Concatenate all batches' distances
    all_distances = np.concatenate(all_distances, axis=1)  # Shape (num_scaffolds, num_structures)

    # Find the closest scaffold for each structure after concatenation
    closest_indices = np.argmin(all_distances, axis=0)  # Get closest scaffold indices for each structure

    return closest_indices

def embed(df_space: pd.DataFrame, df_structures: pd.DataFrame, n_dim: int, row_based=False, use_cupy = True, batch_size=500) -> pd.DataFrame:
    """
    Embeds structures based on reference scaffolds using Pseudo Hilbert Curves (PHCs).

    Args:
        df_space: Pandas DataFrame generated by the hcase.train() method.
        df_structures: Pandas DataFrame containing the structures with the following columns:
            - structure: SMILES representation of the structure.
            - id: unique identifier of the compound.
        n_dim: Number of dimensions for the Hilbert curve.

    Returns:
        A DataFrame containing the embedded Hilbert space coordinates for the structures.
    """
    start_time = time.time()
    max_z = compute_max_phc_order(df_space)
    print(f"[TIMER] compute_max_phc_order took {time.time() - start_time:.4f} seconds.", flush=True)

    str_colname = 'structure'
    id_colname = 'id'

    df_structures = df_structures[[id_colname, str_colname]].copy()
    logger.info(df_structures.columns)

    logger.info("Generating Bemis-Murcko scaffolds for compounds ..")
    start_time = time.time()
    df_structures['bms'] = df_structures['structure'].parallel_apply(smiles2bmscaffold)
    print(f"[TIMER] smiles2bmscaffold took {time.time() - start_time:.4f} seconds.", flush=True)

    df_structures = df_structures[df_structures['bms'] != 'NA']
    logger.info(".. done")

    df_space = df_space.rename(columns={'structure': 'ref_scaffold_smiles'})
    nr_scaffolds = df_space.shape[0]

    logger.info("Generating Scaffold-Keys for the Bemis-Murcko scaffolds of compounds ..")
    start_time = time.time()
    df_structures['sk_struct'] = df_structures['bms'].parallel_apply(lambda x: smiles2scaffoldkey(x, trailing_inchikey=False))
    print(f"[TIMER] smiles2scaffoldkey took {time.time() - start_time:.4f} seconds.", flush=True)

    df_structures = df_structures[df_structures['sk_struct'] != 'NA']
    df_structures = df_structures.reset_index(drop=True)
    df_structures['idx'] = df_structures.index + 1

    logger.info("Identifying the closest reference scaffolds of compounds ..")
    start_time = time.time()

    # Preprocess scaffold keys only once
    scaffold_keys_arr = preprocess_scaffolds(df_space)
    scaffold_keys_arr = scaffold_keys_arr.astype(float)

    # Precompute weights once
    num_features = scaffold_keys_arr.shape[1]  # Assuming scaffold_keys_arr is (num_scaffolds, num_features)
    weights = 1 / np.arange(1, num_features + 1)


    if row_based:
        # Row-based approach (using Pandarallel for parallel processing)
        df_structures['closest_order'] = df_structures['sk_struct'].parallel_apply(
            lambda sk_struct: closest_scaffold(sk_struct, df_space, scaffold_keys_arr, weights, use_cupy=use_cupy)
        )
    else:
        # Vectorized approach (using CuPy or NumPy)
        sk_struct_arr = preprocess_scaffolds_vectorized(df_structures)

        # Use batching for large datasets
        closest_indices = batch_process_find_closest_scaffolds(
            sk_struct_arr, scaffold_keys_arr, weights, batch_size, use_cupy
        )

        # Assign back to DataFrame
        df_structures['closest_order'] = df_space.iloc[closest_indices]['order'].values

    
    print(f"[TIMER] closest_scaffold took {time.time() - start_time:.4f} seconds.", flush=True)

    logger.info(".. done")

    df_res = pd.DataFrame()
    first = True

    for hc_order in range(2, max_z + 1):
        start_time = time.time()
        bucket_nr = math.pow(math.pow(2, hc_order), n_dim)
        bucket_size = int(round(nr_scaffolds / (bucket_nr - 1)))
        logger.info(f'Generating HCASE embedding at PHC order: {hc_order}, '
                    f'nr_scaffolds: {nr_scaffolds}, bucket_nr: {int(bucket_nr)}, bucket_size {bucket_size:.4f} ..')

        hilbert_curve = HilbertCurve(hc_order, n_dim)

        df_hilbert = df_structures.copy()

        logger.info(f'Mapping compounds to pseudo-Hilbert-Curve of z={hc_order} ..')
        start_step_time = time.time()
        df_hilbert['bucket_id'] = df_hilbert['closest_order'].parallel_apply(lambda x: get_bucket_id(x, bucket_size))
        print(f"[TIMER] get_bucket_id took {time.time() - start_step_time:.4f} seconds.", flush=True)

        logger.info(".. done")

        logger.info("Determining the 2D/3D coordinates of compounds in the HCASE map ..")
        start_step_time = time.time()
        df_hilbert['embedded_hs_coordinates'] = df_hilbert['bucket_id'].parallel_apply(lambda x: get_hilbert_coordinates(hilbert_curve, x))
        print(f"[TIMER] get_hilbert_coordinates took {time.time() - start_step_time:.4f} seconds.", flush=True)

        logger.info(".. done")

        start_step_time = time.time()
        df_hilbert = tr_expand_coords(df_hilbert, 'embedded_hs_coordinates', id_colname, delimiter=';')
        print(f"[TIMER] tr_expand_coords took {time.time() - start_step_time:.4f} seconds.", flush=True)

        df_hilbert['hc_order'] = hc_order

        if first:
            df_res = df_hilbert
            first = False
        else:
            df_res = pd.concat([df_res, df_hilbert], ignore_index=True)

        print(f"[TIMER] PHC order {hc_order} processing took {time.time() - start_time:.4f} seconds.", flush=True)
        logger.info(".. done.")

    return df_res


