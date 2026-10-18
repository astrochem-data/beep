import logging
from typing import Any, Dict, List, Tuple, Union, NoReturn
import pandas as pd
import qcelemental as qcel

try:
    from .. import __version__
except ImportError:  # partially initialised package or uninstalled checkout
    __version__ = "0.0.0+unknown"

_LINE = "=" * 88


def beep_banner(workflow_name, quote=None, quote_author=None, authors=None, tagline=None):
    """Return the universal BEEP banner with workflow-specific details."""
    lines = [
        '''
·······················································································
:                                                                                     :
:  ██████╗ ██╗███╗   ██╗██████╗ ██╗███╗   ██╗ ██████╗                                 :
:  ██╔══██╗██║████╗  ██║██╔══██╗██║████╗  ██║██╔════╝                                 :
:  ██████╔╝██║██╔██╗ ██║██║  ██║██║██╔██╗ ██║██║  ███╗                                :
:  ██╔══██╗██║██║╚██╗██║██║  ██║██║██║╚██╗██║██║   ██║                                :
:  ██████╔╝██║██║ ╚████║██████╔╝██║██║ ╚████║╚██████╔╝                                :
:  ╚═════╝ ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝                                 :
:                                                                                     :
:  ███████╗███╗   ██╗███████╗██████╗  ██████╗ ██╗   ██╗                               :
:  ██╔════╝████╗  ██║██╔════╝██╔══██╗██╔════╝ ╚██╗ ██╔╝                               :
:  █████╗  ██╔██╗ ██║█████╗  ██████╔╝██║  ███╗ ╚████╔╝                                :
:  ██╔══╝  ██║╚██╗██║██╔══╝  ██╔══██╗██║   ██║  ╚██╔╝                                 :
:  ███████╗██║ ╚████║███████╗██║  ██║╚██████╔╝   ██║                                  :
:  ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝                                  :
:                                                                                     :
:  ███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗  :
:  ██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║  :
:  █████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║  :
:  ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║  :
:  ███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║  :
:  ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝  :
:                                                                                     :
:  ██████╗ ██╗      █████╗ ████████╗███████╗ ██████╗ ██████╗ ███╗   ███╗              :
:  ██╔══██╗██║     ██╔══██╗╚══██╔══╝██╔════╝██╔═══██╗██╔══██╗████╗ ████║              :
:  ██████╔╝██║     ███████║   ██║   █████╗  ██║   ██║██████╔╝██╔████╔██║              :
:  ██╔═══╝ ██║     ██╔══██║   ██║   ██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║              :
:  ██║     ███████╗██║  ██║   ██║   ██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║              :
:  ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝              :
:                                                                                     :
·······················································································
        '''
        "",
        _LINE,
        f"BEEP \u2014 v{__version__}".center(88),
        _LINE,
        "",
        f"  Workflow:  {workflow_name}",
        "",
    ]
    if quote:
        lines.append(f'  "{quote}"')
        if quote_author:
            lines.append(f"{'':>40}\u2014 {quote_author}")
        lines.append("")
    if tagline:
        lines.append(f"  {tagline}")
        lines.append("")
    if authors:
        lines.append(f"  By: {authors}")
        lines.append("")
    lines.append("-" * 88)
    return "\n".join(lines)


def setup_logging(prefix: str, molecule_name: str) -> logging.Logger:
    """
    Sets up a logging system for a specific molecule sampling process.

    This function configures logging for the beep_sampling module. It creates a log file with a name based
    on the given molecule name. It also sets up logging to both the file and the console with INFO level.
    The log messages are formatted to include only the message part.

    Parameters:
    molecule_name (str): The name of the molecule, which will be used to create the log file name.

    Returns:
    logging.Logger: The configured logger object for the beep_sampling module.
    """
    logger = logging.getLogger("beep")
    logger.setLevel(logging.INFO)

    # File handler setup
    log_file =  prefix +  f"_{molecule_name}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Console handler setup
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


def log_formatted_list(logger: logging.Logger, my_list: List[str], description: str, max_rows: int = 10):
    """
    Converts a list into formatted strings organized in a table with at most max_rows rows
    and logs it using the provided logger.

    Parameters:
    logger (logging.Logger): The logger object to use for logging the information.
    my_list (List[str]): The list to be formatted and logged.
    max_rows (int): Maximum number of rows in the table.
    """
    # Validate max_rows
    if max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")

    # Determine the number of columns needed
    num_columns = len(my_list) // max_rows + (1 if len(my_list) % max_rows else 0)

    # Create a list of rows, each containing up to num_columns items
    table = []
    for row_idx in range(max_rows):
        row = []
        for col_idx in range(num_columns):
            # Calculate the index in the original list
            item_idx = col_idx * max_rows + row_idx
            if item_idx < len(my_list):
                row.append(f"* {my_list[item_idx]}")
            else:
                row.append("")  # Empty string if no item for this slot
        table.append(row)

    # Calculate the width of the widest item for formatting
    max_width = max(len(item) for item in my_list) + 4  # Add 4 for the "* " prefix and some padding

    # Format each row with aligned columns
    formatted_rows = ["\t".join(item.ljust(max_width) for item in row) for row in table]

    # Join the rows into a formatted string
    formatted_string = "\n".join(formatted_rows)

    # Log the formatted string
    logger.info(f"{description}\n{formatted_string}")


def dict_to_log(logger: logging.Logger, data: dict):
    """
    Logs the contents of a dictionary where each key has a list of items.
    Each key is logged followed by a two-column table with the items of the list.

    Parameters:
    logger (logging.Logger): The logger object used for logging.
    data (dict): The dictionary to be logged.
    """
    for key, items in data.items():
        logger.info(f"\nXC Functional Type: {key}")
        logger.info(f"{'Index':<10} | {'Item':<30}")
        logger.info(f"{'-' * 10}+{'-' * 30}")
        for index, item in enumerate(items):
            logger.info(f"{index:<10} | {item:<30}")

def padded_log(logger: logging.Logger, message: str, variable: str ='', padding_char: str = '-', total_length: int = 90):
    """
    Logs a message with custom padding characters on both sides, padding to a fixed total length,
    and includes the value of a variable within the message.

    Parameters:
    logger (logging.Logger): The logger object used for logging.
    message (str): The message template to be logged, with placeholders for variables.
    variable (str): The variable's value to be inserted into the message.
    padding_char (str): The character used for padding.
    total_length (int): The total fixed length of the log message, including padding.
    """
    # Format the message with the variable value
    if '{}' in message and variable:
        formatted_message = message.format(variable)
    else:
        formatted_message = message

    # Length of the text within the padding
    text_length = len(formatted_message)

    # Calculate the number of padding characters needed for padding on each side
    padding_each_side = (total_length - text_length - 2) // 2  # -2 for the spaces after and before the padding chars

    # Adjust for odd total lengths
    padding_extra = "" if (total_length - text_length) % 2 == 0 else padding_char

    # Construct the padded message
    padded_message = f"\n{padding_char * padding_each_side} {formatted_message} {padding_char * padding_each_side}{padding_extra}\n"

    logger.info(padded_message)

def log_progress(logger: logging.Logger, current_step: int, total_steps: int, bar_length: int = 50) -> NoReturn:
    """
    Logs a progress bar to the given logger.

    This function logs a progress bar by calculating the percentage completion and represents it as a bar of '=' characters.

    Parameters:
    logger (logging.Logger): The logger to which the progress bar will be logged.
    current_step (int): The current step (or iteration) of the operation.
    total_steps (int): The total number of steps (or iterations) in the operation.
    bar_length (int): The character length of the progress bar. Defaults to 50.

    Returns:
    NoReturn
    """
    if not (0 <= current_step <= total_steps):
        raise ValueError("Current step is out of the bounds of total steps")

    proportion_complete = current_step / total_steps
    filled_length = int(proportion_complete * bar_length)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    logger.info(f"[{bar}] {current_step}/{total_steps} Complete")

def log_dataframe_averages(logger: logging.Logger, df: pd.DataFrame):
    """
    Logs the average values of each group in the dataframe, excluding the first unnecessary column,
    with the functional group as a header above each group's table. Also logs a summary table at
    the end with the functional group and level of theory corresponding to the minimum average value.

    Parameters:
    logger (logging.Logger): The logger object used for logging.
    df (pd.DataFrame): The dataframe containing the data.
    """
    df = df.iloc[:, 1:]  # Drop the first column
    grouped_columns = df.columns.to_series().groupby(lambda x: x.split('_')[1] + '_' + x.split('_')[2]).groups
    summary_data = {}

    for group, columns in grouped_columns.items():
        logger.info(f"\nFunctional Group: {group}")
        logger.info(f"{'Level of Theory':<30} | Average")
        logger.info('-' * 45)

        group_averages = df[columns].mean()
        min_avg_value = group_averages.min()
        min_avg_theory = group_averages.idxmin().split('_')[3:]  # Assuming the level of theory is from the 4th split
        min_avg_theory = '_'.join(min_avg_theory)

        for col in columns:
            level_of_theory = '_'.join(col.split('_')[3:])
            avg_value = df[col].mean()
            mark = '*' if avg_value == min_avg_value else ' '
            logger.info(f"{level_of_theory:<30} | {avg_value:.6f} {mark}")

        summary_data[group] = (min_avg_theory, min_avg_value)
    # Log the summary table
    logger.info("\nSummary of minimum average RMSD values per functional group:")
    logger.info(f"{'Functional Group':<30} | {'Level of Theory':<30} | {'Min Average RMSD':<30}")
    logger.info('-' * 95)
    for group, (theory, min_avg) in summary_data.items():
        logger.info(f"{group:<30} | {theory:<30} | {min_avg:.6f}")



def log_energy_mae(logger: logging.Logger, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Mean Absolute Error (MAE) for each method_basis pair in a DataFrame and log the results.

    This function takes a DataFrame where each row corresponds to a different structure and each column
    corresponds to a different computational method. It computes the MAE for each method_basis pair,
    and logs the 15 best-performing methods (those with the lowest MAE) for each pair. The best overall
    method is marked with an asterisk.

    Parameters:
    logger (logging.Logger): The logger object to use for logging the MAE results.
    df (pd.DataFrame): The DataFrame containing the computational methods' errors.

    Returns:
    pd.DataFrame: A new DataFrame containing the MAEs for each method_basis pair.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Step 1: Extract the method_basis pair and add it as a new column
    df_copy['method_basis'] = df_copy.index.map(lambda x: '/'.join(x.split('_')[-2:]))
    abs_columns = df_copy.columns.difference(['method_basis'])
    df_copy[abs_columns] = df_copy[abs_columns].abs()
    mae_results = df_copy.groupby('method_basis').mean()

    # Determine the maximum width for method names for formatting
    max_method_len = max(len(method) for method in mae_results.columns) + 1  # Adding space for the pipe symbol

    # Header for the log
    header = f"\n{'Level of Theory':<{max_method_len}} | MAE\n" + "-" * (max_method_len + 6)  # 6 for " | MAE" and newline

    # Log the header for MAE results
    logger.info(header)

    # New logging capability with neat formatting
    for index, row in mae_results.iterrows():
        sorted_row = row.sort_values()
        min_value = sorted_row.iloc[0]
        log_message = [f"Lowest MAEs for {index} geometry:\n"]

        # Formatting each line to match the requested structure
        for col in sorted_row.index[:15]:
            value = sorted_row[col]
            star = "*" if value == min_value else " "
            method_formatted = col.ljust(max_method_len - 2)  # Adjust for the length of " |"
            log_message.append(f"{method_formatted} | {value:.6f}{star}\n")

        # Combine the log message and log it
        log_message_str = ''.join(log_message)
        logger.info(log_message_str)

    return mae_results

def log_dataframe(logger: logging.Logger, df: pd.DataFrame, title: str = "DataFrame") -> None:
    """
    Logs the contents of a DataFrame using the provided logger.

    Args:
        logger (logging.Logger): The logger to use for logging the DataFrame.
        df (pd.DataFrame): The DataFrame to be logged.
        title (str): The title to prepend to the logged DataFrame. Defaults to "DataFrame".

    Returns:
        None
    """
    # Convert the DataFrame to a string with pretty formatting
    df_string = df.to_string()

    # Log the DataFrame with the provided title
    logger.info(f"{title}\n{df_string}")

def log_dictionary(logger: logging.Logger, dictionary: Dict[Any, Any], title: str = "Dictionary") -> None:
    """
    Logs the contents of a dictionary using the provided logger.

    Args:
        logger (logging.Logger): The logger to use for logging the dictionary.
        dictionary (Dict[Any, Any]): The dictionary to be logged.  The values are
            expected to be tuples in the form ``(m, n, R^2)``.
        title (str): The title to prepend to the logged dictionary.
            Defaults to ``"Dictionary"``.

    Returns:
        None
    """
    # Log the dictionary with the provided title
    logger.info(f"{title}:")
    for key, value in dictionary.items():
        logger.info(f"{key}: m = {value[0]}, n = {value[1]}, R^2 = {value[2]}")

def write_energy_log(df: pd.DataFrame, mol: str, existing_content: str = "", comment: str = "") -> str:
    """
    Appends energy values from the DataFrame to a log string.

    Args:
        df (pd.DataFrame): DataFrame containing energy values.
        mol (str): Name of the molecule.
        existing_content (str, optional): Existing log content to append to. Defaults to an empty string.
        comment (str, optional): Additional comment to add to the log. Defaults to an empty string.

    Returns:
        str: The updated log content string with appended energy values.
    """
    content = existing_content
    if not content:
        content += f"{'':<50}{'BE Average':<50}{'BE Standard Deviation'}\n"

    if len(df) == 0:
        content += f"{mol} {comment:<30}NO DATA (all structures filtered out)\n"
        return content

    # BE Average: mean of Mean_Eb_all_dft ± mean of StdDev_all_dft (inter-method spread)
    mean_be = df["Mean_Eb_all_dft"].mean()
    mean_intermethod = df["StdDev_all_dft"].mean()

    # BE Standard Deviation: std of Mean_Eb_all_dft across binding sites ± its SEM
    std_be = df["Mean_Eb_all_dft"].std()
    sem_std = df["StdDev_all_dft"].std()

    content += f"{mol} {comment:<30}"
    # Column 1: BE Average
    avg_values = []
    for unit in ['kcal/mol', 'K']:
        conv = qcel.constants.conversion_factor('kcal/mol', unit)
        avg_val = abs(round(conv * mean_be, 2))
        avg_unc = abs(round(conv * mean_intermethod, 2))
        avg_values.append(f"{avg_val:.2f} ± {avg_unc:.2f} [{unit}]")
    content += f"{'   '.join(avg_values):<50}"
    # Column 2: BE Standard Deviation
    std_values = []
    for unit in ['kcal/mol', 'K']:
        conv = qcel.constants.conversion_factor('kcal/mol', unit)
        std_val = round(conv * std_be, 2)
        std_unc = abs(round(conv * sem_std, 2))
        std_values.append(f"{std_val:.2f} ± {std_unc:.2f} [{unit}]")
    content += f"{'   '.join(std_values):<50}"
    content += "\n"
    return content


def _strip_group_prefix(name: str) -> str:
    """Drop the 'geom_' / 'energy_' style prefix from functional-group names
    so that they match the display style used by ``log_dataframe_averages``
    (e.g. ``geom_hmgga_dz`` → ``hmgga_dz``)."""
    for prefix in ("geom_", "energy_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _bias_tag(bias: float) -> str:
    """Human-readable annotation for a signed-mean-error (kcal/mol).

    Sign convention: BE is negative (stabilisation), so a negative bias
    means the DFT BE is more negative than the reference — DFT overbinds.
    A positive bias means DFT underbinds.

    Magnitude thresholds (kcal/mol):
      |bias| < 0.05  → balanced
      |bias| < 0.5   → mild over/underbind
      otherwise      → strong over/underbind
    """
    mag = abs(bias)
    direction = "underbind" if bias > 0 else "overbind"
    if mag < 0.05:
        return "balanced"
    if mag < 0.5:
        return f"mild {direction}"
    return f"strong {direction}"


def log_energy_mae_per_group(
    logger: logging.Logger,
    df_ae: pd.DataFrame,
    dft_func_dict: dict,
) -> None:
    """Per-category MAE breakdown for the energy_benchmark output.

    Mirrors ``log_trajectory_metrics_per_group``: one table per
    functional category (GGA, Meta-GGA, Hybrid, Meta-Hybrid, LRC,
    Non-local, …) listing the methods present in ``df_ae`` whose
    column name matches a member of that category, sorted ascending by
    MAE, with ``*`` next to the winner. A summary table at the bottom
    lists the per-group winner.

    Functionals may belong to **multiple** categories (e.g. ``B97M-V``
    is both Meta-GGA and Non-local); those columns appear in every
    matching category table.

    Categories with no surviving columns (because every functional in
    that list was filtered out by the integrated-dispersion convention)
    are silently skipped.

    Parameters
    ----------
    logger : logging.Logger
        Active BEEP logger.
    df_ae : pd.DataFrame
        Absolute-error DataFrame (entries × method/basis columns), as
        produced by ``get_errors_dataframe``.
    dft_func_dict : dict
        ``{group_display_name → [functional_name, ...]}`` mapping. The
        functional names are matched case-insensitively against the
        bare-method portion of each column (``col.rsplit("/", 1)[0]``).
    """
    if df_ae.empty:
        logger.warning("\n  No MAE data to display.\n")
        return

    # MAE per column = mean of |signed error|. Despite the "_ae" suffix in
    # the caller's variable name, the DataFrame from get_errors_dataframe is
    # signed (E_dft − E_ref); take abs() before averaging — same convention
    # as the legacy ``log_energy_mae``. The unsigned mean of the signed
    # error is the per-functional bias (see _bias_tag below).
    mae = df_ae.abs().mean(axis=0)
    bias = df_ae.mean(axis=0)
    name_width = 30
    summary = {}   # group_display -> (winner, winner_mae)

    for group, funcs in dft_func_dict.items():
        funcs_lower = {f.lower() for f in funcs}
        in_group_cols = [
            c for c in df_ae.columns
            if c.rsplit("/", 1)[0] in funcs_lower
        ]
        if not in_group_cols:
            continue

        group_maes = mae[in_group_cols].sort_values()
        winner = str(group_maes.idxmin())
        winner_mae = float(group_maes.iloc[0])
        summary[group] = (winner, winner_mae)

        logger.info(f"\nFunctional Group: {group}")
        header = (
            f"{'Level of Theory':<{name_width}} | "
            f"{'MAE':<10} | {'Bias':<10} | Tag"
        )
        logger.info(header)
        logger.info("-" * max(len(header), name_width + 40))
        for col, val in group_maes.items():
            mark = "*" if col == winner else " "
            b = float(bias[col])
            logger.info(
                f"{col:<{name_width}} | {val:<10.6f} | {b:<+10.4f} | "
                f"{_bias_tag(b)} {mark}"
            )

    if not summary:
        return

    logger.info("\nSummary of best functional per group (lowest MAE):")
    summary_header = (
        f"{'Functional Group':<{name_width}} | "
        f"{'Level of Theory':<{name_width}} | {'MAE':<12}"
    )
    logger.info(summary_header)
    logger.info("-" * len(summary_header))
    for group, (winner, val) in summary.items():
        logger.info(
            f"{group:<{name_width}} | {winner:<{name_width}} | {val:.6f}"
        )


def log_trajectory_metrics_per_group(
    logger: logging.Logger,
    traj_metrics: dict,
    dft_geom_functionals: dict,
    ranking_df: pd.DataFrame = None,
    metric_key: str = "rmsd_force",
    metric_label: str = "RMSD_F",
) -> None:
    """Per-group trajectory force-RMSD table, BENCHMARK RESULTS style.

    For each functional group, prints the per-Cartesian-component force
    RMSD (meV/Å) vs reference. The row with the best (lowest) combined
    score in the group is marked with ``*``; falls back to best
    ``rmsd_force`` when no ``ranking_df`` is supplied. A summary table
    lists the per-group winner at the bottom.

    Energy is not displayed — absolute energies aren't meaningful for
    geometry benchmarking (use the energy_benchmark workflow for
    relative-energy comparison).

    Parameters
    ----------
    logger : logging.Logger
        Active BEEP logger.
    traj_metrics : dict
        ``{lowercase_functional → {rmsd_energy, rmsd_force, ...}}``.
    dft_geom_functionals : dict
        ``{group_name → [mixed_case_functional, ...]}`` mapping. The
        group display name has any ``geom_`` / ``energy_`` prefix
        stripped to match ``log_dataframe_averages``.
    ranking_df : pd.DataFrame, optional
        Output of ``combined_zscore_ranking``; if provided, the
        per-group winner is chosen by ``combined_score``.
    metric_key, metric_label : str, optional
        Metric shown and used to pick the winner when ``ranking_df`` is
        None (default: the total force RMSD, ``rmsd_force``).
    """
    if not traj_metrics:
        logger.warning("\n  No trajectory metrics to display.\n")
        return

    name_width = 30
    metric_header = f"{metric_label} (meV/Å)"
    summary = {}   # display_group -> (winner_mixed_case, best_metric_value)

    for group, funcs in dft_geom_functionals.items():
        group_display = _strip_group_prefix(group)
        in_group = [
            f for f in funcs
            if f.lower() in traj_metrics and metric_key in traj_metrics[f.lower()]
        ]
        if not in_group:
            continue

        if ranking_df is not None:
            ranked = [f for f in in_group if f.lower() in ranking_df.index]
            if ranked:
                lower_to_orig = {f.lower(): f for f in ranked}
                best_lc = ranking_df.loc[
                    list(lower_to_orig), "combined_score"
                ].idxmin()
                best_func = lower_to_orig[best_lc]
                summary[group_display] = (
                    best_func, ranking_df.at[best_lc, "combined_score"],
                )
            else:
                best_func = None
        else:
            best_func = min(
                in_group, key=lambda f: traj_metrics[f.lower()][metric_key],
            )
            summary[group_display] = (
                best_func, traj_metrics[best_func.lower()][metric_key],
            )

        logger.info(f"\nFunctional Group: {group_display}")
        header = f"{'Level of Theory':<{name_width}} | {metric_header}"
        logger.info(header)
        logger.info("-" * max(len(header), name_width + 18))
        for func in in_group:
            v = traj_metrics[func.lower()][metric_key]
            mark = "*" if func == best_func else " "
            logger.info(f"{func:<{name_width}} | {v:<14.4f} {mark}")

    if not summary:
        return
    if ranking_df is not None:
        title = "Summary of best functional per group (by combined score):"
        score_header = "Combined Score"
        fmt = "{:<+14.4f}"
    else:
        title = f"Summary of best functional per group (by {metric_label}):"
        score_header = f"{metric_label} (meV/Å)"
        fmt = "{:<14.4f}"

    logger.info(f"\n{title}")
    summary_header = (
        f"{'Functional Group':<{name_width}} | "
        f"{'Level of Theory':<{name_width}} | {score_header:<14}"
    )
    logger.info(summary_header)
    logger.info("-" * len(summary_header))
    for group, (winner, score) in summary.items():
        logger.info(
            f"{group:<{name_width}} | {winner:<{name_width}} | "
            + fmt.format(score)
        )


def log_nm_force_decomposition(
    logger: logging.Logger,
    nm_metrics: dict,
    dft_geom_functionals: dict,
    sort_key: str = "rmsd_force_intermolecular",
    sort_label: str = "Net force",
) -> None:
    """Per-group table of the nm_sampling force-error decomposition.

    Columns: total force RMSD, its split over ± pairs into offset (force
    error at the displacement centre) and response (force error that grows
    with the displacement), the rigid-body intermolecular parts (net force
    and torque per fragment) and the predicted intermolecular shift of the
    minimum (Å). Rows are sorted by ``sort_key`` (ascending); the best row
    is marked ``*`` and listed in a per-group summary. Functionals without
    decomposition data are skipped.
    """
    cols = [
        ("rmsd_force_intermolecular", "Net force", "{:>10.2f}"),
        ("rmsd_torque", "Torque", "{:>10.2f}"),
        ("geometry_shift_intermolecular", "Shift (Å)", "{:>10.3f}"),
        ("rmsd_force_response", "Response", "{:>10.2f}"),
        ("rmsd_force_offset", "Offset", "{:>10.2f}"),
        ("rmsd_force", "Total", "{:>10.2f}"),
    ]
    name_width = 30
    nan = float("nan")

    def _key(func):
        v = nm_metrics[func.lower()].get(sort_key, nan)
        return (v != v, v)          # NaN sorts last

    summary = {}
    for group, funcs in dft_geom_functionals.items():
        in_group = [
            f for f in funcs
            if "rmsd_force_response" in nm_metrics.get(f.lower(), {})
        ]
        if not in_group:
            continue
        in_group = sorted(in_group, key=_key)
        best = in_group[0]
        group_display = _strip_group_prefix(group)
        summary[group_display] = (best, nm_metrics[best.lower()].get(sort_key, nan))

        logger.info(f"\nFunctional Group: {group_display}   (sorted by {sort_label})")
        header = f"{'Level of Theory':<{name_width}}" + "".join(
            f" | {label:>10}" for _, label, _ in cols
        )
        logger.info(header)
        logger.info("-" * len(header))
        for func in in_group:
            m = nm_metrics[func.lower()]
            mark = " *" if func == best else ""
            logger.info(f"{func:<{name_width}}" + "".join(
                " | " + fmt.format(m.get(key, nan)) for key, _, fmt in cols
            ) + mark)

    if not summary:
        return
    logger.info(f"\nSummary of best functional per group (by {sort_label}):")
    summary_header = (
        f"{'Functional Group':<{name_width}} | "
        f"{'Level of Theory':<{name_width}} | {sort_label}"
    )
    logger.info(summary_header)
    logger.info("-" * len(summary_header))
    for group, (winner, value) in summary.items():
        logger.info(f"{group:<{name_width}} | {winner:<{name_width}} | {value:.3f}")


# Per-metric display metadata used by log_trajectory_ranking_table.
# Maps metric column name → (short header, long description with units).
# Add entries here when a new ranking metric is introduced.
_RANKING_METRIC_META = {
    "rmsd_eq":    ("RMSD_eq", "equilibrium-geometry deviation, Å"),
    "rmsd_force": ("RMSD_F",  "per-Cartesian-component force RMSE, meV/Å"),
}


def log_trajectory_ranking_table(
    logger: logging.Logger,
    ranking_df: pd.DataFrame,
    score_weights: dict = None,
) -> None:
    """Combined z-score-weighted ranking with an inline methodology block.

    Explains the score formula in the log itself so anyone reading the
    log can audit the ranking without having to dig into the source.

    Generic over any number of metrics — the columns displayed are
    derived from ``score_weights.keys()`` so the same helper serves the
    geom-trajectory case (2 metrics) and the nm-sampling case (1 metric).

    Parameters
    ----------
    logger : logging.Logger
        Active BEEP logger.
    ranking_df : pd.DataFrame
        Output of ``combined_zscore_ranking``; rows are functionals
        (lowercase), columns include the raw metrics, their z-scores
        (``z_<metric>``), and ``combined_score``.
    score_weights : dict, optional
        ``{metric_name → weight}``. Determines which columns are
        displayed and the methodology block. Defaults to equal weights
        on (``rmsd_eq``, ``rmsd_force``).
    """
    if ranking_df is None or ranking_df.empty:
        logger.warning(
            "\n  Combined ranking unavailable "
            "(no functionals with complete data).\n"
        )
        return

    w = score_weights or {"rmsd_eq": 1.0, "rmsd_force": 1.0}
    metric_names = list(w.keys())
    raw_headers = [_RANKING_METRIC_META.get(m, (m, m))[0] for m in metric_names]
    cols = list(metric_names) + [f"z_{m}" for m in metric_names] + ["combined_score"]
    headers = (
        raw_headers
        + [f"z({h})" for h in raw_headers]
        + ["Score"]
    )

    methods = list(ranking_df.index)
    width = max(36, max(len(m) for m in methods) + 2)

    logger.info("")
    logger.info("  Methodology")
    if len(metric_names) == 1:
        m = metric_names[0]
        h, desc = _RANKING_METRIC_META.get(m, (m, m))
        logger.info(f"    One metric — {h} ({desc}) — converted to a")
        logger.info("    z-score across the benchmarked functionals:")
    else:
        descs = ", ".join(
            f"{_RANKING_METRIC_META.get(m, (m, m))[0]} ({_RANKING_METRIC_META.get(m, (m, m))[1]})"
            for m in metric_names
        )
        logger.info(
            f"    {len(metric_names)} metrics — {descs} — have different"
        )
        logger.info("    units. To combine them we convert each to a z-score")
        logger.info("    across the benchmarked functionals:")
    logger.info("        z_m = (value_m - mean) / std         (population std, ddof=0)")
    logger.info("    The combined score is a weighted sum of the z-scores:")
    formula = " + ".join(f"w_{h} · z({h})" for h in raw_headers)
    logger.info(f"        score = {formula}")
    weights_line = ", ".join(
        f"w_{h} = {w[m]}" for m, h in zip(metric_names, raw_headers)
    )
    logger.info(f"    Current weights: {weights_line}")
    logger.info("    (override via 'score_weights' in the workflow config.)")
    logger.info("    Z-scores are dimensionless: this puts the metrics on a")
    logger.info("    common scale without arbitrary unit conversions.")
    logger.info("    Lower combined score = better.")
    logger.info("")
    logger.info("    Absolute energies are not part of the score — they're")
    logger.info("    dominated by total-energy-scale differences (correlation,")
    logger.info("    basis, BSSE) rather than PES quality. Use the")
    logger.info("    energy_benchmark workflow for relative-energy comparison.")
    logger.info("")
    header = f"  {'Functional':<{width}s}" + "".join(f"{h:<14s}" for h in headers)
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for m in methods:
        line = f"  {m:<{width}s}"
        for col in cols:
            line += f"{ranking_df.at[m, col]:<+14.4f}"
        logger.info(line)
