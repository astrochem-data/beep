"""
Binding energy stoichiometry computation — pure logic.

Uses qcelemental.models.Molecule (standalone, no QCFractal dependency).
"""
import logging
from typing import List, Tuple, Dict
from qcelemental.models.molecule import Molecule


def be_stoichiometry(smol_mol: Molecule, cluster_mol: Molecule, struc_mol: Molecule, logger: logging.Logger) -> Dict[str, List[Tuple[Molecule, float]]]:
    """
    Generates the Binding Energy (BE) stoichiometry for a given molecular system.

    This function computes the BE stoichiometry for different scenarios, including the
    BSSE-corrected BE stoichiometry, BE without counterpoise (nocp), counterpoise-corrected
    interaction energy (ie), ghost-free interaction energy (ie_nocp), and deformation
    energy (de).

    ``ie_nocp`` is the interaction energy with the fragments taken at their in-complex
    geometry but WITHOUT ghost atoms (real fragments only). It equals ``be_nocp - de``
    and, unlike ``ie``, is well-defined for basis-set-free potentials (e.g. MACE MLPs),
    which cannot evaluate the counterpoise ghost atoms used by ``ie``/``bsse``.

    Parameters:
        smol_mol (Molecule): The small molecule bound to the surface.
        cluster_mol (Molecule): The surface or cluster model molecule.
        struc_mol (Molecule): The full structure with both the small molecule and
            cluster bound together.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        dict: A dictionary mapping scenario names (``'bsse'``, ``'be_nocp'``,
        ``'ie'``, ``'de'``) to lists of ``(Molecule, float)`` tuples representing
        the stoichiometry coefficients.
    """
    # Flatten the structure geometry and get the symbols
    geom = struc_mol.geometry.flatten()
    symbols = struc_mol.symbols
    surf_symbols = cluster_mol.symbols

    # Create a fragmented molecule with the surface as one fragment and the small molecule as another
    f_struc_mol = Molecule(
        symbols=symbols,
        geometry=geom,
        molecular_multiplicity=smol_mol.molecular_multiplicity,
        fragments=[
            list(range(0, len(surf_symbols))),
            list(range(len(surf_symbols), len(symbols))),
        ],
        fragment_multiplicities = [1, smol_mol.molecular_multiplicity]
    )

    # Fragment extraction
    j5 = f_struc_mol.get_fragment(0)  # Surface fragment
    j4 = f_struc_mol.get_fragment(1)  # Small molecule fragment
    j7 = f_struc_mol.get_fragment(0, 1)  # Combined surface and small molecule
    j6 = f_struc_mol.get_fragment(1, 0)  # Alternative combined fragment

    logger.debug(f"Fragments generated: j4={j4}, j5={j5}, j6={j6}, j7={j7}")
    logger.debug(
    "Fragment multiplicities: "
    f"j4 (small molecule) = {j4.molecular_multiplicity}, "
    f"j5 (surface) = {j5.molecular_multiplicity}, "
    f"j6 (combined 1,0) = {j6.molecular_multiplicity}, "
    f"j7 (combined 0,1) = {j7.molecular_multiplicity}"
)

    # Binding energy stoichiometry dictionary
    be_stoic = {
        "bsse": [
            (f_struc_mol, 1.0),
            (j4, 1.0),
            (j5, 1.0),
            (j7, -1.0),
            (j6, -1.0),
            (cluster_mol, -1.0),
            (smol_mol, -1.0),
        ],
        "be_nocp": [
            (f_struc_mol, 1.0),
            (cluster_mol, -1.0),
            (smol_mol, -1.0),
        ],
        "ie": [(f_struc_mol, 1.0), (j7, -1.0), (j6, -1.0)],
        "ie_nocp": [(f_struc_mol, 1.0), (j5, -1.0), (j4, -1.0)],
        "de": [(cluster_mol, -1.0), (smol_mol, -1.0), (j4, 1.0), (j5, 1.0)],
    }

    return be_stoic
