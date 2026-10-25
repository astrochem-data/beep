def gga():
    return [
        # Baked-in dispersion
        'B97-D',
        # D4
        'PBE-D4',
        'BLYP-D4',
        'REVPBE-D4',
        'RPBE-D4',
        'PW91-D4',
        # D3BJ only
        'BP86-D3BJ',
        'N12-D3BJ',
    ]

def meta_gga():
    return [
        # Baked-in dispersion (Minnesota / VV10)
        'B97M-V',
        'M06-L',
        'MN15-L',
        'REVM06-L',
        # D4
        'SCAN-D4',
        'R2SCAN-D4',
        'TPSS-D4',
        'REVTPSS-D4',
        # D3BJ only
        'MN12-L-D3BJ',
        'M11-L-D3BJ',
    ]

def meta_hybrid_gga():
    return [
        # Baked-in dispersion (Minnesota / VV10)
        'M05',
        'M05-2X',
        'M06',
        'M06-2X',
        'M06-HF',
        'M08-HX',
        'M08-SO',
        'M11',
        'MN12-SX',
        'MN15',
        'WB97M-V',
        # D4
        'MPW1B95-D4',
        'MPWB1K-D4',
        'PW6B95-D4',
        'TPSSH-D4',
        'REVTPSSH-D4',
        'PWPB95-D4',
        'R2SCAN0-D4',
        # D3BJ only
        'BMK-D3BJ',
        'PWB6K-D3BJ',
        'PTPSS-D3BJ',
        'WB97M-D3BJ',
    ]

def lrc():
    return [
        # Baked-in dispersion (VV10 / -D)
        'WB97X-V',
        'WB97X-D',
        'LC-VV10',
        # D4
        'HSE06-D4',
        'HSE03-D4',
        'WB97X-D4',
        'WB97-D4',
        'REVPBE0-D4',
        # D3BJ only
        'CAM-B3LYP-D3BJ',
        'N12-SX-D3BJ',
        'WPBE-D3BJ',
    ]

def hybrid_gga():
    return [
        # Baked-in dispersion (NL / composite)
        'B3LYP-NL',
        'PBE0-NL',
        'B3PW91-NL',
        'REVPBE0-NL',
        'HF-D3BJ',
        # D4
        'B3LYP-D4',
        'PBE0-D4',
        'B1LYP-D4',
        'O3LYP-D4',
        'X3LYP-D4',
        # D3BJ only
        'B3P86-D3BJ',
        'B3PW91-D3BJ',
        'SOGGA11-X-D3BJ',
        'B97-1-D3BJ',
        'B97-2-D3BJ',
    ]

def double_hybrid():
    return [
        'B2GPPLYP',
        'B2GPPLYP-D3BJ2B',
        'B2GPPLYP-D3BJATM',
        'B2GPPLYP-D3ZERO2B',
        'B2GPPLYP-D3ZEROATM',
        'B2GPPLYP-NL',
        'B2PLYP',
        'B2PLYP-D3BJ2B',
        'B2PLYP-D3BJATM',
        'B2PLYP-D3MBJ2B',
        'B2PLYP-D3MBJATM',
        'B2PLYP-D3MZERO2B',
        'B2PLYP-D3MZEROATM',
        'B2PLYP-D3ZERO2B',
        'B2PLYP-D3ZEROATM',
        'B2PLYP-NL',
        'CORE-DSD-BLYP',
        'CORE-DSD-BLYP-D3BJ2B',
        'CORE-DSD-BLYP-D3BJATM',
        'DSD-BLYP',
        'DSD-BLYP-D3BJ',
        'DSD-BLYP-D3BJ2B',
        'DSD-BLYP-D3BJATM',
        'DSD-BLYP-D3ZERO2B',
        'DSD-BLYP-D3ZEROATM',
        'DSD-BLYP-NL',
        'DSD-PBEB95',
        'DSD-PBEB95-D3BJ',
        'DSD-PBEB95-NL',
        'DSD-PBEP86',
        'DSD-PBEP86-D3BJ',
        'DSD-PBEP86-NL',
        'DSD-PBEPBE',
        'DSD-PBEPBE-D3BJ',
        'DSD-PBEPBE-NL',
        'MP2D',
        'MP2MP2',
        'PBE0-2',
        'PBE0-DH',
        'PBE0-DH-D3BJ2B',
        'PBE0-DH-D3BJATM',
        'PBE0-DH-D3ZERO2B',
        'PBE0-DH-D3ZEROATM',
        'PBE0-DH-NL',
        'PTPSS',
        'PTPSS-D3BJ2B',
        'PTPSS-D3BJATM',
        'PTPSS-D3ZERO2B',
        'PTPSS-D3ZEROATM',
        'PWPB95',
        'PWPB95-D3BJ2B',
        'PWPB95-D3BJATM',
        'PWPB95-D3ZERO2B',
        'PWPB95-D3ZEROATM',
        'PWPB95-NL',
    ]

def geom_hmgga_dz():
    return [
        "MPWB1K-D4_def2-svpd",
        "PBE0-D4_def2-svpd",
        "PBE0_def2-svpd",
        "B3LYP-D4_def2-svpd",
        "M06-2X_def2-svpd",
        "WB97X-D4_def2-svpd",
        "R2SCAN-D4_def2-svpd",
    ]

def geom_hmgga_tz():
    return [
        "M05-2X_def2-tzvp",
        "M05-2X_def2-tzvpd",
        "MPWB1K-D4_def2-tzvp",
        "MPWB1K-D4_def2-tzvpd",
        "B3LYP-D4_def2-tzvp",
        "B3LYP-D4_def2-tzvpd",
        "PBE0-D4_def2-tzvp",
        "PBE0-D4_def2-tzvpd",
        "PBE0_def2-tzvp",
        "PBE0_def2-tzvpd",
        "M06-2X_def2-tzvp",
        "M06-2X_def2-tzvpd",
        "WB97X-D4_def2-tzvp",
        "WB97X-D4_def2-tzvpd",
        "R2SCAN-D4_def2-tzvp",
        "R2SCAN-D4_def2-tzvpd",
    ]

def geom_gga_dz():
    return [
        "B97-D_def2-svpd",
        "BLYP-D4_def2-svpd",
        "PBE-D4_def2-svpd",
    ]

def geom_gga_tz():
    return [
        "PBE-D4_def2-tzvp",
        "PBE-D4_def2-tzvpd",
    ]

def geom_sqm_mb():
    return [
        "HF3C_MINIX",
        "PBEh3c_def2-msvp",
        "B97-3c_def2-mTZVP",
        "R2SCAN-3c_def2-mTZVPP",
        "WB97X-3c_vDZP",
    ]

def geom_nl():
    """OPTIONAL VV10 "-V" candidates for the geometry/force benchmarks.

    Excluded from the default geom_* groups because Psi4 has no analytic VV10
    gradient. These are routed to the ORCA driver with ``SCNL`` (see
    ``is_nl_method``), which does give a complete VV10 gradient. Enabled per
    workflow via the ``include_nl_functionals`` config flag.
    """
    return [
        "B97M-V_def2-tzvpd",
        "WB97M-V_def2-tzvpd",
        "WB97X-V_def2-tzvpd",
    ]


def non_local():
    """Functionals with non-local (NL / VV10) dispersion treatment.

    Cross-cuts the other categories: every entry here also lives in one
    of ``gga`` / ``meta_gga`` / ``hybrid_gga`` / ``meta_hybrid_gga`` /
    ``lrc``. The per-group log helpers walk every group dict, so a
    functional with NL dispersion appears both in its structural class
    table and in the Non-local table.
    """
    return [
        # VV10 (atom-pairwise integrand on the density)
        'B97M-V',         # also meta_gga
        'WB97M-V',        # also meta_hybrid_gga
        'WB97X-V',        # also lrc
        'LC-VV10',        # also lrc
        # -NL post-hoc non-local
        'B3LYP-NL',       # also hybrid_gga
        'PBE0-NL',        # also hybrid_gga
        'B3PW91-NL',      # also hybrid_gga
        'REVPBE0-NL',     # also hybrid_gga
    ]


# ---------------------------------------------------------------------------
# gCP applicability
#
# The geometric counterpoise correction (Kruse & Grimme, JCP 136, 154101
# (2012)) is parametrized for specific (SCF-type, basis) combinations only.
# For BEEP's energy_benchmark workflow we restrict gCP to the single
# (DFT, def2-tzvp) target, which keeps the workflow neat and lean and
# covers the production use case.
# ---------------------------------------------------------------------------

_THREE_C_METHODS = {
    "HF3C", "HF-3C",
    "PBEH3C", "PBEH-3C",
    "B97-3C", "B973C",
    "R2SCAN-3C", "R2SCAN3C",
    "WB97X-3C", "WB97X3C",
}


def is_3c_method(name: str) -> bool:
    """True if ``name`` is a -3c composite (gCP already baked in).

    Used to skip BSSE/CP-fragment submissions for these methods —
    counterpoise on top of an already-gCP-corrected SCF double-corrects.
    """
    if not name:
        return False
    return name.upper() in _THREE_C_METHODS


_NL_METHODS = {
    "B97M-V", "WB97M-V", "WB97X-V", "LC-VV10",
}


def is_nl_method(name: str) -> bool:
    """True if ``name`` is a VV10 "-V" / non-local functional.

    Such functionals need ORCA's self-consistent non-local (``SCNL``) for a
    complete analytic gradient; Psi4 has no analytic VV10 gradient. The
    nm_sampling workflow uses this to route these methods to the ORCA driver
    with ``SCNL`` injected.
    """
    if not name:
        return False
    return name.upper() in _NL_METHODS


def gcp_compatible_functionals():
    """Functionals that get an explicit gCP correction when the user
    enables ``gcp_correction`` in the energy_benchmark workflow.

    Concatenation of the standard semi-local lists, excluding:
      - -3c composites (gCP is already part of the method)
      - Double hybrids (gCP is not parametrized for the MP2-correlated
        regime; published parameters fit semi-local functionals only)
      - HF-D3BJ (gCP for HF needs the ``hf/def2-tzvp`` parameter set, a
        separate code path we don't support — bare HF + dispersion is
        a niche entry rather than a production binding-energy method)
    """
    excluded = {"HF-D3BJ"}
    pool = [*gga(), *meta_gga(), *hybrid_gga(), *meta_hybrid_gga(), *lrc()]
    return [m for m in pool if m.upper() not in excluded and not is_3c_method(m)]
