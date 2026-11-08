NAB_ENSEMBLE = {"members": ("matrix_profile_raw", "zscore"), "w_factor": 2.0}


def nab_variant(detector: str) -> str:
    if detector == "ensemble":
        return f"w{NAB_ENSEMBLE['w_factor']:g}|{'+'.join(sorted(NAB_ENSEMBLE['members']))}"
    return "w1"

SELECTED_STAGE = {"nab": "C_context_length", "skab": "A_previous"}
SKAB_VARIANT = "local-w20"
