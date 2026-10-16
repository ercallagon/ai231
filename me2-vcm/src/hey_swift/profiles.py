"""Public profile aliases; historical names and artifact paths stay stable."""
ALIASES = {"champion": "selected", "baseline": "run043", "selected": "selected", "run043": "run043"}
PUBLIC = {"selected": "champion", "run043": "baseline"}
DISPLAY = {"selected": "Champion (6s BC8-TCN-ASP)", "run043": "Baseline (2s BC8)"}


def resolve_profile(value):
    try:
        return ALIASES[value]
    except KeyError:
        raise ValueError("Choose champion or baseline (legacy: selected or run043)") from None
