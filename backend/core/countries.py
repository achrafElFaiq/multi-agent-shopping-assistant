"""Country codes."""

COUNTRY_CODES = {"france": "FR", "belgium": "BE", "germany": "DE", "spain": "ES", "italy": "IT", "united kingdom": "GB"}


def country_code(country: str, default: str = "FR") -> str:
    """ISO code from a country name or code: "France" or "fr" -> "FR". `default` when unknown."""
    country = country.strip()
    if len(country) == 2:
        return country.upper()
    return COUNTRY_CODES.get(country.lower(), default)
