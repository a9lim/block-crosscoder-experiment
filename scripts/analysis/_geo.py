"""Country ground-truth coordinates for the atlas-tranche geo tests.

Capital-city lat/lon (degrees, rounded — the geo statistic is a linear
decode against standardized coordinates, so ~0.1° precision is far below
the noise floor of class means). Order matches labels.COUNTRIES.
Continent tags drive figure coloring and the clustering read; Russia is
tagged Europe (Moscow-side centroid of usage), Turkey Asia.
"""

from __future__ import annotations

# name -> (lat, lon, continent)
COUNTRY_GEO: dict[str, tuple[float, float, str]] = {
    "England": (51.5, -0.1, "Europe"),
    "France": (48.9, 2.4, "Europe"),
    "Germany": (52.5, 13.4, "Europe"),
    "Italy": (41.9, 12.5, "Europe"),
    "Spain": (40.4, -3.7, "Europe"),
    "Portugal": (38.7, -9.1, "Europe"),
    "Ireland": (53.3, -6.3, "Europe"),
    "Scotland": (56.0, -3.2, "Europe"),
    "Russia": (55.8, 37.6, "Europe"),
    "China": (39.9, 116.4, "Asia"),
    "Japan": (35.7, 139.7, "Asia"),
    "India": (28.6, 77.2, "Asia"),
    "Australia": (-35.3, 149.1, "Oceania"),
    "Canada": (45.4, -75.7, "Americas"),
    "Mexico": (19.4, -99.1, "Americas"),
    "Brazil": (-15.8, -47.9, "Americas"),
    "Egypt": (30.0, 31.2, "Africa"),
    "Israel": (31.8, 35.2, "Asia"),
    "Iran": (35.7, 51.4, "Asia"),
    "Iraq": (33.3, 44.4, "Asia"),
    "Turkey": (39.9, 32.9, "Asia"),
    "Greece": (38.0, 23.7, "Europe"),
    "Poland": (52.2, 21.0, "Europe"),
    "Sweden": (59.3, 18.1, "Europe"),
    "Norway": (59.9, 10.7, "Europe"),
    "Denmark": (55.7, 12.6, "Europe"),
    "Finland": (60.2, 24.9, "Europe"),
    "Austria": (48.2, 16.4, "Europe"),
    "Switzerland": (46.9, 7.4, "Europe"),
    "Netherlands": (52.4, 4.9, "Europe"),
    "Belgium": (50.8, 4.4, "Europe"),
    "Argentina": (-34.6, -58.4, "Americas"),
    "Chile": (-33.5, -70.7, "Americas"),
    "Peru": (-12.0, -77.0, "Americas"),
    "Cuba": (23.1, -82.4, "Americas"),
    "Kenya": (-1.3, 36.8, "Africa"),
    "Nigeria": (9.1, 7.5, "Africa"),
    "Ethiopia": (9.0, 38.7, "Africa"),
    "Vietnam": (21.0, 105.8, "Asia"),
    "Korea": (37.6, 127.0, "Asia"),
    "Thailand": (13.8, 100.5, "Asia"),
    "Indonesia": (-6.2, 106.8, "Asia"),
    "Pakistan": (33.7, 73.1, "Asia"),
    "Afghanistan": (34.5, 69.2, "Asia"),
    "Ukraine": (50.5, 30.5, "Europe"),
    "Hungary": (47.5, 19.0, "Europe"),
    "Romania": (44.4, 26.1, "Europe"),
    "Iceland": (64.1, -21.9, "Europe"),
}

CONTINENT_ORDER = ["Europe", "Asia", "Africa", "Americas", "Oceania"]
