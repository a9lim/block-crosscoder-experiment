"""Named-block registry for the 4b pilot dictionaries (figure annotations).

Compiled from the pilot findings (zoo probe, geometry pass, identity
decodes — docs/findings-phase096-pilot4b.md). Soft labels: top-1 capture
plus context decodes, not causal claims. Clique lists from the J>0.9
co-activation components on the 1M-token eval split.

Renorm store site-RMS scalars ride here too (site order [9..30]): the
renorm arm trains on scalar-rescaled whitened acts, so any figure
comparing its decoder output against the raw whitened stream divides
site s by SITE_RENORM_SCALARS[s].
"""

from __future__ import annotations

import numpy as np

SITE_RENORM_SCALARS = np.array([
    4.9814, 5.3197, 4.9289, 3.6428, 2.6511, 2.2413, 2.0583, 1.8617,
])

NAMES = {
    "primary": {
        2146: "cardinal line two–nineteen",
        382: "ordinal segment 3rd–12th",
        1270: "spring (swallowed Mar–May)",
        2982: "weekday (capture, no order)",
        127: "compass N/W/E",
        349: "South/Latin geo prefix",
        636: "quantity digits",
        1018: "'197x' digit-7",
        1623: "astonishment plane",
    },
    "renorm": {
        595: "month ring",
        862: "weekday ring",
        3194: "cardinal 16/20",
        1393: "ordinal-suffix 'th'",
        3227: "late-teens ordinals",
        1808: "spelled round decades",
        1018: "'197x' digit-7",
        3234: "'196x' digit-6",
        2820: "'15xx' years",
        1609: "durations 30/45/90",
        2407: "generic '3'",
        1255: "ISBN digit '5'",
        2446: "sentence-initial 'One'",
        1512: "winter habitat",
        2295: "'Central' region head",
        510: "Latin",
        2324: "duration",
        2987: "magnitude",
        1219: "dollar digits",
        636: "quantity digits",
    },
}

CLIQUES = {
    "primary": [80, 242, 257, 355, 651, 734, 1103, 1297, 1694, 1825,
                2338, 2545, 2608, 2917, 2927, 3492, 3533, 3988],
    "renorm": [416, 552, 819, 1825, 1987],
}
