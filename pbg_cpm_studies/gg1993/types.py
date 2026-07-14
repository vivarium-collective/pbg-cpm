"""Cell-type constants for the Glazier & Graner (1993) reproduction.

Matching the repo's existing demo convention (`demos/cell_sorting_2d.json`):
medium is always owner/type 0; dark (low surface energy, tau=d) is type 1;
light (high surface energy, tau=l) is type 2.
"""

MEDIUM = 0
DARK = 1   # tau = d  (dark, low surface energy)
LIGHT = 2  # tau = l  (light, high surface energy)

TYPE_NAME = {MEDIUM: "medium", DARK: "dark", LIGHT: "light"}
