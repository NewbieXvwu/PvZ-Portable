"""task192: engine facts for the opt-in card-attribute CNN, computed once in memory.

No card lists or tactical categories: numeric definitions, pure Plant predicates,
PlantSubClass, and required upgrade identities come from this checkout's C++.
The base identity records a prerequisite, not a complete CanPlantAt decision
(e.g. cob cannon also needs the engine's valid two-cell spot).
"""
from functools import lru_cache
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pvz_constants as constants

PREDICATES = ('IsNocturnal', 'IsFungus', 'IsAquatic', 'IsFlying', 'IsUpgrade', 'IsSpiky')


@lru_cache(maxsize=None)
def card_attribute_matrix(plant_types=54):
    """Return (float32[type, attribute], column names), without any disk cache.

    Numerical columns are divided by their source-table maximum; booleans and
    subclass/base identities are one-hot. Unused/clipped IDs retain zero facts.
    """
    source = constants._read(constants.PLANT_CPP)
    enums = constants._parse_enum(constants._read(constants.CONST_ENUMS), 'SeedType')
    definitions = constants.plant_defs()
    predicates = {}
    for name in PREDICATES:
        start, end = constants._function_span(source, 'bool Plant::' + name + '(')
        predicates[name] = {enums[seed] for seed in re.findall(r'SeedType::(SEED_\w+)', source[start:end])}
    start, end = constants._function_span(source, 'bool Plant::IsUpgradableTo(')
    bases = {enums[upgrade]: enums[base] for upgrade, base in re.findall(
        r'theUpgradedType\s*==\s*SeedType::(SEED_\w+)\s*&&\s*mSeedType\s*==\s*SeedType::(SEED_\w+)',
        source[start:end])}
    subclasses = constants._parse_enum(constants._read(ROOT / 'src/Lawn/Plant.h'), 'PlantSubClass')
    subclass_rows = {enums[seed]: subclass for seed, subclass in re.findall(
        r'\.mSeedType\s*=\s*SeedType::(\w+).*?\.mSubClass\s*=\s*PlantSubClass::(\w+)',
        constants._array_body(source, 'gPlantDefs'))}
    columns = ('cost/max_cost', 'recharge/max_recharge', 'launch/max_launch', *PREDICATES,
               *('subclass/' + name for name in subclasses),
               *(f'upgrade_base/{kind}' for kind in range(plant_types)))
    matrix = np.zeros((plant_types, len(columns)), dtype=np.float32)
    maxima = [max(d[key] for d in definitions) for key in ('cost', 'recharge', 'launch')]
    for definition in definitions:
        kind = definition['type']
        if kind >= plant_types:
            continue
        matrix[kind, :3] = [definition[key] / max(1, scale)
                           for key, scale in zip(('cost', 'recharge', 'launch'), maxima)]
        for offset, name in enumerate(PREDICATES, 3):
            matrix[kind, offset] = kind in predicates[name]
        matrix[kind, 3 + len(PREDICATES) + list(subclasses).index(subclass_rows[kind])] = 1
        if kind in bases:
            matrix[kind, 3 + len(PREDICATES) + len(subclasses) + bases[kind]] = 1
    matrix.flags.writeable = False
    return matrix, columns
