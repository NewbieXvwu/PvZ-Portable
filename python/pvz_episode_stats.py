"""Read episode planting counts without confusing card slots with plant identities.

New records use planted_types and planted_slots. Historical planted is always slots.
Legacy conveyor slot totals cannot be translated through an initial deck: return None.
"""
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
import sys


def _counts(value, name):
    if isinstance(value, Mapping):
        items=value.items()
    elif isinstance(value, Sequence) and not isinstance(value,(str,bytes)):
        items=enumerate(value)
    else:
        raise ValueError(name+' must be a count object or array')
    result=Counter()
    for key,count in items:
        if isinstance(key,bool) or isinstance(count,bool) or not isinstance(count,Integral) or count<0:
            raise ValueError(name+' has invalid indices or counts')
        try:index=int(key)
        except (ValueError,TypeError):raise ValueError(name+' has an invalid index') from None
        if index<0 or str(index)!=str(key):raise ValueError(name+' has an invalid index')
        if count:result[str(index)]+=int(count)
    return dict(sorted(result.items(),key=lambda pair:int(pair[0])))


def planted_slots(game):
    """Return normalized slot totals, or None if the record has no slot telemetry."""
    key='planted_slots' if 'planted_slots' in game else 'planted' if 'planted' in game else None
    return _counts(game[key],key) if key is not None else None


def planted_by_type(game, deck, *, level=None, conveyor_start=None):
    """Return {str(plant_type): count}, or None when historical identity is unknowable.

    deck may be the task dictionary (preferred) or the actual per-game deck sequence.
    Callers with a sequence must pass the known level/conveyor mode for old records.
    The initial task27 planted_by_type object is a supported intermediate schema;
    an array under that name is NOT accepted as identity telemetry (old CUDA used slots).
    """
    if 'planted_types' in game:
        if not isinstance(game['planted_types'],Mapping):raise ValueError('planted_types must be a JSON object')
        return _counts(game['planted_types'],'planted_types')
    if isinstance(game.get('planted_by_type'),Mapping):
        return _counts(game['planted_by_type'],'planted_by_type')
    if isinstance(deck,Mapping):
        level=deck.get('level',level);conveyor_start=deck.get('conveyor_start',conveyor_start)
        deck=deck.get('deck')
    level=game.get('level',level)
    conveyor_start=game.get('conveyor_start',conveyor_start)
    if conveyor_start:
        return None
    if level is not None:
        # Use the existing engine-derived classification, including special conveyor levels.
        scripts=str(Path(__file__).resolve().parents[1]/'scripts')
        if scripts not in sys.path:sys.path.append(scripts)
        from pvz_tasks import conveyor_belt_level
        if conveyor_belt_level(int(level)):return None
    slots=planted_slots(game)
    if slots is None or deck is None:return None
    totals=Counter()
    for slot,count in slots.items():
        index=int(slot)
        if index>=len(deck) or deck[index]<0:return None
        totals[str(int(deck[index]))]+=count
    return dict(sorted(totals.items(),key=lambda pair:int(pair[0])))


def plant_type_before_action(observation, packet):
    """Read a live observation's slot before planting can consume/move a conveyor card."""
    for card in observation['packets']:
        if isinstance(card,Mapping):
            if card['index']!=packet:continue
            return card.get('imitater_type',-1) if card['type']==48 else card['type']
        if card[0]==packet:return card[1]
    return None
