"""Prospective checkpoint comparison with a fixed proposer, mode and opponent roster."""
import json
import os

from .collection_entry import checked_configuration
from .hybrid_entry import HybridTable
from .selector import Selector


def create_agent():
    mode=os.environ.get('WEREWOLF_HYBRID_MODE','hybrid_greedy')
    if mode not in {'hybrid_greedy','hybrid_collect'}:
        raise ValueError('Select the fixed greedy or sampled evaluation mode')
    choices=json.loads(os.environ.get('WEREWOLF_SELECTOR_CHECKPOINTS','{}'))
    if not isinstance(choices,dict) or list(choices)!=['incumbent','adapted']:
        raise ValueError('Provide incumbent then adapted checkpoint paths')
    checked=[]
    for arm,path in choices.items():
        selector=Selector(mode,path)
        checked.append((arm,path))
    if Selector(mode,checked[0][1]).checkpoint_id==Selector(mode,checked[1][1]).checkpoint_id:
        raise ValueError('Checkpoint comparison requires distinct policy weights')
    opponents=checked_configuration()
    os.environ['WEREWOLF_HYBRID_MODE']=mode
    return HybridTable(0,mode,None,{},opponents,(),
                       os.environ.get('WEREWOLF_SELECTOR_COMPARE_SEED','r03-checkpoints-v1'),True,
                       assignment_version='wolf-selector-v1',checkpoint_choices=tuple(checked))
