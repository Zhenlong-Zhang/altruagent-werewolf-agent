"""One sampled focal learner at a pre-observation hashed seat; fixed opponent roster."""
import os
from pathlib import Path

from .hybrid_entry import HybridTable
from .selector import Selector
from .variants import LLMPolicy


def checked_configuration():
    backend = os.environ.get('WEREWOLF_MODEL_PATH') or os.environ.get('WEREWOLF_CPU_MODEL')
    if not backend or not (Path(backend)/'config.json').is_file():
        raise ValueError('Collection/evaluation requires a staged Hugging Face backend')
    validator = LLMPolicy('hybrid')
    validator.client.close()
    opponents = os.environ.get('WEREWOLF_HYBRID_OPPONENTS','rules,rules,rules,rules,rules,rules').split(',')
    if len(opponents)!=6 or any(item not in {'rules','direct','belief'} for item in opponents):
        raise ValueError('Provide six fixed opponent slots using rules, direct or belief')
    if os.environ.get('WEREWOLF_HYBRID_MANIFEST') or os.environ.get('WEREWOLF_HYBRID_COMPARE_PROPOSERS'):
        raise ValueError('Do not combine cohort/checkpoint routing with another assignment scheme')
    return opponents


def create_agent():
    mode=os.environ.get('WEREWOLF_HYBRID_MODE','hybrid_collect')
    if mode!='hybrid_collect':
        raise ValueError('Collection requires hybrid_collect sampling')
    weights=os.environ.get('WEREWOLF_HYBRID_WEIGHTS')
    if not weights:
        raise ValueError('Collection requires an explicit starting checkpoint')
    Selector(mode,weights)
    opponents=checked_configuration()
    proposer=os.environ.get('WEREWOLF_PROPOSER','grounded')
    # The exact competition factory reads this mode inside the focal worker.
    os.environ['WEREWOLF_HYBRID_MODE']=mode
    os.environ['WEREWOLF_PROPOSER']=proposer
    return HybridTable(0,mode,weights,{},opponents,(proposer,),
                       os.environ.get('WEREWOLF_COLLECTION_SEED','r03-onpolicy-v1'),True,
                       assignment_version='wolf-collect-v1',assignment_event='collection_assignment')
