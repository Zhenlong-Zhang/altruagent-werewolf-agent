"""Compare complete fixed policies, including guard, mode and selector changes."""
import hashlib
import json
import os
from pathlib import Path

from .collection_entry import checked_configuration
from .competition_entry import create_agent as create_competitor
from .hybrid_entry import HybridTable
from .policy import WerewolfPolicy
from .selector import Selector
from .variants import LLMPolicy


class PolicyTable(HybridTable):
    def __init__(self, profiles, opponents, seed):
        super().__init__(0,'hybrid_greedy',None,{},opponents)
        self.profiles=profiles
        self.profile_seed=seed

    def _policy(self,context):
        if self.policy is not None:
            return self.policy
        if context.tournament_id is not None:
            self.policy=WerewolfPolicy()
            return self.policy
        digest=hashlib.sha256(f'wolf-policy-v1:{self.profile_seed}:{context.session_id}'.encode()).digest()
        focal=int.from_bytes(digest[:8],'big')%7
        profile=self.profiles[int.from_bytes(digest[8:16],'big')%len(self.profiles)]
        if context.seat_position==focal:
            os.environ.update(WEREWOLF_PROPOSER='ids',WEREWOLF_EVIDENCE_GUARD=profile['guard'],
                              WEREWOLF_HYBRID_MODE=profile['mode'],WEREWOLF_HYBRID_WEIGHTS=profile['weights'])
            self.policy=create_competitor()
            self.policy.opponent_pool=list(self.opponents)
            self.policy._ensure_logger(context)._write(dict(
                event='policy_assignment',seat=focal,mode=profile['mode'],arm=profile['name'],
                proposer='ids',evidence_guard=profile['guard'],assignment_version='wolf-policy-v1',
                assignment_seed=self.profile_seed,checkpoint_id=self.policy.selector.checkpoint_id,
                opponent_pool=list(self.opponents),competition_factory=True))
        else:
            positions=[seat for seat in range(7) if seat!=focal]
            opponent=self.opponents[positions.index(context.seat_position)]
            self.policy=WerewolfPolicy() if opponent=='rules' else LLMPolicy(opponent)
            if opponent!='rules':
                self.policy._strategy_version=f'pool-{opponent}:{self.policy.model}'
        return self.policy


def create_agent():
    profiles=json.loads(os.environ.get('WEREWOLF_POLICY_PROFILES','[]'))
    if not isinstance(profiles,list) or not 2<=len(profiles)<=4:
        raise ValueError('Provide two to four predeclared complete policy profiles')
    names=[]
    for profile in profiles:
        if not isinstance(profile,dict) or set(profile)!={'name','mode','guard','weights'}:
            raise ValueError('Each profile requires name, mode, guard and weights')
        if (not isinstance(profile['name'],str) or not profile['name'] or profile['name'] in names
                or profile['name'].strip()!=profile['name'] or ',' in profile['name']):
            raise ValueError('Profile names must be distinct nonempty strings')
        names.append(profile['name'])
        if profile['mode'] not in {'hybrid_greedy','hybrid_collect'}:
            raise ValueError('Profile mode must be greedy or sampled')
        if profile['guard'] not in {'none','verified-innocent-v1'}:
            raise ValueError('Invalid profile evidence guard')
        if (not isinstance(profile['weights'],str) or not profile['weights'].strip()
                or not Path(profile['weights']).is_file()):
            raise ValueError('Each policy requires an explicit existing checkpoint file')
        Selector(profile['mode'],profile['weights'])
    opponents=checked_configuration()
    return PolicyTable(profiles,opponents,os.environ.get('WEREWOLF_POLICY_COMPARE_SEED','r03-policy-v1'))
