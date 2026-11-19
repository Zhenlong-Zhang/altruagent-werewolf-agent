"""Measure full-context legal-slate latency inside an existing Slurm allocation."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

from werewolf_agent.hybrid import HybridPolicy, Option


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    messages = [dict(sender=i % 5, content=(
        'I suspect Player1, but I have no verified check. Please compare the previous vote with the public claims. '
        'An accusation alone does not prove a role. We should discuss a common target and explain uncertainty. '
        'Player2 should respond.')[:240]) for i in range(10)]
    view = dict(seat=6, role='villager', phase='day_vote', day=2, alive=list(range(7)),
                private_knowledge={6:'villager'}, prior_ballots=[{'0':1,'1':2,'2':1,'3':2,'4':1,'5':2,'6':0}],
                revealed_deaths=[], messages=messages)
    votes = [Option(str(i),f'Vote to eliminate Player{i}',i,0.) for i in range(6)]
    wolf = dict(view, role='wolf', private_knowledge={6:'wolf',5:'wolf'})
    rows = []
    for balance in ('none','cyclic'):
        os.environ['WEREWOLF_PROPOSER']='grounded'
        os.environ['WEREWOLF_GROUNDED_ORDER_BALANCE']=balance
        policy=HybridPolicy('llm_select')
        cases=[('vote_six',view,votes,False),
               ('vote_seven_supported_bound',view,votes+[Option('7','Abstain',None,0.)],False),
               ('actual_wolf_speech_five',wolf,policy._speech(wolf,{}),True)]
        try:
            for label,current,options,speech in cases:
                started=time.monotonic()
                _,ranked,error,_=policy._propose(current,options,speech)
                rows.append(dict(balance=balance,scenario=label,offered=len(options),accepted=bool(ranked),
                                 ranked=ranked,error=error,elapsed_seconds=time.monotonic()-started,
                                 configuration=policy._model.proposal_configuration(),
                                 diagnostics=policy._model.proposal_diagnostics))
                args.out.parent.mkdir(parents=True,exist_ok=True)
                args.out.write_text(json.dumps(dict(rows=rows,interpretation='Latency diagnostic; no win-rate inference.'),indent=2)+'\n')
                print(json.dumps({key:rows[-1][key] for key in ['balance','scenario','offered','accepted','error','elapsed_seconds']}),flush=True)
        finally:
            if policy._model:
                policy._model.client.close()
        del cases,policy
        gc.collect()


if __name__=='__main__':
    main()
