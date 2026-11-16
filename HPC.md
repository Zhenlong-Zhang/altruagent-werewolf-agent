# Hoffman2 workspace

This project uses the production Slurm cluster (`hoffman2.oarc.ucla.edu`). The project checkout is separate from existing class/research folders and jobs.

## Current cluster snapshot

- Account: `pi_ra_lab`, routed to campus; no highp access.
- Python module available for this project: `python/3.12.13` (the default `python3` is 3.9 and is too old for the starter).
- At initial setup, the only existing job was `1780018` (`260e-part2`, pending priority) in `/u/home/z/zzl/260e-part2`. This project does not edit or cancel it.

## Environment

Load the current Python module, create a project-local virtual environment, install the starter and this policy package. Keep credentials in a private ignored `.env` inside the starter checkout only; never in this repo or job scripts.

```bash
module load python/3.12.13
python --version
python -m venv ~/altruagent-starter/.venv
source ~/altruagent-starter/.venv/bin/activate
cd ~/altruagent-starter
pip install -e ".[dev]"
pip install -e "$HOME/altruagent-werewolf-agent"
```

For any future self-play policy search or training, submit a Slurm batch job with `-A pi_ra_lab`, explicit measured memory and walltime, and only the CPU resources the workload uses. Do not run simulations or training on a login node. No compute job is submitted yet: the next useful computation depends on having platform Testing trajectories or a validated local simulator.
