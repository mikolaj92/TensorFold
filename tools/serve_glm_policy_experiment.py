"""Isolated full-model experiment; override EXL3 auto policy before CLI startup.

Set GLM_EXPERIMENT_POLICY identically on both ranks; pass normal serve arguments.
This is not a new production default or an HTTP policy option.
"""
import os

from tensorfold.families.glm5_next.cuda import engine
from tensorfold.cli import main

policy = os.environ["GLM_EXPERIMENT_POLICY"]
engine.encode_policy(policy)  # validate before allocating/loading the model
engine.EXL3_AUTO = policy
print(f"[experiment] EXL3 auto resolves to {policy}", flush=True)
raise SystemExit(main())
