import sys
import lib
import os
import torch
import math
import datetime

sys.path.insert(0, os.path.dirname(__file__)+"/../..")

model_map = {
    "fit_parallel_branches_ababab_many_ndr": "NDR",
    "fit_parallel_branches_ababab_many_rnn": "Bi-LSTM",
    "fit_parallel_branches_ababab_many_trafo": "Trafo",
}

variant_name_map = {
    "parallel_branch": "R",
    "alternate": "A",
}

# KEY = "validation/test/accuracy/test_group_alternate"

gall = {}

for sw, name in model_map.items():
    runs = lib.get_runs([sw])
    lg = lib.common.group(runs, ['fit.variant'])
    gall.update({f"{name}/{k}": v + gall.get(f"{name}/{k}", []) for k, v in lg.items()})


os.makedirs("tmp", exist_ok=True)

nparams = {}
iters = {}
for k, v in gall.items():
    run = v[0]
    for f in run.files(per_page=10000):
        if not f.name.startswith("checkpoint/") or "/model-" not in f.name:
            continue
        target = f"tmp/{f.id}"
        os.makedirs(target, exist_ok=True)
        name = f.download(root=target, replace=True)
        ckpt = torch.load(name.name)

        assert k not in nparams
        nparams[k] = sum(v.nelement() for v in ckpt["model"].values())
        iters[k] = run.summary["iteration"]


runtime = {k: sum(run.summary["_wandb"]["runtime"] for run in v) / len(v) for k, v in gall.items()}

for k in gall.keys():
    model, variant = k.split("/")
    variant = variant_name_map[variant[12:]]

    if variant == "A":
        # it does not depend on the variant
        continue

    print(f"{model} & {math.ceil(nparams[k]/1000)}k & {iters[k]//1000}k & {str(datetime.timedelta(seconds=runtime[k])).split('.')[0]} \\\\")
# print(nparams)