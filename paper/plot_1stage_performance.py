import lib
import os

os.makedirs("out", exist_ok=True)

variant_name_map = {
    "parallel_branch": "R",
    "alternate": "A",
}

model_map = {
    "fit_parallel_branches_ababab_many_ndr": "NDR",
    "fit_parallel_branches_ababab_many_trafo": "Trafo",
    "fit_parallel_branches_ababab_many_rnn": "Bi-LSTM",
}

gall = {}

for sw, name in model_map.items():
    runs = lib.get_runs([sw])
    lg = lib.common.group(runs, ['fit.variant'])
    gall.update({f"{name}/{k}": v + gall.get(f"{name}/{k}", []) for k, v in lg.items()})

g = {}
for k, v in gall.items():
    filtered = [r for r in v if r.summary["validation/iid/accuracy/total"] > 0.5]
    if len(filtered) < len(v):
        print(f"WARNING: {100 - len(filtered)/len(v)*100:.2f}% of runs did not converge for {k}!")

    assert len(filtered) >=  25
    g[k] = filtered[:25]

s = lib.common.calc_stat(g, lambda k: k in {"validation/test/accuracy/test_group_alternate", "validation/test/accuracy/test_self_loop", "validation/iid/accuracy/total"})

s = {k: {k2: v2.get() for k2, v2 in v.items()} for k, v in s.items()}

# print(s)

def get_ood_key(name):
    if "variant_parallel_branch" in name:
        return 'validation/test/accuracy/test_group_alternate' 
    else:
        return "validation/test/accuracy/test_self_loop"

iid_acc = {k: v['validation/iid/accuracy/total'] for k, v in s.items()}
ood_acc = {k: v[get_ood_key(k)] for k, v in s.items()}


ood_vals = {k: [r.summary[get_ood_key(k)] for r in v] for k, v in g.items()}
ood_props = {k: sum([1 if a > 0.95 else 0 for a in v])/len(v) for k, v in ood_vals.items()}


decoded_map = {}
all_models = set()
all_datasets = set()

for k in sorted(iid_acc.keys()):
    v = [p for p in k.split("/") if p.startswith("fit.variant_")][0][12:]
    name = variant_name_map[v]
    model = k.split("/")[0]

    decoded_map[name, model] = k
    all_models.add(model)
    all_datasets.add(name)


print("Model & Dataset & IID Accuracy & OOD Accuracy \\\\")
print("\\midrule")
for mi, m in enumerate(sorted(all_models)):
    if mi > 0:
        print("\\midrule")
    for dsi, ds in enumerate(sorted(all_datasets)):
        k = decoded_map[ds, m]
        mm = "" if dsi > 0 else f"\\multirow{{{len(all_datasets)}}}{{*}}{{{m}}}"
        print(f"{mm} & {ds} & ${iid_acc[k].mean:.2f} \\pm {iid_acc[k].std:.2f}$ & ${ood_acc[k].mean:.2f} \\pm {ood_acc[k].std:.2f}$ \\\\")


print("Proportion of successful seeds:")
for mi, m in enumerate(sorted(all_models)):
    for dsi, ds in enumerate(sorted(all_datasets)):
        print(f"   {m}: {ood_props[decoded_map[ds, m]]:.2f}")
