import json
import pickle
from pathlib import Path

PID_TO_POS = Path("/Users/secilsen/PhD/protein-go-semantic-align/src/data/training_ready/proteins/pid_to_positives.json")               # mevcut mapping
GO_BASIC_OBO = Path("/Users/secilsen/PhD/protein-go-semantic-align/src/data/raw/go_basic_obo_terms_v2.pkl")         # ontology
OUT_PID2POS = Path("../data/training_ready/proteins/pid_to_positives_canonical.json")    # yeni canonical mapping

# 1) Yükle: protein -> [GO ids]
with PID_TO_POS.open("r") as f:
    pid2pos = json.load(f)  # {protein_id: [go_int_ids]}

# 2) OBO tarafını yükle: go_id -> {...}
with GO_BASIC_OBO.open("rb") as f:
    obo_terms = pickle.load(f)

# OBO id seti (keyler integer ise direk, string ise senin mappingine göre)
obo_ids_str = set(map(str, obo_terms.keys()))
obo_ids = set(int(str(x).replace("GO:", "")) for x in obo_ids_str)

# 3) GOA tarafındaki tüm GO id'leri topla
goa_ids = set()
for gids in pid2pos.values():
    for g in gids:
        goa_ids.add(int(g))

print("GO ids in pid_to_positives:", len(goa_ids))
print("GO ids in OBO:", len(obo_ids))

canonical_ids = goa_ids & obo_ids
missing_in_obo = sorted(goa_ids - obo_ids)

print("Canonical GO ids:", len(canonical_ids))
print("Missing in OBO (first few):", len(missing_in_obo), missing_in_obo[:20])

# 4) pid_to_positives'i filtrele
pid2pos_canonical = {}
dropped_proteins = 0
dropped_labels = 0

for pid, gids in pid2pos.items():
    new_gids = [int(g) for g in gids if int(g) in canonical_ids]
    if not new_gids:
        dropped_proteins += 1
        continue
    dropped_labels += len(gids) - len(new_gids)
    pid2pos_canonical[pid] = new_gids

print("Dropped proteins with no remaining GO:", dropped_proteins)
print("Dropped label assignments:", dropped_labels)
print("Final proteins:", len(pid2pos_canonical))

# 5) Kaydet
with OUT_PID2POS.open("w") as f:
    json.dump(pid2pos_canonical, f)

print("Saved canonical pid_to_positives ->", OUT_PID2POS)
