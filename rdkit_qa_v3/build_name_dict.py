"""One-time dev script: curated common-name list -> PubChem -> canonicalized SMILES,
baked into data/v3/name_to_smiles.json. Not run at train/eval time -- oracle.py loads
the static JSON offline (rdkit_qa_v3/name_dict.py), so training/eval never depend on
network access or PubChem uptime. Re-run manually only if the curated list changes."""
from __future__ import annotations
import json, time
from pathlib import Path

import requests
from rdkit import Chem

DATA = Path(__file__).resolve().parent.parent / "data" / "v3"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES,IsomericSMILES/JSON"

# common drugs, solvents, reagents -- names a chemist would actually type
NAMES = [
    "acetone", "aspirin", "ibuprofen", "caffeine", "acetaminophen", "paracetamol",
    "ethanol", "methanol", "isopropanol", "benzene", "toluene", "xylene",
    "acetic acid", "formic acid", "citric acid", "oxalic acid", "sulfuric acid",
    "hydrochloric acid", "sodium chloride", "glucose", "fructose", "sucrose",
    "cholesterol", "urea", "glycerol", "acetonitrile", "dichloromethane",
    "chloroform", "diethyl ether", "tetrahydrofuran", "dimethyl sulfoxide",
    "dimethylformamide", "pyridine", "phenol", "aniline", "naphthalene",
    "anthracene", "cyclohexane", "cyclohexanone", "hexane", "heptane", "octane",
    "propane", "butane", "ethylene glycol", "propylene glycol", "acetamide",
    "formaldehyde", "acetaldehyde", "benzaldehyde", "benzoic acid", "salicylic acid",
    "nicotine", "morphine", "codeine", "penicillin", "amoxicillin", "warfarin",
    "metformin", "atorvastatin", "simvastatin", "omeprazole", "diazepam",
    "amphetamine", "methamphetamine", "cocaine", "thc", "cbd", "melatonin",
    "serotonin", "dopamine", "adrenaline", "epinephrine", "testosterone",
    "estradiol", "progesterone", "insulin", "adenine", "guanine", "cytosine",
    "thymine", "uracil", "histidine", "glycine", "alanine", "leucine",
    "tryptophan", "phenylalanine", "tyrosine", "lysine", "arginine", "proline",
    "vitamin c", "ascorbic acid", "vitamin d", "retinol", "biotin", "folic acid",
    "riboflavin", "thiamine", "niacin", "pyridoxine", "tocopherol",
    "sodium bicarbonate", "sodium hydroxide", "potassium hydroxide", "ammonia",
    "hydrogen peroxide", "ozone", "carbon dioxide", "carbon monoxide", "methane",
    "ethylene", "propylene", "acetylene", "styrene", "vinyl chloride",
    "polyethylene", "phenylalanine", "menthol", "camphor", "vanillin", "eugenol",
    "limonene", "thymol", "carvone", "geraniol", "linalool", "citral",
    "quinine", "atropine", "strychnine", "capsaicin", "resveratrol", "curcumin",
    "quercetin", "catechin", "tannic acid", "lactic acid", "malic acid",
    "tartaric acid", "succinic acid", "adipic acid", "stearic acid", "oleic acid",
    "palmitic acid", "linoleic acid", "cholic acid", "retinoic acid",
    "chlorpromazine", "fluoxetine", "sertraline", "diphenhydramine",
    "loratadine", "cetirizine", "ranitidine", "famotidine", "hydrocortisone",
    "prednisone", "dexamethasone", "levothyroxine", "propranolol", "atenolol",
    # second batch, added to widen name diversity (fewer repeats needed per
    # name once chain volume goes up) rather than only repeating the batch above
    "ibuprofen sodium", "naproxen", "diclofenac", "meloxicam", "celecoxib",
    "clopidogrel", "amlodipine", "losartan", "valsartan", "lisinopril",
    "furosemide", "spironolactone", "digoxin", "heparin", "gabapentin",
    "pregabalin", "tramadol", "oxycodone", "fentanyl", "lidocaine",
    "bupivacaine", "ketamine", "propofol", "midazolam", "haloperidol",
    "risperidone", "olanzapine", "quetiapine", "lithium carbonate", "valproic acid",
    "carbamazepine", "phenytoin", "lamotrigine", "topiramate", "levetiracetam",
    "azithromycin", "erythromycin", "ciprofloxacin", "doxycycline", "vancomycin",
    "metronidazole", "clindamycin", "cephalexin", "fluconazole", "acyclovir",
    "oseltamivir", "ritonavir", "methotrexate", "cyclophosphamide", "doxorubicin",
    "cisplatin", "paclitaxel", "tamoxifen", "imatinib", "sildenafil",
    "tadalafil", "finasteride", "metoprolol", "carvedilol", "clonidine",
    "hydrochlorothiazide", "pantoprazole", "esomeprazole", "loperamide", "ondansetron",
    "diphenoxylate", "insulin glargine", "glipizide", "pioglitazone", "sitagliptin",
    "empagliflozin", "rosuvastatin", "ezetimibe", "niacinamide", "pyridoxal",
    "cobalamin", "calcitriol", "cholecalciferol", "alpha tocopherol", "phylloquinone",
    "beta carotene", "lycopene", "lutein", "zeaxanthin", "astaxanthin",
    "spermine", "spermidine", "putrescine", "histamine", "gaba",
    "glutamate", "aspartate", "cysteine", "methionine", "serine",
    "threonine", "asparagine", "glutamine", "valine", "isoleucine",
    "acetylsalicylic acid", "para-aminobenzoic acid", "sulfanilamide", "trimethoprim", "sulfamethoxazole",
    "chlorhexidine", "povidone iodine", "benzalkonium chloride", "sodium lauryl sulfate", "triclosan",
    "ethylenediaminetetraacetic acid", "sodium citrate", "sodium benzoate", "potassium sorbate", "monosodium glutamate",
    "aspartame", "sucralose", "saccharin", "xylitol", "sorbitol",
    "mannitol", "maltose", "lactose", "galactose", "ribose",
    "deoxyribose", "starch", "cellulose", "chitin", "collagen",
]


def fetch(name: str) -> str | None:
    try:
        r = requests.get(PUG.format(name=name), timeout=10)
        if r.status_code != 200:
            return None
        props = r.json()["PropertyTable"]["Properties"][0]
        raw = props.get("CanonicalSMILES") or props.get("IsomericSMILES") \
            or props.get("ConnectivitySMILES")
        if not raw:
            return None
        mol = Chem.MolFromSmiles(raw)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception as e:
        print(f"  FAILED {name}: {e}")
        return None


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, name in enumerate(NAMES):
        smi = fetch(name)
        if smi:
            out[name] = smi
            print(f"  [{i+1}/{len(NAMES)}] {name} -> {smi}")
        else:
            print(f"  [{i+1}/{len(NAMES)}] {name} -> UNRESOLVED (skipped)")
        time.sleep(0.2)  # be polite to PubChem, avoid rate limiting

    path = DATA / "name_to_smiles.json"
    with path.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\nwrote {len(out)}/{len(NAMES)} entries to {path}")


if __name__ == "__main__":
    main()
