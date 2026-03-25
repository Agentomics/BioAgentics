"""Task #1086: Patient subgroup network characterization.

Characterizes patient heterogeneity subgroups, their network signatures,
identifies universal vs subgroup-specific hubs, and maps subgroups to
predicted symptom profiles.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from collections import defaultdict
import networkx as nx

DATA_DIR = Path("data/pandas_pans/autoantibody_network")


def load_data():
    with open(DATA_DIR / "patient_subgroups.json") as f:
        subgroups = json.load(f)
    shared_private = pd.read_csv(DATA_DIR / "shared_vs_private_analysis.tsv", sep="\t")
    subnetwork_edges = pd.read_csv(DATA_DIR / "subgroup_subnetworks.tsv", sep="\t")
    hub_metrics = pd.read_csv(DATA_DIR / "hub_centrality_metrics.tsv", sep="\t")
    kegg_node_map = pd.read_csv(DATA_DIR / "kegg_node_pathway_mapping.tsv", sep="\t")
    symptom_enrichment = pd.read_csv(DATA_DIR / "symptom_seed_enrichment.tsv", sep="\t")
    return subgroups, shared_private, subnetwork_edges, hub_metrics, kegg_node_map, symptom_enrichment


def characterize_subgroups(subgroups, shared_private, subnetwork_edges, kegg_node_map):
    """Quantify subnetwork size, unique pathways, shared vs private disruption."""
    results = []

    for sg_name, sg_info in subgroups["subgroups"].items():
        # Get nodes in this subgroup
        sg_edges = subnetwork_edges[subnetwork_edges["subgroup"] == sg_name]
        sg_nodes = set(sg_edges["source"].unique()) | set(sg_edges["target"].unique())

        # Shared/private classification
        sg_shared = shared_private[
            shared_private["subgroups"].str.contains(sg_name, na=False, regex=False)
        ]
        n_universal = sg_shared[sg_shared["is_universal"] == True].shape[0]
        n_shared = sg_shared[sg_shared["n_subgroups"] > 1].shape[0]
        n_private = sg_shared[sg_shared["n_subgroups"] == 1].shape[0]

        # Pathways for this subgroup's genes
        sg_pathways = kegg_node_map[kegg_node_map["gene_symbol"].isin(sg_nodes)]
        n_pathways = sg_pathways["pathway_id"].nunique()

        # Unique pathways (not found in other subgroups' nodes)
        all_other_nodes = set()
        for other_name in subgroups["subgroups"]:
            if other_name != sg_name:
                other_edges = subnetwork_edges[subnetwork_edges["subgroup"] == other_name]
                all_other_nodes |= set(other_edges["source"].unique()) | set(other_edges["target"].unique())
        private_nodes = sg_nodes - all_other_nodes
        private_pathways = kegg_node_map[kegg_node_map["gene_symbol"].isin(private_nodes)]
        n_unique_pathways = private_pathways["pathway_id"].nunique()

        shared_ratio = n_shared / max(len(sg_nodes), 1)

        results.append({
            "subgroup": sg_name,
            "description": sg_info["description"],
            "clinical_phenotype": sg_info["clinical_phenotype"],
            "n_seeds": len(sg_info["seed_genes"]),
            "seed_genes": ", ".join(sg_info["seed_genes"]),
            "n_nodes": sg_info["n_nodes"],
            "n_edges": sg_info["n_edges"],
            "n_universal_nodes": n_universal,
            "n_shared_nodes": n_shared,
            "n_private_nodes": n_private,
            "shared_ratio": round(shared_ratio, 3),
            "n_pathways": n_pathways,
            "n_unique_pathways": n_unique_pathways,
        })

    return pd.DataFrame(results)


def hub_subgroup_specificity(subgroups, subnetwork_edges, hub_metrics):
    """Identify which hubs are universal vs subgroup-specific."""
    top_hubs = hub_metrics.nlargest(50, "hub_score")["gene_symbol"].tolist()

    hub_presence = {}
    for hub in top_hubs:
        present_in = []
        for sg_name in subgroups["subgroups"]:
            sg_edges = subnetwork_edges[subnetwork_edges["subgroup"] == sg_name]
            sg_nodes = set(sg_edges["source"].unique()) | set(sg_edges["target"].unique())
            if hub in sg_nodes:
                present_in.append(sg_name)
        hub_presence[hub] = present_in

    n_subgroups = len(subgroups["subgroups"])
    results = []
    for hub, groups in hub_presence.items():
        hub_info = hub_metrics[hub_metrics["gene_symbol"] == hub].iloc[0]
        results.append({
            "gene_symbol": hub,
            "hub_score": round(hub_info["hub_score"], 4),
            "node_type": hub_info["node_type"],
            "n_subgroups_present": len(groups),
            "subgroups": ", ".join(groups),
            "is_universal": len(groups) == n_subgroups,
            "is_subgroup_specific": len(groups) == 1,
            "specificity": "universal" if len(groups) == n_subgroups else (
                "specific" if len(groups) == 1 else "shared"
            ),
        })

    return pd.DataFrame(results)


def subgroup_random_comparison(subgroups, subnetwork_edges, kegg_node_map, n_perms=1000):
    """Test if subgroup subnetworks are significantly different from random subnetworks."""
    # Full network nodes
    all_nodes = set(subnetwork_edges["source"].unique()) | set(subnetwork_edges["target"].unique())
    all_nodes_list = list(all_nodes)
    rng = np.random.default_rng(42)

    # Build full graph
    G = nx.Graph()
    for _, row in subnetwork_edges.iterrows():
        G.add_edge(row["source"], row["target"])

    results = {}
    for sg_name, sg_info in subgroups["subgroups"].items():
        sg_size = sg_info["n_nodes"]
        if sg_size < 5:
            results[sg_name] = {"pval": None, "reason": "subgroup too small"}
            continue

        # Observed: pathway count for this subgroup
        sg_edges = subnetwork_edges[subnetwork_edges["subgroup"] == sg_name]
        sg_nodes = set(sg_edges["source"].unique()) | set(sg_edges["target"].unique())
        sg_pathways = kegg_node_map[kegg_node_map["gene_symbol"].isin(sg_nodes)]["pathway_id"].nunique()

        # Observed: density of the subgraph
        sg_graph = G.subgraph(sg_nodes)
        observed_density = nx.density(sg_graph) if len(sg_nodes) > 1 else 0

        # Permutation: random subsets of same size
        random_pathways = []
        random_densities = []
        for _ in range(n_perms):
            random_nodes = set(rng.choice(all_nodes_list, size=min(sg_size, len(all_nodes_list)), replace=False))
            rp = kegg_node_map[kegg_node_map["gene_symbol"].isin(random_nodes)]["pathway_id"].nunique()
            random_pathways.append(rp)
            rg = G.subgraph(random_nodes)
            random_densities.append(nx.density(rg) if len(random_nodes) > 1 else 0)

        pathway_pval = np.mean([rp >= sg_pathways for rp in random_pathways])
        density_pval = np.mean([rd >= observed_density for rd in random_densities])

        results[sg_name] = {
            "n_nodes": sg_size,
            "observed_pathways": sg_pathways,
            "random_pathway_mean": round(np.mean(random_pathways), 1),
            "pathway_pval": round(pathway_pval, 4),
            "observed_density": round(observed_density, 6),
            "random_density_mean": round(np.mean(random_densities), 6),
            "density_pval": round(density_pval, 4),
            "significantly_different_pathways": pathway_pval < 0.05,
            "significantly_different_density": density_pval < 0.05,
        }

    return results


def subgroup_symptom_mapping(subgroups, symptom_enrichment):
    """Map subgroups to predicted symptom profiles using existing enrichment data."""
    # Use group-level enrichment results
    group_enrich = symptom_enrichment[symptom_enrichment["test_type"] == "group"]

    seed_to_group = {
        "DRD1": "dopaminergic", "DRD2": "dopaminergic",
        "CAMK2A": "calcium_kinase",
        "PKM": "metabolic", "ALDOC": "metabolic", "ENO1": "metabolic", "ENO2": "metabolic", "ENO3": "metabolic",
        "TUBB3": "structural",
        "FOLR1": "structural",  # closest match
    }

    results = {}
    for sg_name, sg_info in subgroups["subgroups"].items():
        # Map seeds to functional groups
        relevant_groups = set()
        for seed in sg_info["seed_genes"]:
            if seed in seed_to_group:
                relevant_groups.add(seed_to_group[seed])

        # Aggregate symptom predictions
        symptom_scores = defaultdict(float)
        sig_domains = set()
        for group in relevant_groups:
            grp_rows = group_enrich[group_enrich["seed_or_group"] == group]
            for _, row in grp_rows.iterrows():
                if row["significant"]:
                    sig_domains.add(row["symptom_domain"])
                symptom_scores[row["symptom_domain"]] += row["fold_enrichment"]

        results[sg_name] = {
            "clinical_phenotype": sg_info["clinical_phenotype"],
            "seed_genes": sg_info["seed_genes"],
            "functional_groups": sorted(relevant_groups),
            "predicted_significant_symptoms": sorted(sig_domains),
            "symptom_enrichment_scores": {k: round(v, 2) for k, v in
                                          sorted(symptom_scores.items(), key=lambda x: -x[1])[:5]},
            "phenotype_match": assess_phenotype_match(sg_info["clinical_phenotype"], sig_domains),
        }

    return results


def assess_phenotype_match(clinical_phenotype, predicted_domains):
    """Check if predicted symptom domains match the expected clinical phenotype."""
    phenotype_lower = clinical_phenotype.lower()
    matches = []
    if "ocd" in phenotype_lower and "ocd_compulsive" in predicted_domains:
        matches.append("OCD")
    if "tic" in phenotype_lower and "tic_motor" in predicted_domains:
        matches.append("tics")
    if ("anxiety" in phenotype_lower or "emotional" in phenotype_lower) and "anxiety" in predicted_domains:
        matches.append("anxiety/emotional")
    if "cognitive" in phenotype_lower and "cognitive" in predicted_domains:
        matches.append("cognitive")
    if "eating" in phenotype_lower and "eating_restriction" in predicted_domains:
        matches.append("eating")

    if not predicted_domains:
        return "no predictions"
    if matches:
        return f"matched: {', '.join(matches)}"
    return "no match with stated phenotype"


def main():
    print("=" * 60)
    print("Task #1086: Patient Subgroup Network Characterization")
    print("=" * 60)

    subgroups, shared_private, subnetwork_edges, hub_metrics, kegg_node_map, symptom_enrichment = load_data()

    # 1. Characterize subgroups
    print("\n[1/5] Characterizing subgroups...")
    char_df = characterize_subgroups(subgroups, shared_private, subnetwork_edges, kegg_node_map)
    char_df.to_csv(DATA_DIR / "subgroup_characterization.tsv", sep="\t", index=False)
    for _, row in char_df.iterrows():
        print(f"  {row['subgroup']:25s} nodes={row['n_nodes']:4d} edges={row['n_edges']:4d} "
              f"pathways={row['n_pathways']:3d} unique_pw={row['n_unique_pathways']:3d} "
              f"shared_ratio={row['shared_ratio']:.2f}")

    # 2. Hub specificity
    print("\n[2/5] Analyzing hub subgroup specificity...")
    hub_spec = hub_subgroup_specificity(subgroups, subnetwork_edges, hub_metrics)
    hub_spec.to_csv(DATA_DIR / "hub_subgroup_specificity.tsv", sep="\t", index=False)
    n_universal = hub_spec[hub_spec["is_universal"]].shape[0]
    n_specific = hub_spec[hub_spec["is_subgroup_specific"]].shape[0]
    n_shared = hub_spec[hub_spec["specificity"] == "shared"].shape[0]
    print(f"  Top 50 hubs: {n_universal} universal, {n_shared} shared, {n_specific} subgroup-specific")

    # 3. Random comparison
    print("\n[3/5] Permutation tests (1000 perms per subgroup)...")
    random_results = subgroup_random_comparison(subgroups, subnetwork_edges, kegg_node_map, n_perms=1000)
    for sg, res in random_results.items():
        if "pathway_pval" not in res:
            print(f"  {sg:25s}: {res.get('reason', 'skipped')}")
        else:
            print(f"  {sg:25s}: pathways p={res['pathway_pval']:.4f}, density p={res['density_pval']:.4f}")

    # 4. Symptom mapping
    print("\n[4/5] Mapping subgroups to symptom profiles...")
    symptom_map = subgroup_symptom_mapping(subgroups, symptom_enrichment)
    for sg, pred in symptom_map.items():
        symptoms = ", ".join(pred["predicted_significant_symptoms"][:4]) or "none"
        match = pred["phenotype_match"]
        print(f"  {sg:25s}: {symptoms:50s} [{match}]")

    # 5. Clinical stratification assessment
    print("\n[5/5] Clinical stratification assessment...")
    stratification = {
        "feasibility": "HIGH - autoantibody panels (Cunningham Panel) already clinically available",
        "actionable_subgroups": [],
        "recommendations": [],
    }

    for sg_name, sg_info in subgroups["subgroups"].items():
        pred = symptom_map.get(sg_name, {})
        symptoms = pred.get("predicted_significant_symptoms", [])
        if symptoms:
            stratification["actionable_subgroups"].append({
                "subgroup": sg_name,
                "seeds": sg_info["seed_genes"],
                "predicted_symptoms": symptoms,
                "clinical_action": f"Monitor for {', '.join(symptoms)} symptoms",
            })

    stratification["recommendations"] = [
        "Autoantibody profiling via expanded Cunningham Panel could predict symptom domains",
        "Dopaminergic-dominant patients (anti-DRD1/DRD2): anticipate OCD, anxiety, eating restriction",
        "CaMKII-positive patients: additional anxiety risk; consider anxiolytic support",
        "Metabolic/structural profiles: symptom prediction requires brain-circuit models beyond gene-level association",
        "Epitope spreading monitoring: serial antibody testing could track disease progression",
        "JAK inhibitors (tofacitinib, baricitinib) target the top enriched pathway across all subgroups",
        "PIK3R1/AKT1 pathway modulators represent cross-layer therapeutic opportunities",
    ]

    # Compile final summary
    summary = {
        "subgroup_characterization": char_df.to_dict("records"),
        "hub_specificity": {
            "n_universal": n_universal,
            "n_shared": n_shared,
            "n_specific": n_specific,
            "universal_hubs": hub_spec[hub_spec["is_universal"]]["gene_symbol"].tolist(),
        },
        "random_comparison": random_results,
        "symptom_predictions": symptom_map,
        "clinical_stratification": stratification,
    }

    with open(DATA_DIR / "subgroup_characterization_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("PATIENT SUBGROUP CHARACTERIZATION COMPLETE")
    print(f"Output: subgroup_characterization.tsv, hub_subgroup_specificity.tsv,")
    print(f"        subgroup_characterization_summary.json")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    summary = main()
