---
type: Note
_width: normal
---
# Protein design report

two parts evotuning and dpo.

goal: start from pretrained protein language model ESM2 (we are working both with 35m and 650m). after evotuning + dpo we sample new sequences from the model (starting base C05 CDRH3 region) that hopefully have high log-enrichment.

### Evotuning

let's see in detail what we did with evotuning.

dataset: Observed antibody space (OAS) we filtered sequences that have:

- species: human
- vaccine: flu
- chain: heavy

this resulted in XXX sequences (xxx MB). this dataset is located as a fasta file (seq_id + sequence) at /cluster/project/infk/krause/mdenegri/protein-design/datasets/oas_filtered.fasta. we also saved some metadata about each sequence to be able to make some more studies at: /cluster/project/infk/krause/mdenegri/protein-design/datasets/oas_filtered.csv.gz. here we are saving:

- list of columns saved

Sequences were clustered at 99% identity using MMseqs2 easy-linclust (coverage ≥ 90%) to remove clonal duplicates. this reduced the number of sequences to XXX sequences ( xxx MB) and the dedup dataset is stored at /cluster/project/infk/krause/mdenegri/protein-design/datasets/oas_dedup_rep_seq.fasta. this last was also divided into train/val/test following the proportions x/x/x. each split can be found at /cluster/project/infk/krause/mdenegri/protein-design/datasets/oas_dedup_rep_seq_{"split"}.fasta

here a CDRH3 lenth distribution over OAS filtered corpus (this corresponds to a previous dataset, should be recomputed, but the idea is similar)

![BlockNote image](attachments/1778257987447-cdrh3_length_distribution.png)

We also tried to look if C05 was present in the dataset and it's not present. we were expecting this results

we also tried to look for C05-similar sequences in the dataset. More information can be found at: /cluster/project/infk/krause/mdenegri/protein-design/reports/search_report.txt. this report.txt was mainly done looking for whole sequence C05 similarities in the filterd dataset. and then starting from the found sequences we looked for similarities in the CDRH3 region only.

another study was made looking directly at sequence with similar CDRH3 region in the oas_filtered both using MMseqs2 and blosum. we created different datasets that can be found here: /cluster/project/infk/krause/mdenegri/protein-design/datasets/c05/.

The C05 was conducted hoping that to teach some more specific information about C05 to the simple evotuned model.

Here some results:

(/cluster/project/infk/krause/mdenegri/protein-design/plots/meeting_20260415_094343/cdr_perplexity_comparison.png)

(plots to be refreshed with new evotuned checkpoint) (should also add spearman correlation and spearman correlation on left and right flanks)

()

![BlockNote image](attachments/1778259306099-cdr_perplexity_comparison.png)

The TTT was conducted following the paper: <https://arxiv.org/pdf/2411.02109>

### DPO

different techniques tried

unlikelihood failuere
