# Oz et al. 2014: measured cross-resistance among 22 antibiotics in evolved E. coli

Tag: DATA

## Citation and URL

Oz et al. "Strength of selection pressure is an important parameter contributing to
the complexity of antibiotic resistance evolution." Molecular Biology and Evolution
31(9):2387-2401, 2014. DOI 10.1093/molbev/msu191. PMCID PMC4137714.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4137714/

Access date: 2026-11-16.

## Exact claim (reported by the paper)

- Organism: Escherichia coli MG1655, drug-sensitive wild type, M9 minimal medium, 30 C.
- Drugs: 22 antibiotics.
- Design: 88 populations evolved for 21 days; for each drug, two under strong
  selection and two under mild selection. Two strong-selection cefoxitin populations
  became mutators and were excluded from the genomic analyses.
- Measurement: MIC of every evolved population against all 22 drugs (growth for about
  22 h across a concentration series, MIC interpolated at a fixed final optical
  density, in duplicate). Each value is normalised as
  log10(MIC / MIC_wild-type) / log10(MIC_max / MIC_wild-type) and also coded as
  -1 (increased susceptibility), 0 (unchanged) or +1 (increased resistance).
- Stronger selection produced more cross-resistance; populations evolved to
  aminoglycosides under strong selection became more susceptible to several other
  classes.

## Data location and licence

- Supplementary table S1: daily MIC values for all evolved populations.
- Supplementary table S2: the normalised resistance matrix (evolved population by
  drug). This is the cross-resistance matrix.
- Supplementary table S3: genetic changes. Sequencing reads: NCBI SRA SRP042989.
- The supplement is attached to the journal article; it was not downloaded in this
  pass, so file format and completeness of S2 are unconfirmed. No separate data
  repository is named.
- Licence: article states Creative Commons Attribution Non-Commercial 3.0
  (CC BY-NC 3.0). Non-commercial reuse only; this is not CC BY.

## Mechanism grouping and class sizes

Drug list and labels are the paper's own Table 1.

| Drug | Class column | Target column |
|---|---|---|
| chloramphenicol, clindamycin, erythromycin, spiramycin, fusidic acid | protein synthesis inhibitor | protein synthesis, 50S |
| amikacin, tobramycin, streptomycin, kanamycin | aminoglycoside | protein synthesis, 30S |
| tetracycline, doxycycline, spectinomycin | protein synthesis inhibitor | protein synthesis, 30S |
| piperacillin, ampicillin, cefoxitin | beta-lactam | cell wall |
| nalidixic acid, lomefloxacin, ciprofloxacin | DNA gyrase inhibitor | DNA gyrase |
| sulfamonomethoxine, trimethoprim, sulfamethoxazole | folic acid synthesis inhibitor | folic acid synthesis |
| nitrofurantoin | multiple | multiple mechanisms |

Class-size vectors, each with the rule that produces it:

- Rule "target column as printed": K = 22, M = 6, n = (7, 5, 3, 3, 3, 1)
  [30S, 50S, cell wall, gyrase, folate, multiple].
- Same rule, leaving out the one drug the table assigns no single target
  (nitrofurantoin): K = 21, M = 5, n = (7, 5, 3, 3, 3).
- Rule "class column crossed with target column": K = 22, M = 7,
  n = (5, 4, 3, 3, 3, 3, 1) [50S, aminoglycoside, other 30S, beta-lactam, gyrase,
  folate, multiple].
- Rule "cellular process" (all ribosomal drugs merged): K = 22, M = 5,
  n = (12, 3, 3, 3, 1).

## Limits

- The three rules above give three different vectors from one table, so the data fix
  class sizes only once a grouping rule is fixed.
- Groups are by drug target, which stands in for shared resistance mechanism; the
  paper reports cross-resistance across groups, so the measured matrix is not
  block-diagonal by group.
- Two populations per drug per selection strength.
