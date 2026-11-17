# Imamovic et al. 2018: collateral sensitivity network of P. aeruginosa across 24 antibiotics

Tag: DATA

## Citation and URL

Imamovic et al. "Drug-driven phenotypic convergence supports rational treatment
strategies of chronic infections." Cell 172(1-2):121-134.e14, 2018.
DOI 10.1016/j.cell.2017.12.012. PMCID PMC5766827.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC5766827/

Access date: 2026-11-16.

## Exact claim (reported by the paper)

- Organism: Pseudomonas aeruginosa PAO1, evolved in synthetic cystic-fibrosis sputum
  medium; plus DK2 clinical isolates from three patients.
- Drugs: 24 antibiotics; PAO1 evolved to resistance against each and tested against
  all 24.
- Measurement: MIC of each resistant strain relative to a PAO1 control adapted to the
  medium without drug. A decrease is collateral sensitivity, an increase collateral
  resistance. No fixed fold-change cutoff; significance by simulation from five
  replicate growth-inhibition measurements and one-sided t tests over fold changes
  from 1 to 10.
- Replication: the text read describes 24 resistant PAO1 strains, one per drug. The
  number of independent lineages per drug was not confirmed.

## Data location and licence

- Table S1 ("Susceptibility Profiles", mmc1.xlsx): the MIC matrix behind Fig. 1A.
  Tables S2-S4: collateral-sensitivity cycles and correlations.
- The key resources table also points to a Mendeley Data deposit for the
  susceptibility data; the link shown is a sign-in redirect and its DOI and licence
  could not be confirmed.
- Sequencing: NCBI SRA PRJNA414086.
- Licence: article states CC BY-NC-ND 4.0. Non-commercial and no derivatives; this is
  the most restrictive licence among the sources recorded and bears on redistributing
  a derived matrix.

## Mechanism grouping and class sizes

Drug list and class labels are the paper's own Table 1.

- beta-lactams (10): ampicillin, piperacillin, carbenicillin, ticarcillin, aztreonam,
  cefepime, cefuroxime, ceftazidime, meropenem, imipenem
- aminoglycosides (3): amikacin, gentamicin, tobramycin
- macrolides (3): azithromycin, erythromycin, clarithromycin
- quinolones (2): ciprofloxacin, levofloxacin
- tetracyclines (2): minocycline, doxycycline
- polymyxin (1): colistin
- fosfomycin (1)
- rifamycin (1): rifampicin
- antifolate (1): trimethoprim/sulfamethoxazole

Class-size vectors, each with the rule that produces it:

- Rule "class label as printed": K = 24, M = 9, n = (10, 3, 3, 2, 2, 1, 1, 1, 1).
- Rule "class label, classes with more than one member only": K = 20, M = 5,
  n = (10, 3, 3, 2, 2).
- Rule "cellular process": K = 24, M = 6, n = (11, 8, 2, 1, 1, 1)
  [cell wall, protein synthesis, gyrase, membrane, RNA polymerase, folate].
- The text says "eight chemical classes" while the table carries nine labels.

## Limits

- Licence (above).
- One evolved strain per drug in the PAO1 matrix as far as could be confirmed, so no
  within-drug replication to separate signal from lineage-to-lineage variation.
- Different organism from the other four sources; matrices should not be pooled.
