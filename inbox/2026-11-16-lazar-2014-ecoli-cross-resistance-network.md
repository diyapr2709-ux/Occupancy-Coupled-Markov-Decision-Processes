# Lazar et al. 2014: cross-resistance network among 12 antibiotics in evolved E. coli

Tag: DATA

## Citation and URL

Lazar et al. "Genome-wide analysis captures the determinants of the antibiotic
cross-resistance interaction network." Nature Communications 5:4352, 2014.
DOI 10.1038/ncomms5352. PMCID PMC4102323.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4102323

Access date: 2026-11-16.

## Exact claim (reported by the paper)

- Organism: Escherichia coli K-12.
- Drugs: 12 antibiotics.
- Design: 120 populations, 10 per antibiotic, from one ancestral clone, plus 10
  control lines evolved without drug.
- Measurement: growth (optical density after 14 h) of every evolved population at
  sublethal doses of each of the other antibiotics, Z-test against the control lines;
  MIC changes confirmed by E-test on a subset (reported false-positive rate about 5%,
  false-negative about 16%).
- Cross-resistance from A to B is called when at least 50% of populations adapted to A
  are less sensitive to B.
- "On average, 52% of all investigated antibiotic pairs showed cross-resistance in at
  least one direction" (66 unordered pairs). Lines adapted to aminoglycosides showed
  few cross-resistance interactions.

## Data location and licence

- Supplementary Data 1 (ncomms5352-s2.xls): the cross-resistance dataset, susceptibility
  change of each of the 120 populations against the other 11 antibiotics.
- Supplementary Data 2 (ncomms5352-s3.xls): validation of interactions.
- Supplementary Data 7 (ncomms5352-s8.xls): cross-resistance patterns of lines with
  selected mutations.
- Sequencing: NCBI BioProject PRJNA248327.
- Licence: article states Creative Commons Attribution 4.0 International (CC BY 4.0).

## Mechanism grouping and class sizes

- The paper's Table 1 ("Antibiotics employed and their modes of actions") is an image
  and could not be read. The text confirms ciprofloxacin, cefoxitin, chloramphenicol,
  ampicillin, doxycycline, trimethoprim, nitrofurantoin and aminoglycosides.
- UNVERIFIED list from recollection, to be checked against Table 1 before any use:
  ampicillin, cefoxitin; ciprofloxacin, nalidixic acid; kanamycin, tobramycin;
  tetracycline, doxycycline; chloramphenicol, erythromycin; trimethoprim;
  nitrofurantoin.
- If that list is right, rule "mode of action as in the paper's Fig. 1a grouping":
  K = 12, M = 7, n = (2, 2, 2, 2, 2, 1, 1). Not confirmed.

## Limits

- K = 12 is short of 20, and the classes are close to equal in size, so this source
  says little about unequal classes.
- Its value is replication (10 populations per drug) and an open licence. It records
  cross-resistance only; collateral sensitivity for the same strain set is in a
  separate 2013 paper not examined in this pass.
- Dense between-class cross-resistance: the measured matrix is not block-diagonal by
  mode of action.
