# Maeda et al. 2020: IC50 profiles of 192 evolved E. coli strains across 47 stressors

Tag: DATA

## Citation and URL

Maeda et al. "High-throughput laboratory evolution reveals evolutionary constraints
in Escherichia coli." Nature Communications 11:5970, 2020.
DOI 10.1038/s41467-020-19713-w. PMCID PMC7686311.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC7686311/
- https://github.com/jiwasawa/evolved-strain-analysis (viewed only; nothing cloned or run)

Access date: 2026-11-16.

## Exact claim (reported by the paper)

- Organism: Escherichia coli MDS42 (insertion-sequence-free strain).
- Design: laboratory evolution under 95 stressors, six lines each. 192 evolved strains
  kept for analysis: four from each of 47 stressors plus controls.
- Measurement: IC50 of each of the 192 strains against all 47 chemicals (9,024
  values), from a sigmoidal fit to optical density after 24 h.
- Cross-resistance / collateral sensitivity: IC50 of the four strains evolved under
  stressor A, measured in stressor B, compared with the parent (13 replicates) by a
  Mann-Whitney U test at 5% false discovery rate. 336 cross-resistance and 157
  collateral-sensitivity pairs out of 2,162 directed combinations.

## Data location and licence

- Article supplement: Supplementary Data 1-5 (xlsx) and a Source Data zip. Supplementary
  Data 1 holds the stressor list (not opened in this pass).
- Repository file data/resistance_norm.csv: 192 rows (strains) by 48 value columns.
  Column headers as found: CP, RFP, 5.FU, H2O2, CMZ, LVAL, B.Cl.Ala, AZT, KM, 6.MP,
  AF, NiCl, CBPC, NFLX, SDC, SXZ, FOS, KTe, PS, BZ, DCS, MMC, VCM, ATP, PLM, PHEN,
  3.AT, NVA, MEC, TET, M9, NQO, EDTA, SHX, PMZ, NIT, SS, BSD, GAH, FTD, EM, HSE,
  CCCP, ABU, PUR, DVAL, 5.FOA, NMNO. Values run from roughly -17 to +15; the page
  does not define the normalisation. 48 columns against 47 stressors is unexplained
  on the page (the M9 column is probably the no-stress control; unconfirmed).
- Licence: article CC BY 4.0. Repository MIT.

## Mechanism grouping and class sizes

- Most of the 47 stressors are not clinical antibiotics (metabolic inhibitors, metals,
  detergents, amino-acid analogues).
- Reading the abbreviations (my reading, to be checked against Supplementary Data 1;
  the article text itself confirms only carbenicillin, cefmetazole, aztreonam,
  tetracycline, erythromycin, chloramphenicol, puromycin, phleomycin, norfloxacin),
  the clinical antibiotics are about 15: beta-lactams CMZ, AZT, CBPC, MEC; then one
  each for CP, EM, TET, KM, NFLX, RFP, SXZ, FOS, DCS, VCM, NIT.
- By drug class that gives one class of 4 and about eleven singletons, K about 15.
  By cellular process, roughly: cell wall 7 (four beta-lactams, FOS, DCS, VCM),
  protein synthesis 4, and singletons for gyrase, RNA polymerase, folate,
  nitrofuran: n about (7, 4, 1, 1, 1, 1). Both vectors are provisional until the
  abbreviations are confirmed.
- A K = 20 instance would have to draw on non-antibiotic stressors.

## Limits

- Best data access of the sources found (a ready CSV under MIT), weakest fit to an
  antibiotic-only K = 20, M = 5 instance.
- The paper's own analysis groups evolved strains by clustering of resistance
  profiles rather than by drug class.
