# Sakenova et al. 2025: chemical-genetics map of cross-resistance and collateral sensitivity in E. coli

Tag: DATA

## Citation and URL

Sakenova et al. "Systematic mapping of antibiotic cross-resistance and collateral
sensitivity with chemical genetics." Nature Microbiology 10(1):202-216, 2025
(online 2 December 2024). DOI 10.1038/s41564-024-01857-w. PMCID PMC11726442.

- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11726442/
- Full text read through https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11726442/fullTextXML
  (the PMC and publisher pages returned a browser check / login redirect).

Access date: 2026-11-16.

## Exact claim (reported by the paper)

- Organism: Escherichia coli BW25113 (training data from earlier studies also used MG1655).
- Drugs: 40 antibiotics.
- The interaction map is INFERRED, not measured pair by pair. Fitness s-scores of
  3,904 single-gene deletion mutants under each of the 40 antibiotics (a previously
  published chemical-genetics dataset, the paper's ref. 31) are turned into an
  "outlier concordance-discordance metric" (OCDM) per drug pair. Cutoffs: OCDM above
  105.159057 is called cross-resistance (XR), below 27.224792 collateral sensitivity
  (CS), otherwise neutral. The metric was fitted on 206 drug pairs taken from four
  published evolution studies (111 neutral, 70 XR, 25 CS).
- Result: 840 drug pairs classified as 404 XR, 267 CS, 169 neutral.
- Measured part: 70 inferred interactions were tested by experimental evolution
  (12 lineages, about 50 generations, IC90 in both directions; 14 pairs to about 100
  generations). 64 of 70 validated, where validated means a log2 IC90 change beyond
  +1 or -1 in at least one lineage and direction.
- "Antibiotics from the same class exhibited exclusively XR interactions"; pairs from
  different classes showed both XR and CS.

## Data location and licence

- Supplementary Table 1: chemical-genetics data used. Supplementary Table 2: the
  XR / neutral / CS call per drug pair, with the gene sets behind it. Supplementary
  Table 3: drugs used. All attached to the article.
- Interactive copy: https://shiny-portal.embl.de/shinyapps/app/21_xrcs (page did not
  render as text; download options and terms not confirmed).
- Sequencing reads: https://doi.org/10.5281/zenodo.10572857
- Licence: article states Creative Commons Attribution 4.0 International (CC BY 4.0).
  No separate licence statement was seen for the Shiny portal.

## Mechanism grouping and class sizes

- The paper groups drugs "according to their modes of action" and its Fig. 4d legend
  counts 8 predominantly-CS classes plus 12 predominantly-XR classes, which implies
  about 20 classes for 40 drugs.
- Only six classes have more than one member: beta-lactams, aminoglycosides,
  quinolones, macrolides, tetracyclines, sulfonamides.
- Class-size vector: NOT OBTAINED. The per-drug list is in Supplementary Table 3
  (xlsx), which could not be read in this pass. The text alone names 14 of the 40
  drugs. No vector is given here rather than guessing one.

## Limits for use as a drug-by-mechanism matrix

- Calls are categorical (XR / CS / neutral) and undirected; the paper says the metric
  cannot assess direction. Only the 70 validated pairs carry measured, directed
  magnitudes.
- Getting from 40 drugs to about 20 requires choosing a subset, which is a selection
  step the data do not make.
- 840 classified pairs exceeds the 780 unordered pairs among 40 drugs; the text read
  does not explain the difference. Check Supplementary Table 2 before relying on the
  count.
