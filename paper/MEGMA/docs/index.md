## Welcome to MEGMA tutorial Pages!

This tutorial is about the metagenomic deep learning and biomarker discovery based on **MEGMA**.

[![PyPI version](https://badge.fury.io/py/aggmap.svg)](https://badge.fury.io/py/aggmap)
[![Documentation Status](https://readthedocs.org/projects/bidd-aggmap/badge/?version=latest)](https://bidd-aggmap.readthedocs.io/en/latest/?badge=latest)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dkawtw4hanY3ks0mBMqvN1beskF6usjC)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6450642.svg)](https://doi.org/10.5281/zenodo.6450642)

### MEGMA
**MEGMA** is short for metagenomic **M**icrobial **E**mbedding, **G**rouping, and **M**apping **A**lgorithm (MEGMA) , which is a further step development of **AggMap** that specific for metagenomic data learning. **MEGMA** is developed to transform the tabular metagenomic data into spatially-correlated color image-like 2D-representations, named as the 2D-microbiomeprints (3D tensor data in the form of row, column and channel). 2D-microbiomeprints are multichannel feature maps (Fmaps) and are the inputs of ConvNet-based AggMapNet models. 

**MEGMA** is released in the **aggmap** package, in this tutorial, we will show how to employ the **aggmap** package for **MEGMA** implementary.

for metagenomic-based disease prediction by deep learning model and identifying the important signatures.



### Tutorial Content

[Metagenomic deep learning and biomarker discovery](#metagenomic-deep-learning-and-biomarker-discovery)

*   [1\. Introduction](./pages/example_00_Introduction.html)

    *   [1.1 MEGMA introduction](./pages/example_00_Introduction.html#1.1-MEGMA-introduction)
    *   [1.2 Metagenomic cross nation datasets and tasks](./pages/example_00_Introduction.html#1.2-Metagenomic-cross-nation-datasets-and-tasks)
    *   [1.3 MEGMA fitting and AggMapNet training strategy](./pages/example_00_Introduction.html#1.3-MEGMA-fitting-and-AggMapNet-training-strategy)
*   [2\. MEGMA Training & Transformation](./pages/example_01_MEGMA.html)
    *   [2.1 Fitting MEGMA on metagenomic abundance data of all countries](./pages/example_01_MEGMA.html#2.1-Fitting-MEGMA-on-metagenomic-abundance-data-of-all-countries)
        *   [2.1.1 Read and preprocess data for MEGMA](./pages/example_01_MEGMA.html#2.1.1-Read-and-preprocess-data-for-MEGMA)
        *   [2.1.2 MEGMA initialization, fitting and dump](./pages/example_01_MEGMA.html#2.1.2-MEGMA-initialization,-fitting-and-dump)
        *   [2.1.3 MEGMA loading and 2D-microbiomeprints transformation](./pages/example_01_MEGMA.html#2.1.3-MEGMA-loading-and-2D-microbiomeprints-transformation)
        *   [2.1.4 MEGMA 2D-microbiomeprints visulization](./pages/example_01_MEGMA.html#2.1.4-MEGMA-2D-microbiomeprints-visulization)
        *   [2.1.5 Well-trained MEGMA to transform the abundance data of each country](./pages/example_01_MEGMA.html#2.1.5-Well-trained-MEGMA-to-transform-the-abundance-data-of-each-country)
    *   [2.2 Fitting MEGMA on metagenomic abundance data of one country only](./pages/example_01_MEGMA.html#2.2-Fitting-MEGMA-on-metagenomic-abundance-data-of-one-country-only)
        *   [2.2.1 Read and preprocess data for MEGMA](./pages/example_01_MEGMA.html#2.2.1-Read-and-preprocess-data-for-MEGMA)
        *   [2.2.2 MEGMA initialization & fitting](./pages/example_01_MEGMA.html#2.2.2-MEGMA-initialization-&-fitting)
        *   [2.2.3 MEGMA 2D-microbiomeprints transformation](./pages/example_01_MEGMA.html#2.2.3-MEGMA-2D-microbiomeprints-transformation)
        *   [2.2.4 MEGMA Fmaps visulization](./pages/example_01_MEGMA.html#2.2.4-MEGMA-Fmaps-visulization)
        *   [2.2.5 Transform the abandance data of the rest countries by country-specific MEGMA](./pages/example_01_MEGMA.html#2.2.5-Transform-the-abandance-data-of-the-rest-countries-by-country-specific-MEGMA)
        *   [2.2.6 Fitting country-specific megma for all countries](./pages/example_01_MEGMA.html#2.2.6-Fitting-country-specific-megma-for-all-countries)
    *   [2.3 Discussions & conclusions on MEGMA 2D-microbiomeprints](./pages/example_01_MEGMA.html#2.3-Discussions-&-conclusions-on-MEGMA-2D-microbiomeprints)
*   [3\. Training the CRC detection models based on MEGMA Fmaps](./pages/example_02_AggMapNet.html)
    *   [3.1 Training and test AggMapNet on overall MEGMA Fmaps](./pages/example_02_AggMapNet.html#3.1-Training-and-test-AggMapNet-on-overall-MEGMA-Fmaps)
    *   [3.2 Training and test AggMapNet on country specific MEGMA Fmaps](./pages/example_02_AggMapNet.html#3.2-Training-and-test-AggMapNet-on-country-specific-MEGMA-Fmaps)
    *   [3.3 Comparing the STST performance and discussion](./pages/example_02_AggMapNet.html#3.3-Comparing-the-STST-performance-and-discussion)
*   [4\. Important microbial marker identification](./pages/example_03_Explaination.html)
    *   [4.1 Calculate the global feature importance](./pages/example_03_Explaination.html#4.1-Calculate-the-global-feature-importance)
        *   [4.1.1 GFI for model trained on overall MEGMA Fmaps](./pages/example_03_Explaination.html#4.1.1-GFI-for-model-trained-on-overall-MEGMA-Fmaps)
        *   [4.1.2 GFI for model trained on country specific MEGMA Fmaps](./pages/example_03_Explaination.html#4.1.2-GFI-for-model-trained-on-country-specific-MEGMA-Fmaps)
    *   [4.2 Generate the explaination saliency map](./pages/example_03_Explaination.html#4.2-Generate-the-explaination-saliency-map)
        *   [4.2.1 Saliency map for overall MEGMA Fmaps](./pages/example_03_Explaination.html#4.2.1-Saliency-map-for-overall-MEGMA-Fmaps)
        *   [4.2.2 Saliency map country specific MEGMA Fmaps](./pages/example_03_Explaination.html#4.2.2-Saliency-map-country-specific-MEGMA-Fmaps)
    *   [4.3 Global feature importance correlation](./pages/example_03_Explaination.html#4.3-Global-feature-importance-correlation)
    *   [4.4 Discussions and conclusions on saliency map](./pages/example_03_Explaination.html#4.4-Discussions-and-conclusions-on-saliency-map)
*   [5\. Toplogical analyisis on the important microbes](./pages/example_04_toplogical_analyisis.html)
    *   [5.1 Plotting the the embedded and arranged microbes](./pages/example_04_toplogical_analyisis.html#5.1-Plotting-the-the-embedded-and-arranged-microbes)
    *   [5.2 Plotting the linear assignment of the embedded microbes](./pages/example_04_toplogical_analyisis.html#5.2-Plotting-the-linear-assignment-of-the-embedded-microbes)
    *   [5.3 Fetching the optimized toplogical graph and clustering](./pages/example_04_toplogical_analyisis.html#5.3-Fetching-the-optimized-toplogical-graph-and-clustering)
    *   [5.4 Exploring the toplogical relationship of the important microbes](./pages/example_04_toplogical_analyisis.html#5.4-Exploring-the-toplogical-relationship-of-the-important-microbes)
*   [6\. Exploring the embedding & grouping in MEGMA](./pages/example_05_embedding_grouping.html)
    *   [6.1 Microbial embedding](./pages/example_05_embedding_grouping.html#6.1-Microbial-embedding)
        *   [6.1.1 Manifold embedding](./pages/example_05_embedding_grouping.html#6.1.1-Manifold-embedding)
        *   [6.1.2 Ramdom embedding](./pages/example_05_embedding_grouping.html#6.1.2-Ramdom-embedding)
    *   [6.2 Microbial grouping](./pages/example_05_embedding_grouping.html#6.2-Microbial-grouping)
        *   [6.2.1 Hierarchical clustering tree based grouping](./pages/example_05_embedding_grouping.html#6.2.1-Hierarchical-clustering-tree-based-grouping)
        *   [6.2.2 Taxonomic tree based grouping](./pages/example_05_embedding_grouping.html#6.2.2-Taxonomic-tree-based-grouping)
