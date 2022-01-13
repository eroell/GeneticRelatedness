# Characterising the importance of genetic relatedness in phenotype prediction

## Scope
Short Lab Rotation of the Computational Biology and Bioinformatics Msc Programme, ETH Zürich
Workload of 6 weeks, 9 ECTS

## Student
Eljas Röllin

## Supervisors
Lucie Bourguignon, Dr. Michael Adamer
Research Group of Prof. Dr. Karsten M. Borgwardt

## Abstract
The effect of single nucleotide polymorphisms (SNPs) on complex phenotypes is often modelled assuming a linear relation of SNPs and the phenotype. In this work, we challenge this assumption and investigate the potential importance of genetic relatedness. We investigate whether individuals with similar SNPs have a similiar phenotype, for which we consider adult human height. Considering similarity in high dimensions, we discuss the curse of dimensionality and approaches to reduce its effects. We focus on developing a software tool which implements these approaches and is able to deal with large datasets. The resulting software tool implements the k-Nearest Neighbor (kNN) rule, and features batched data loading and usage of Graphics Processing Unit (GPU) acceleration. We find that genetic relatedness by usage of the kNN rule has no eminent importance in adult human height. The developed software tool offers flexibility on many levels of the kNN rule, and outperforms its corresponding scikit-learn implementation in terms of execution time and memory requirements.
