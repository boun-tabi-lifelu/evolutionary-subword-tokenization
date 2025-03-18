<h1 align="center"> evoBPE: Evolutionary Protein Sequence Tokenization </h1>

<center><img src="figures/evoBPE.svg" alt="evoBPE" width="750px"></center>

</br>

This repo contains all data and source code used during [this work](https://www.arxiv.org/abs/2503.08838).
We provide codes and notebooks to reproduce our work.

- **[Prog](Prog)** contains codes and notebooks.
  - **[Prog/helper_classes.py](Prog/helper_classes.py):** Data Structures of evoBPE.
  - **[Prog/bpe_functions.py](Prog/bpe_functions.py):** Training functions of evoBPE.
  - **[Prog/multiprocess_training.py](Prog/multiprocess_training.py):** The main script to train tokenizers. Utilizes multiprocessing.
  - **[Prog/vocabulary_functions.py](Prog/vocabulary_functions.py):** Useful functions related to vocabulary manipulation.
  - **[Prog/dataset_schema.ipynb](Prog/dataset_schema.ipynb)** covers where we obtained all the data and how we processed them to be utilized for our study.
  - **[Prog/general_tokenizer_statistics.ipynb](Prog/general_tokenizer_statistics.ipynb):** Preliminary Analysis - General Tokenizer Statistics.
  - **[Prog/effect_of_substitution_matrices.ipynb](Prog/effect_of_substitution_matrices.ipynb):** Preliminary Analysis - Effect of Substitution Matrices.
  - **[Prog/pretokenized_dataset_generation.ipynb](Prog/pretokenized_dataset_generation.ipynb)** shows pre-tokenization process.
  - **[Prog/adherence_to_linguistic_laws.ipynb](Prog/adherence_to_linguistic_laws.ipynb):** Preliminary Analysis - Adherence to Linguistic Laws.
  - **[Prog/domain_conservation_analysis.ipynb](Prog/domain_conservation_analysis.ipynb):** Experiments - Domain Conservation Analysis.
  - **[Prog/esm_embedding_similarity_analysis_for_mutations.ipynb](Prog/esm_embedding_similarity_analysis_for_mutations.ipynb):** Experiments - ESM-2 Embedding Similarity Analysis for Mutations.
- **[RSRC/vocabs](RSRC/vocabs)** contains internal and huggingface versions of BPE and evoBPE vocabulary files for the vocabulary size of 6400.
- **[RSRC/dataset](RSRC/dataset)** contains standard and pre-tokenized versions of the UniRef50 human taxanomy proteins. Codes that create fasta files can be found in [Prog/dataset_schema.ipynb](Prog/dataset_schema.ipynb) notebook's 'Generate Fasta Files' section.
  - **[RSRC/dataset/uniref_50.fasta](RSRC/dataset/uniref_50.fasta):** Standard versions of the proteins.
    ```
    format:
    >uniprot_id
    protein sequence

    example:
    >A0A087WZT3
    MELSAEYLREKLQRDLEAEHVLPSPGGVGQVRGETAASETQLGS
    ```
  - **[RSRC/dataset/uniref_50_pretokenized.fasta](RSRC/dataset/uniref_50_pretokenized.fasta):** Pre-tokenized versions of the proteins.
    ```
    format:
    >uniprot_id occurrence=occurence order | source={out_of_domain or InterPro ID or TED ID}
    protein sequence

    example:
    >Q5W8V9 occurrence=2 | source=IPR034325
    PKLLQGVITVIDVFYQYATQHGEYDTLNKAELKELLENEFHQILKNPNDPDTVD
    ```

## Abstract

Recent advancements in computational biology have drawn compelling parallels between protein sequences and linguistic structures, highlighting the need for sophisticated tokenization methods that capture the intricate evolutionary dynamics of protein sequences. Current subword tokenization techniques, primarily developed for natural language processing, often fail to represent protein sequences' complex structural and functional properties adequately. This study introduces evoBPE, a novel tokenization approach that integrates evolutionary mutation patterns into sequence segmentation, addressing critical limitations in existing methods. By leveraging established substitution matrices, evoBPE transcends traditional frequency-based tokenization strategies. The method generates candidate token pairs through biologically informed mutations, evaluating them based on pairwise alignment scores and frequency thresholds. Extensive experiments on human protein sequences show that evoBPE performs better across multiple dimensions. Domain conservation analysis reveals that evoBPE consistently outperforms standard Byte-Pair Encoding, particularly as vocabulary size increases. Furthermore, embedding similarity analysis using ESM-2 suggests that mutation-based token replacements preserve biological sequence properties more effectively than arbitrary substitutions. The research contributes to protein sequence representation by introducing a mutation-aware tokenization method that better captures evolutionary nuances. By bridging computational linguistics and molecular biology, evoBPE opens new possibilities for machine learning applications in protein function prediction, structural modeling, and evolutionary analysis.

If you use this repository, please cite the following related [paper]():
```bibtex
@article{suyunu2025evobpe,
  title={evoBPE: Evolutionary Protein Sequence Tokenization},
  author={Suyunu, Burak and Dolu, {\"O}zdeniz and {\"O}zg{\"u}r, Arzucan},
  journal={arXiv preprint arXiv:2503.08838},
  year={2025}
}
```


## License

The data in this repository is released under terms of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE.txt) for details.

This code base is licensed under the MIT license. See [MIT-LICENSE](MIT-LICENSE.txt) for details.
