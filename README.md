<h1 align="center"> evoBPE: Evolutionary Protein Sequence Tokenization </h1>

</br>

## Abstract

Recent advancements in computational biology have drawn compelling parallels between protein sequences and linguistic structures, highlighting the need for sophisticated tokenization methods that capture the intricate evolutionary dynamics of protein sequences. Current subword tokenization techniques, primarily developed for natural language processing, often fail to represent protein sequences' complex structural and functional properties adequately. This study introduces evoBPE, a novel tokenization approach that integrates evolutionary mutation patterns into sequence segmentation, addressing critical limitations in existing methods. By leveraging established substitution matrices, evoBPE transcends traditional frequency-based tokenization strategies. The method generates candidate token pairs through biologically informed mutations, evaluating them based on pairwise alignment scores and frequency thresholds.  Extensive experiments on human protein sequences show that evoBPE performs better across multiple dimensions. Domain conservation analysis reveals that evoBPE consistently outperforms standard Byte-Pair Encoding, particularly as vocabulary size increases. Furthermore, embedding similarity analysis using ESM-2 suggests that mutation-based token replacements preserve biological sequence properties more effectively than arbitrary substitutions. The research contributes to protein sequence representation by introducing a mutation-aware tokenization method that better captures evolutionary nuances. By bridging computational linguistics and molecular biology, evoBPE opens new possibilities for machine learning applications in protein function prediction, structural modeling, and evolutionary analysis.

If you use this repository, please cite the following related [paper]():
```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={},
  publisher={}
}
```


## License

This code base is licensed under the MIT license. See [LICENSE](license.md) for details.
