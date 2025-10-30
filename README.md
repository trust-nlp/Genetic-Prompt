## Genetic Prompt （EMNLP2025）
This repo contains the code for paper [Attributes as Textual Genes: Leveraging LLMs as Genetic Algorithm Simulators for Conditional Synthetic Data Generation](https://arxiv.org/abs/2509.02040), which will appear at Findings of EMNLP 2025. 

## Framework
![Geneticprompt](framework.pdf)

## Dataset

We use eight datasets covering diverse domains and tasks.  
The table below lists the basic statistics and download sources.  
We also provide preprocessing scripts in `/Data-preprocess` to convert the raw data into expected format.

| Dataset | Domain | Task | Download |
|----------|---------|-------|-----------|
| AG News | News | Classification (CLS) | [Link](https://huggingface.co/datasets/yyu/agnews-attrprompt) |
| StackExchange | Science | Classification (CLS) | [Link](https://huggingface.co/datasets/yyu/stackexchange-attrprompt) |
| ChemProt | Biomedicine | Relation Extraction (RE) | [Link](https://huggingface.co/datasets/AdaptLLM/ChemProt) |
| DDI | Pharmacology | Relation Extraction (RE) | [Link](https://github.com/isegura/DDICorpus) |
| SemEval | Web | Relation Extraction (RE) | [Link](https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |
| CoNLL04 | News | Relation Extraction (RE) | [Link](https://huggingface.co/datasets/DFKI-SLT/conll04) |
| SciTLDR | Science | Abstractive Summarization (ABS) | [Link](https://github.com/allenai/scitldr/tree/master/SciTLDR-Data) |
| MeQSum | Medical | Abstractive Summarization (ABS) | [Link](https://github.com/abachaa/MeQSum) |

## Synthetic Data Generation

See `/Generation` and `/Generation-scripts` for details.

## Downstream Model Training

See `/Downstream` and `/Downstream-scripts` for details.

## Contact
Feel free to contact ghan AT memphis DOT edu for any questions and collaboration opportunities.

## Citation
If you find this repository helpful, please kindly consider citing the corresponding paper. Thanks in advance!

```
@misc{han2025attributestextualgenesleveraging,
      title={Attributes as Textual Genes: Leveraging LLMs as Genetic Algorithm Simulators for Conditional Synthetic Data Generation}, 
      author={Guangzeng Han and Weisi Liu and Xiaolei Huang},
      year={2025},
      eprint={2509.02040},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.02040}, 
}
```

## Acknowledgement

Inspired by and partly based on [AttrPrompt](https://github.com/yueyu1030/AttrPrompt). Thanks to the authors for open-sourcing it.



