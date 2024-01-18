# Emotion Recognition using Emotion Embeddings, MLM, and emotion correlations

This repo contains code and execution scripts for ICASSP '23 papers [Leveraging Label Correlations in a Multi-label Setting: A Case Study in Emotion](https://arxiv.org/abs/2210.15842) and [Using Emotion Embeddings to Transfer Knowledge Between Emotions, Languages, and Annotation Formats](https://arxiv.org/abs/2211.00171).

## INCAS Phase 2 Evaluation instruction

We have tested *this* functionality with `Python 3.10.13`, so be sure to create a suitable `Python virtual environment`. With your environment activated, install our dependencies:

```bash
pip install .
```

Then, *download* and unzip our pretrained [model](https://drive.google.com/file/d/1LPh-iEpdqhpO9TqoU-BCZHYiZ4c2ImTq/view?usp=share_link). The created folder contains the model parameters, brief model configuration files, and brief README of the sequence of training. To get the *predictions*, run the `annotate.py` script:

```bash
python annotate.py --pretrained-folder /path/to/pretrained/model/folder \
--emotion-config ./emotion_configs/paletz_revised.json --domain twitter \
--input-filename /path/to/your/input/file --input-format jsonl --out /path/to/output.jsonl \
--device cuda:0 --text-column contentText --id-column id
```

Use `--help` to see the rest of the arguments if necessary (e.g., you may need / want to adjust the batch size depending on the VRAM of the GPU, default values take ~11GBs). If the file containing the samples is large, slowing down the system when loaded at once, you can break it into multiple files and provide them *all* in `--input-filename`.