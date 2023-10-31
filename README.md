# FMNERG

Here are codes and dataset for our ACM MM2023 paper: [Fine-Grained Multimodal Named Entity Recognition and Grounding with a Generative Framework](https://dl.acm.org/doi/10.1145/3581783.3612322)

- ## Dataset

  Our dataset is built on the [GMNER dataset](https://github.com/NUSTM/GMNER).

  - The preprocessed CoNLL format files are provided in this repo. For each tweet, the first line is its image id, and the following lines are its textual contents.
  - Download each tweet's associated images via this link (<https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view>)

  ## Usage

  ### Training for TIGER

  ```
  sh run.sh
  ```

  ### Evaluation

  ```
  sh eval.sh
  ```

  ## Acknowledgements

  - Using the dataset means you have read and accepted the copyrights set by Twitter and original dataset providers.
  - Some codes are based on the codes of  [VL-T5](https://github.com/j-min/VL-T5), thanks a lot!
