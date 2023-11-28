# FMNERG

Here are codes and dataset for our ACM MM2023 paper: [Fine-Grained Multimodal Named Entity Recognition and Grounding with a Generative Framework](https://dl.acm.org/doi/10.1145/3581783.3612322)

- ## Dataset

  Our dataset is built on the [GMNER dataset](https://github.com/NUSTM/GMNER).

  - The preprocessed CoNLL format files are provided in this repo. For each tweet, the first line is its image id, and the following lines are its textual contents.
  - Download each tweet's associated images via this link (<https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view>)
  - Use [VinVL](https://github.com/pzzhang/VinVL) to identify all the candidate objects, and put them under the folder named "twitterFMNERG_vinvl_extract36". We have uploaded the features extracted by VinVL to [Google Drive](https://drive.google.com/drive/folders/1w7W4YYeIE6bK2lAfqRtuwxH-tNqAytiK?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1QqjOlAAjCqAk_qL6ejeARw?pwd=TwVi) (code: TwVi).

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
