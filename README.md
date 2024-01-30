# MTPNet
Code of paper ["Multi-scale Transformer Pyramid Networks for Multivariate Time Series Forecasting"] ([IEEE Access](https://ieeexplore.ieee.org/abstract/document/10412052))

MTPNet achieves SOTA on nine benchmarks.

## Introduction
Multivariate Time Series (MTS) forecasting involves modeling temporal dependencies within historical records. Transformers have demonstrated remarkable performance in MTS forecasting due to their capability to capture long-term dependencies. However, prior work has been confined to modeling temporal dependencies at either a fixed scale or multiple scales that exponentially increase (most with base 2). This limitation hinders their effectiveness in capturing diverse seasonalities, such as hourly and daily patterns. In this paper, we introduce a dimension invariant embedding technique that captures short-term temporal dependencies and projects MTS data into a higher-dimensional space, while preserving the dimensions of time steps and variables in MTS data. Furthermore, we present a novel Multi-scale Transformer Pyramid Network (MTPNet), specifically designed to effectively capture temporal dependencies at multiple unconstrained scales. The predictions are inferred from multi-scale latent representations obtained from transformers at various scales. Extensive experiments on nine benchmark datasets demonstrate that the proposed MTPNet outperforms recent state-of-the-art methods.

## Train and Test
1. Install the required packages: `pip install -r requirements.txt`
2. Data are publicly available at [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b8f4a78a39874ac9893e/?dl=1).
3. To reproduce the experimental results presented in the paper. Simply run the scripts at "/MTPNet/scripts/" as follows:
   ```
   bash ./scripts/ETTh1.sh
   bash ./scripts/ETTh2.sh
   bash ./scripts/ETTm1.sh
   bash ./scripts/ETTm2.sh
   bash ./scripts/electricity.sh
   bash ./scripts/Exchange_Rate.sh
   bash ./scripts/Traffic.sh
   bash ./scripts/WTH.sh
   bash ./scripts/ILI.sh
   ```

## Citation
If you find this repository beneficial for your research, kindly include a citation:
```
@ARTICLE{10412052,
  author={Zhang, Yifan and Wu, Rui and Dascalu, Sergiu M. and Harris, Frederick C.},
  journal={IEEE Access}, 
  title={Multi-scale Transformer Pyramid Networks for Multivariate Time Series Forecasting}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Transformers;Forecasting;Decoding;Market research;Time series analysis;Predictive models;Data models;Time series analysis;time series forecasting;transformer;multi-scale feature pyramid;value embedding},
  doi={10.1109/ACCESS.2024.3357693}}

```

## Acknowledgements
We sincerely appreciate the foundational code from the following GitHub repositories: \
https://github.com/wanghq21/MICN \
https://github.com/zhouhaoyi/Informer2020 \
https://github.com/Thinklab-SJTU/Crossformer \
https://github.com/thuml/Time-Series-Library \
https://github.com/cure-lab/SCINet
