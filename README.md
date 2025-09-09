#  An Endmember-Oriented Transformer Network for Bundle-Based Hyperspectral Unmixing
**Demo code for "[An Endmember-Oriented Transformer Network for 
Bundle-Based Hyperspectral Unmixing, TGRS, 2025](https://ieeexplore.ieee.org/abstract/document/10843765)"**


 EOT-Net leverages the advantages of endmember bundles to introduce variability while providing stable endmember results
 with clear physical meaning. We design an endmember-oriented Transformer (EOT) to capture endmember-specific features
 through directional subspace projection and a low-redundancy attention (LRA) mechanism. Subsequently, the proposed network
 is divided into two branches: endmember generation and abundance estimation, to process endmember-specific features. In the
 endmember generation branch, endmember-specific features are transformed into intraclass weights that are used to combine
 signatures within the bundles, and a set of endmembers is generated for each pixel. In the abundance estimation branch,
 endmember-specific features are integrated using a heterogeneous information fusion (HIF) module that leverages the spatial dis
tribution heterogeneity of the endmembers, ultimately producing the abundance results.

**If you use the code in your research, we would appreciate a citation to the original paper:**

```
@ARTICLE{10843765,
  author={Xiang, Shu and Li, Xiaorun and Chen, Shuhan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={An Endmember-Oriented Transformer Network for Bundle-Based Hyperspectral Unmixing}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  keywords={Feature extraction;Transformers;Hyperspectral imaging;Attention mechanisms;Estimation;Classification algorithms;Generators;Data mining;Change detection algorithms;Accuracy;Autoencoder (AE) network;endmember bundle;hyperspectral unmixing (HU);spectral variability (SV);Transformer},
  doi={10.1109/TGRS.2025.3530642}}
```
