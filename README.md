# Multi-source data and knowledge fusion via deep learning for dynamical systems: applications to spatiotemporal cardiac modeling


Advanced sensing and imaging provide unprecedented opportunities to collect data from diverse sources for increasing information visibility in spatiotemporal dynamical systems. Furthermore, the fundamental physics of the dynamical system is commonly elucidated through a set of partial differential equations (PDEs), which plays a critical role in delineating the manner in which the sensing signals can be modeled. Reliable predictive modeling of such spatiotemporal dynamical systems calls upon the effective fusion of fundamental physics knowledge and multi-source sensing data. This paper proposes a multi-source data and knowledge fusion framework via deep learning for dynamical systems with applications to spatiotemporal cardiac modeling. This framework not only achieves effective data fusion through capturing the physics-based information flow between different domains, but also incorporates the geometric information of a 3D system through a graph Laplacian for robust spatiotemporal predictive modeling. We implement the proposed framework to model cardiac electrodynamics under both healthy and diseased heart conditions. Numerical experiments demonstrate the superior performance of our framework compared with traditional approaches that lack the capability for effective data fusion or geometric information incorporation.


# Requirements


Python: 3.10.4

Tensorflow 2.9.1

Cuda: 11.7

Cudnn: 8.4



# Citation
Please cite our paper if you use our code in your research:

@article{yao2024multi,
  title={Multi-source data and knowledge fusion via deep learning for dynamical systems: applications to spatiotemporal cardiac modeling},
  author={Yao, Bing},
  journal={IISE Transactions on Healthcare Systems Engineering},
  pages={1--14},
  year={2024},
  publisher={Taylor \& Francis}
}