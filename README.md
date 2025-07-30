<img width="936" height="431" alt="Screenshot 2025-07-10 at 5 12 49 PM" src="https://github.com/user-attachments/assets/8bcd69d5-c02c-4f17-bb33-25c6363f1935" />

## LBM - TRI (wip)

Implementation of the Large Behavioral Model architecture for [dexterous manipulation](https://arxiv.org/abs/2507.05331) from Toyota Research Institute. 

[Project Page](https://toyotaresearchinstitute.github.io/lbm1/)

[Data Normalization Takeaway](https://github.com/tuul-ai/robotbuilder/blob/main/timestep_norm.md)


## Install

```shell
$ pip install TRI-LBM
```

## Usage

```python
import torch
from TRI_LBM.lbm import LBM

lbm = LBM(
    action_dim = 20,
    dim_pose = 10
)

commands = ['pick up the apple and place in the blue tray']
images = torch.randn(1, 3, 3, 224, 224)
actions = torch.randn(1, 16, 20)
pose = torch.randn(1, 10)

loss = lbm(
    text = commands,
    images = images,
    pose = pose,
    actions = actions,
)

loss.backward()

# after much training

sampled_actions = lbm.sample(
    text = commands,
    images = images,
    pose = pose,
) # (1, 16, 20)
```

## Citations

```bibtex
@misc{trilbmteam2025carefulexaminationlargebehavior,
    title  = {A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation}, 
    author = {TRI LBM Team and Jose Barreiros and Andrew Beaulieu and Aditya Bhat and Rick Cory and Eric Cousineau and Hongkai Dai and Ching-Hsin Fang and Kunimatsu Hashimoto and Muhammad Zubair Irshad and Masha Itkina and Naveen Kuppuswamy and Kuan-Hui Lee and Katherine Liu and Dale McConachie and Ian McMahon and Haruki Nishimura and Calder Phillips-Grafflin and Charles Richter and Paarth Shah and Krishnan Srinivasan and Blake Wulfe and Chen Xu and Mengchao Zhang and Alex Alspach and Maya Angeles and Kushal Arora and Vitor Campagnolo Guizilini and Alejandro Castro and Dian Chen and Ting-Sheng Chu and Sam Creasey and Sean Curtis and Richard Denitto and Emma Dixon and Eric Dusel and Matthew Ferreira and Aimee Goncalves and Grant Gould and Damrong Guoy and Swati Gupta and Xuchen Han and Kyle Hatch and Brendan Hathaway and Allison Henry and Hillel Hochsztein and Phoebe Horgan and Shun Iwase and Donovon Jackson and Siddharth Karamcheti and Sedrick Keh and Joseph Masterjohn and Jean Mercat and Patrick Miller and Paul Mitiguy and Tony Nguyen and Jeremy Nimmer and Yuki Noguchi and Reko Ong and Aykut Onol and Owen Pfannenstiehl and Richard Poyner and Leticia Priebe Mendes Rocha and Gordon Richardson and Christopher Rodriguez and Derick Seale and Michael Sherman and Mariah Smith-Jones and David Tago and Pavel Tokmakov and Matthew Tran and Basile Van Hoorick and Igor Vasiljevic and Sergey Zakharov and Mark Zolotas and Rares Ambrus and Kerri Fetzer-Borelli and Benjamin Burchfiel and Hadas Kress-Gazit and Siyuan Feng and Stacie Ford and Russ Tedrake},
    year   = {2025},
    eprint = {2507.05331},
    archivePrefix = {arXiv},
    primaryClass = {cs.RO},
    url = {https://arxiv.org/abs/2507.05331}, 
}
```

```bibtex
@inproceedings{Wagenmaker2025SteeringYD,
    title   = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
    author  = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:279464702}
}
```

```bibtex
@misc{heng2025vitacformerlearningcrossmodalrepresentation,
    title   = {ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation}, 
    author  = {Liang Heng and Haoran Geng and Kaifeng Zhang and Pieter Abbeel and Jitendra Malik},
    year    = {2025},
    eprint  = {2506.15953},
    archivePrefix = {arXiv},
    primaryClass = {cs.RO},
    url     = {https://arxiv.org/abs/2506.15953}, 
}
```
