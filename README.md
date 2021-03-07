# pointnet.phasedetection





<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![Product Name Screen Shot][product-screenshot]
This repo uses PointNet [[1]](https://arxiv.org/abs/1612.00593), a neural network designed for computer vision applications using point clouds. In this study, a properly-trained PointNet was demonstrated to be highly generalizable on morphology detection in molecular simulations, and can be potentially extended to discovery of emerging ordered patterns from non-equilibrium systems.

The PointNet was trained on atomic coordinates of mesophases including lamellar (LAM), body-centered cubic (BCC), hexagonally-packed cylinder (HPC), hexagonally-perforeated lamellar (HPL), and disorderd (DIS) from molecular dynamics (MD) simulation trajectories from our previous work [[2]](https://pubs.acs.org/doi/10.1021/jacs.0c01829), [[3]](https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.9b01248), [[4]](https://pubs.acs.org/doi/abs/10.1021/acsnano.7b09122) and synthetic point clouds for ordered network morphologies that were absent from previous simulations. 



### Built With





<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  pip install -m requirements.txt
  ```

### Data Downloads


1. Doownload pre-processed and raw training data from [https://drive.google.com/drive/folders/1N8BjACdNCKTmEnRF46VKkoHufLV8VoMt?usp=sharing](https://drive.google.com/drive/folders/1N8BjACdNCKTmEnRF46VKkoHufLV8VoMt?usp=sharing)




<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Project Link: [https://github.com/donshen/pointnet.phasedetection](https://github.com/donshen/pointnet.phasedetection)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/POINTNET_SCHEME_PRE.png
