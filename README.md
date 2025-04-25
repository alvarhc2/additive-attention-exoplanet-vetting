# Additive Attention for Vetting Transiting Exoplanet Candidates

This repository contains the code and resources associated with the paper:

**"Additive Attention for Vetting Transiting Exoplanet Candidates"**
**Àlvar Hernández-Carnerero, Miquel Sànchez-Marrè, Juan Carlos Morales**

(Accepted for publication in *The Astronomical Journal*)  
[![DOI](https://zenodo.org/badge/919767503.svg)](https://doi.org/10.5281/zenodo.15212407)

## 🛰️ Overview

In this study, we propose a deep learning architecture combining **Long Short-Term Memory (LSTM)** networks and **Additive Attention mechanisms** to vet transiting exoplanet candidates in light curve data. Our architecture is evaluated on datasets from NASA's **Kepler** and **TESS** missions, and demonstrates competitive performance with reduced model complexity compared to standard convolutional approaches.

Key features of our model include:

- Joint use of **CNN**, **LSTM**, and **Additive Attention** layers
- Emphasis on **model interpretability** via attention
- Generalizable across **different datasets** and survey conditions
- Designed for **scientific reproducibility** and ease of experimentation

## 🔬 Abstract

> In this study we propose a deep learning model architecture using Long Short-Term Memory (LSTM) networks and Additive Attention mechanisms for *vetting transiting exoplanet candidates*. Our method is applied to two different datasets (*Kepler* and *TESS*) comprising light curves, representing periodic fluctuations in star brightness potentially indicative of planetary transits. Making use of pre-processed data from the *Kepler* and *TESS* missions, we evaluate the effectiveness of LSTM-based approaches in contrast to conventional CNN methods. Through extensive experimentation, we demonstrate the efficacy of our final model, which integrates CNN, LSTM, and Additive Attention layers, coupled with Feed-Forward Neural Networks, achieving competitive performance with limited model complexity. Notably, the incorporation of Additive Attention enhances prediction accuracy across various model scales without significantly increasing model complexity. Our findings highlight the utility of Additive Attention in *vetting transiting exoplanet candidates*, offering understanding of the model decision-making processes. Overall, our proposed architecture presents a general-purpose approach for addressing transit detection challenges across different datasets, contributing to advancements in astronomical data analysis.

## 📁 Repository Structure

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow (keras)
- NumPy, Pandas, Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

### Training

To train the model on the Kepler dataset:

```bash
python main.py --dataset kepler --model attn_lstm
```

To train on the TESS dataset:

```bash
python main.py --dataset tess --model attn_lstm
```

### Evaluation

To evaluate a trained model:

```bash
python main.py --dataset tess --eval --checkpoint path/to/model.ckpt
```

## 📊 Results

We show that incorporating **Additive Attention** improves classification accuracy while maintaining low model complexity. See the ```results/``` folder for example performance metrics and attention maps.

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this code for your research, please cite:

```bibtex
@article{hernandez2025additive,
  title={Additive Attention for Vetting Transiting Exoplanet Candidates},
  author={Hernàndez-Carnerero, Àlvar and Sànchez-Marrè, Miquel and Morales, Juan Carlos},
  journal={The Astronomical Journal},
  year={2025},
  note={Accepted}
}
```

## 👥 Code Author

- Alvar Hernandez Carnerero – [@alvarhc2](https://github.com/alvarhc2)  
  PhD Candidate in Artificial Intelligence  
  Universitat Politècnica de Catalunya (UPC)  
  [alvar.hernandez@upc.edu]

## 🙌 Acknowledgments

This work is based on data from the NASA [Kepler](https://www.nasa.gov/mission_pages/kepler/main/index.html) and [TESS](https://tess.mit.edu/) missions. We thank the open-source community and the astronomical ML community for foundational tools and ideas.
