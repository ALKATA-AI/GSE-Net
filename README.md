# GSE-Net: Multi-Scale Feature Extraction for Single-Channel Speech Enhancement.
### Alireza Kayed-Nezami, Rahil Mahdian Toroghi, Hassan Zareian
**Official implementation of GSE-Net, accepted at [International Conference on Electrical Engineering (ICEE) 2026](https://ieeexplore.ieee.org/document/11710484).**

**Abstract:** Single-channel speech enhancement remains a central challenge in signal processing, where the goal is to recover clean speech from signals contaminated by environmental noise. Although CNN–Transformer hybrids have shown strong performance, the field still requires models capable of capturing complex time–frequency dependencies and extracting multiscale representations. This paper introduces GSE-Net, a deep hybrid architecture that combines the local feature modeling capabilities of convolutional networks with the long-range dependency modeling of attention-based modules. This design strengthens the model’s capacity to jointly capture global and local structures. Experimental evaluations on the VoiceBank+DEMAND dataset show that GSE-Net delivers results that are comparable to, and in many cases surpass, leading state-of-the-art methods. The model maintains high speech intelligibility and significantly enhances perceptual quality, effectively advancing single-channel speech enhancement toward near-clean speech reconstruction.

## Note
Pre-trained model weights are not provided. The model can be trained from scratch using the provided source code.

## Pre-requisites

**<p>1. Python >= 3.6.</p>**
   
**<p>2. Clone this repository:</p>**

```bash
git clone https://github.com/ALKATA-AI/GSE-Net.git
```

**<p>3. Install PyTorch with CUDA 11.8:</p>**

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

**<p>4. Install the Python requirements:</p>**
   
Please refer to [requirements.txt](https://github.com/ALKATA-AI/GSE-Net/blob/main/requirements.txt).

```bash
pip install -r requirements.txt
```

**<p>5. Download and extract the <a href="https://datashare.ed.ac.uk/items/e3eb2725-46b6-40c6-88cf-3ccad62cd419">VoiceBank+DEMAND dataset</a>:</p>**

Resample all wav files to 16 kHz, and move the clean and noisy WAV files to <code>VoiceBank+DEMAND/wavs_clean</code> and <code>VoiceBank+DEMAND/wavs_noisy</code>, respectively. Alternatively, you can directly download the downsampled 16 kHz dataset <a href="https://drive.google.com/drive/folders/19I_thf6F396y5gZxLTxYIojZXC0Ywm8l">here</a>.

## Training

## Inference

## Model Structure
<p align="center">
  The overall architecture of the GSE-Net and GPCFormer, Encoder and Decoder Structural Details.
  <img src="Figures/Fig1-BMP.bmp" alt="GSE-Net Architecture">
</p>

<p align="center">
  Structural details of GPCAM, GPGU, CAM & DDFG in GSE-Net.
  <img src="Figures/Fig2-BMP.bmp" alt="GSE-Net Architecture">
</p>

## Comparison with other SE models
<p align="center">
  <img src="Figures/Table2 - GSE-Net.png" alt="GSE-Net Architecture">
</p>

## Acknowledgements 
We referred to [PrimeK-Net](https://github.com/huaidanquede/PrimeK-Net), [CBAM](https://github.com/Peachypie98/CBAM) and [CMGAN](https://github.com/ruizhecao96/CMGAN) to implement this.

## Citation

```bibtex
@INPROCEEDINGS{11710484,
  author={Kayed-Nezami, Alireza and Toroghi, Rahil Mahdian and Zareian, Hassan},
  booktitle={2026 34th International Conference on Electrical Engineering (ICEE)}, 
  title={GSE-Net: Multi-Scale Feature Extraction for Single-Channel Speech Enhancement}, 
  year={2026},
  pages={299-303},
  doi={10.1109/ICEE70628.2026.11710484}
}
```
