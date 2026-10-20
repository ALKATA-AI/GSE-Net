# GSE-Net: Multi-Scale Feature Extraction for Single-Channel Speech Enhancement.
### Alireza Kayed-Nezami, Rahil Mahdian Toroghi, Hassan Zareian
**Official implementation of GSE-Net, accepted at [International Conference on Electrical Engineering (ICEE) 2026](https://ieeexplore.ieee.org/document/11710484).**

**Abstract:** Single-channel speech enhancement remains a central challenge in signal processing, where the goal is to recover clean speech from signals contaminated by environmental noise. Although CNN–Transformer hybrids have shown strong performance, the field still requires models capable of capturing complex time–frequency dependencies and extracting multiscale representations. This paper introduces GSE-Net, a deep hybrid architecture that combines the local feature modeling capabilities of convolutional networks with the long-range dependency modeling of attention-based modules. This design strengthens the model’s capacity to jointly capture global and local structures. Experimental evaluations on the VoiceBank+DEMAND dataset show that GSE-Net delivers results that are comparable to, and in many cases surpass, leading state-of-the-art methods. The model maintains high speech intelligibility and significantly enhances perceptual quality, effectively advancing single-channel speech enhancement toward near-clean speech reconstruction.

## Note

Pre-trained model weights are not included in this repository. The model can be trained from scratch using the provided source code.

## Prerequisites

### 1. Python Version

Python >= 3.8

### 2. Clone This Repository

```bash
git clone https://github.com/ALKATA-AI/GSE-Net.git
cd GSE-Net
```

### 3. Install the Python Requirements

Install the dependencies listed in [`requirements.txt`](https://github.com/ALKATA-AI/GSE-Net/blob/main/requirements.txt) by running the following command from the repository root directory:

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

### 4. Download and Extract the VoiceBank+DEMAND Dataset

Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/items/e3eb2725-46b6-40c6-88cf-3ccad62cd419).

Resample all WAV files to 16 kHz and place the clean and noisy WAV files in the following directories, respectively:

- `VoiceBank+DEMAND/wavs_clean`
- `VoiceBank+DEMAND/wavs_noisy`

Alternatively, you can download the downsampled 16 kHz dataset [here](https://drive.google.com/drive/folders/19I_thf6F396y5gZxLTxYIojZXC0Ywm8l).

## Training

Single-GPU training is recommended. At least 15 GB of GPU memory is required.

Training parameters can be configured directly in `train.py` or provided as command-line arguments.

```bash
python train.py \
  --input_clean_wavs_dir "VoiceBank+DEMAND/wavs_clean" \
  --input_noisy_wavs_dir "VoiceBank+DEMAND/wavs_noisy" \
  --checkpoint_path "checkpoints" \
  --config "config.json" \
  --input_training_file "VoiceBank+DEMAND/training.txt" \
  --input_validation_file "VoiceBank+DEMAND/test.txt"
```

## Inference

Inference parameters can be specified directly in `inference.py` or provided as command-line arguments.

```bash
python inference.py \
  --checkpoint_file "checkpoints/g_best" \
  --input_clean_wavs_dir "VoiceBank+DEMAND/wavs_clean" \
  --input_noisy_wavs_dir "VoiceBank+DEMAND/wavs_noisy" \
  --input_test_file "VoiceBank+DEMAND/test.txt" \
  --output_dir "generated_files/g_best"
```

## Evaluation

Evaluation parameters can be specified directly in `evaluation.py` or provided as command-line arguments.

```bash
python evaluation.py \
  --config_file "config.json" \
  --input_clean_wavs_dir "VoiceBank+DEMAND/wavs_clean" \
  --input_test_file "VoiceBank+DEMAND/test.txt" \
  --input_enhanced_wavs_dir "generated_files/g_best"
```

## Model Structure
<p align="center">
  The overall architecture of the GSE-Net and GPCFormer, Encoder and Decoder Structural Details.
  <img src="Figures/Fig1.bmp" alt="GSE-Net Architecture">
</p>

<p align="center">
  Structural details of GPCAM, GPGU, CAM & DDFG in GSE-Net.
  <img src="Figures/Fig2.bmp" alt="GSE-Net Architecture">
</p>

## Comparison with other SE models
<p align="center">
  <img src="Figures/Table2.png" alt="GSE-Net Architecture">
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
