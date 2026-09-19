# 🌌 Galaxy Morphology Classifier ✨

A deep learning web application that classifies galaxy images into 10 morphological types using a fine-tuned ResNet-50 model trained on the Galaxy10 DECaLS dataset.

## Demo
Upload any galaxy image and receive an instant classification with confidence scores across all 10 galaxy types.

## Galaxy Types
- Disturbed Galaxies
- Merging Galaxies
- Round Smooth Galaxies
- In-between Round Smooth Galaxies
- Cigar Shaped Smooth Galaxies
- Barred Spiral Galaxies
- Unbarred Tight Spiral Galaxies
- Unbarred Loose Spiral Galaxies
- Edge-on Galaxies without Bulge
- Edge-on Galaxies with Bulge

## Results
| Model | Validation Accuracy |
|-------|-------------------|
| Custom CNN (baseline) | 55.5% |
| ResNet-50 (transfer learning) | 70.2% |

## Tech Stack
- **Python** — core language
- **PyTorch** — model training and inference
- **ResNet-50** — pretrained model architecture
- **Streamlit** — web application framework
- **Galaxy10 DECaLS** — 17,736 labeled galaxy images

## Project Structure
```
galaxy-classifier/
    app.py              ← Streamlit web app
    src/
        explore_data.py ← Data exploration
        preprocess.py   ← Data preprocessing
        model.py        ← Custom CNN architecture
        train.py        ← CNN training script
        transfer_learning.py ← ResNet-50 fine-tuning
    data/               ← Dataset (not included)
    models/             ← Saved models (not included)
    results/            ← Sample outputs
```


## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download the Galaxy10 DECaLS dataset from [Zenodo](https://zenodo.org/records/10845026)
4. Run preprocessing: `python src/preprocess.py`
5. Train the model: `python src/transfer_learning.py`
6. Launch the app: `streamlit run app.py`

## Dataset
Galaxy10 DECaLS dataset containing 17,736 galaxy images across 10 morphological classes, sourced from DESI Legacy Imaging Surveys with labels from Galaxy Zoo.