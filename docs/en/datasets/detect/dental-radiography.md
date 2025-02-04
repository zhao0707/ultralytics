---
comments: true
description: Explore the dental radiography dataset with X-ray images. Essential for training AI models to detect implants, fillings, impacted teeth, and cavities.
keywords: dental radiography dataset, dental X-rays, dental imaging, implant detection, cavity detection, AI in dentistry, computer vision, dental health, early diagnosis
---

# Dental Radiography Dataset

The dental radiography dataset consists of dental X-ray images, providing detailed information for detecting dental conditions such as implants, fillings, impacted teeth, and cavities. This dataset is vital for [training](https://www.ultralytics.com/glossary/training-data) computer vision models, enhancing the accuracy of automated dental diagnostics and early treatment planning.

## Dataset Structure

The dental radiography dataset is divided into three subsets:

- **Training set**: 1,075 images with corresponding annotations.
- **Validation set**: 121 images with paired annotations.
- **Test set**: 73 images, each annotated for accurate evaluation.

## Applications

This dataset enables applications in dental diagnostics, including:

- **Automated Implant Detection:** Identify dental implants with precision.
- **Cavity Detection:** Enhance early diagnosis of cavities.
- **Fillings Identification:** Detect the presence and condition of dental fillings.
- **Impacted Teeth Recognition:** Assist in diagnosing impacted teeth for surgical planning.

## Dataset YAML

The dataset configuration is defined in a YAML file, detailing dataset paths, classes, and other key information. Access the YAML file at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dental-radiography.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dental-radiography.yaml).

!!! example "ultralytics/cfg/datasets/dental-radiography.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/dental-radiography.yaml"
    ```

## Usage

To [train](../../models/yolo11.md) a YOLO11n model on the dental radiography dataset for 100 epochs with an image size of 640 using the code snippets below.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="dental-radiography.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=dental-radiography.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a fine-tuned dental model

        # Inference using the model
        results = model.predict("https://ultralytics.com/assets/dental-sample.jpg")
        ```

    === "CLI"

        ```bash
        # Start prediction with a fine-tuned *.pt model
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/dental-sample-2.jpg"
        ```

## Sample Images and Annotations

The dataset includes diverse dental X-ray images, annotated for implants, fillings, impacted teeth, and cavities.

![Dental radiography dataset sample image](https://github.com/ultralytics/docs/releases/download/0/dental-radiography-dataset-sample-image.avif)

- **Mosaiced Image:** This image showcases a training batch with mosaiced dental X-rays. Mosaicing merges multiple images, increasing batch diversity and improving model generalization.

## Citations and Acknowledgments

We’d like to thank the authors of the [dataset](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography) for creating this amazing resource, which can be used for medical research. The original dataset wasn’t in YOLO format, the Ultralytics team has converted it and added it as an official dataset for use with the [Ultralytics](https://github.com/ultralytics/ultralytics) [Python package](https://pypi.org/project/ultralytics/).

## FAQ

### What classes are included in the dental radiography dataset?

The dataset includes four key dental conditions, implants, fillings, impacted teeth, and cavities. Each class helps improve model performance in detecting specific dental issues.

### How is the dataset structured for model training?

The dataset is divided into training (1,075 images), validation (121 images), and test (73 images) subsets. This structure ensures effective model training, validation, and performance evaluation.

### How can I improve the performance of my YOLO11 model with this dataset?

To improve model performance:
- Use pretrained YOLO models as a base.
- Apply data augmentation techniques like rotation and scaling.
- Fine-tune hyperparameters based on validation performance.

### What are the benefits of using AI for dental radiography analysis?

AI models enhance early detection of dental conditions, improve diagnostic accuracy, reduce human error, and support faster clinical workflows, benefiting both dental professionals and patients.

### How do I perform inference with my trained dental radiography model?

Inference can be done via Python or CLI commands:

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("path/to/best.pt")
        results = model.predict("https://ultralytics.com/assets/dental-sample.jpg")
        ```

    === "CLI"

        ```bash
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/dental-sample.jpg"
        ```

### Where can I find the YAML configuration for the dental radiography dataset?

You can find the YAML file at [dental-radiography.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dental-radiography.yaml), detailing dataset paths, class names, and configuration settings for model training.
