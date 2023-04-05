# PROJECT : Watermarking ML Models

- This project is about watermarking machine learning models using [the ml-model-watermarking library](https://github.com/SAP/ml-model-watermarking). The library provides methods for inserting watermarks into trained models, as well as extracting and verifying them.
- Watermark models on various tasks, such as image classification or sentiment analysis, with compatibility with the main Machine Learning frameworks like Scikit-learn, PyTorch, or the HuggingFace library.
- Here,we have created some baseline models and then watermarked them using the above library
- our aim to do an experiments to verify the watermaking in the extracted models of baseline models and if not verified, understanding its limitation.


## Installation

To install the dependencies for this project, run the following command:
    `pip install -r requirements.txt`

### Using MLModelWatermarking

[The MLModelWatermarking library](https://github.com/SAP/ml-model-watermarking) provides watermark machine learning models.ML Model Watermarking acts as a wrapper for your model, providing a range of techniques referred from the provided reference paper below for watermarking your model as well as ownership detection function
#### To watermark in the model
    from mlmodelwatermarking.markface import TrainerWM 
    trainer = TrainerWM(model=your_model)
    ownership = trainer.watermark()
    watermarked_model = trainer.get_model()
#### To verify if a model has been stolen:
    from mlmodelwatermarking.marktorch import TrainerWM
    from mlmodelwatermarking.verification import verify
    trainer = TrainerWM(model=suspect_model, ownership=ownership)
    trainer.verify()
    #return a dictionary with information about the verification status


## Reference Papers used

- [Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring](https://www.usenix.org/conference/usenixsecurity18/presentation/adi)by Adi et al.
- [Protecting Intellectual Property of Deep Neural Networks with Watermarking](https://dl.acm.org/doi/abs/10.1145/3196494.3196550?casa_token=RZrfzSIO_uwAAAAA:N7ohyz15GCGfoXRMtew-dX5dV-heZyI-N5Tod1xyKFWb46MXLPeqdfhMLizAFXlVE_VfZP_m2T3M) by Zhang et al.
- [Adversarial frontier stitching for remote neural network watermarking](https://arxiv.org/pdf/1711.01894.pdf) by Merrer et al.
- [DAWN: Dynamic Adversarial Watermarking of Neural Networks](https://arxiv.org/pdf/1906.00830.pdf) by Szyller et al.

