# Copyright 2022 Arkadiusz Choruży


from pathlib import Path

from sc.subsystems.classifier.model import LitModel as Classifier


CKPT_PATH = Path(__file__).parent/'../../artifacts/classification_model.ckpt'


def predict(data):
    model = Classifier.load_from_checkpoint(CKPT_PATH)
    predicted = model.predict(data)
    return predicted