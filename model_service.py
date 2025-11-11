from transformers import pipeline
from typing import Optional


def infer(
        text_for_classification: str,
        model_name: Optional[str] = None
):
    """
    Функция для вызова Pipeline. По умолчанию pipeline использует модель `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
    :param text_for_classification: текст, с которым будем инференсить модель
    :param model_name: название модели, если нужно использовать не деефолтную
    :return:
    """
    # Объявляем классификатор
    classifier = pipeline(task="sentiment-analysis", model=model_name)

    # инференсим классификатор с текстом, который пришел на вход функции
    output = classifier(text_for_classification)
    return output
