from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_service import infer as model_infer


class RequestBody(BaseModel):
    text: str
    language: Optional[str] = "en"


class ResponseBody(BaseModel):
    label: str
    score: float


app = FastAPI()


@app.post("/infer")
async def infer(
        body: RequestBody
) -> ResponseBody:
    """
    Метод для взаимодействия с пайплайном sentiment-analysis от HuggingFace.
    Можно использовать как для английского, так и для русского языка (параметр language в теле запроса "en"/"ru")
    :param body:
    :return:
    """
    if body.language == "en":
        result = model_infer(body.text)
    elif body.language == "ru":
        result = model_infer(body.text, model_name="blanchefort/rubert-base-cased-sentiment")
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported language",
        )
    return ResponseBody(**result[0])
