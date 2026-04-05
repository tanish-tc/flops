import ray
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification

app = FastAPI()

class DocumentRequest(BaseModel):
    document_id: str
    ocr_text: str

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1})
@serve.ingress(app)
class DeadlineDeployment:
    def __init__(self):
        # Load the optimized ONNX model
        model = ORTModelForTokenClassification.from_pretrained("./onnx_model")
        tokenizer = AutoTokenizer.from_pretrained("./onnx_model")
        self.ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    @app.post("/predict")
    async def predict(self, req: DocumentRequest):
        results = self.ner(req.ocr_text)
        deadlines = [{"extracted_text": e["word"], "event_type": "DUE_DATE", "confidence": float(e["score"])} for e in results]
        return {"document_id": req.document_id, "deadlines": deadlines}

# Bind the deployment so Ray can run it
entrypoint = DeadlineDeployment.bind()
