from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

app = FastAPI()

model_name = "openai/clip-vit-base-patch16"
clip = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

labels = ['a photo of a cat', 'a photo of a dog', 'not a cat or dog']

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))


    inputs = processor(text=labels, return_tensors="pt", padding=True)


    inputs_image = processor(images=image, return_tensors="pt", padding=True)


    outputs = clip(**inputs, **inputs_image)
    probs = torch.nn.functional.softmax(outputs.logits_per_image[0], dim=0)


    results = {labels[i]: probs[i].item() * 100 for i in range(len(labels))}

    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
