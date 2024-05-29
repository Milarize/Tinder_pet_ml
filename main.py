from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

app = FastAPI()

model_name = "openai/clip-vit-base-patch16"
clip = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)


labels = ['a photo of a cat', 'a photo of a dog', 'not a cat or dog', 'an art of a cat', 'an art of a dog']

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))


    inputs = processor(text=labels, return_tensors="pt", padding=True)


    inputs_image = processor(images=image, return_tensors="pt", padding=True)


    outputs = clip(**inputs, **inputs_image)
    probs = torch.nn.functional.softmax(outputs.logits_per_image[0], dim=0)


    results = {labels[i]: probs[i].item() * 100 for i in range(len(labels))}


    max_label = max(results, key=results.get)
    max_prob = results[max_label]

    #conditions
    if max_label in ['a photo of a cat', 'a photo of a dog']:
        if max_prob <= 80:
            raise HTTPException(status_code=400, detail=f"Error: '{max_label}' must be more than 80%")
        if results['not a cat or dog'] > 50:
            raise HTTPException(status_code=400, detail="Error: 'not a cat or dog' is more than 50%")
        if results['an art of a cat'] > 50 or results['an art of a dog'] > 50:
            raise HTTPException(status_code=400, detail="Error: The image is classified as art with more than 50% confidence")
    else:
        raise HTTPException(status_code=400, detail="Error: The highest probability label must be 'a photo of a cat' or 'a photo of a dog'")


    return JSONResponse(content={max_label: max_prob})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
