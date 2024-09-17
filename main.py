from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

model_name = "openai/clip-vit-base-patch16"
clip = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

labels = ['a photo of a cat', 'a photo of a dog', 'not a cat or dog', 'an art of a cat', 'an art of a dog']
label_mapping = {
    'a photo of a cat': 'cat',
    'a photo of a dog': 'dog'
}

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Load image
        image = Image.open(io.BytesIO(await file.read()))

        # Prepare text inputs
        text_inputs = processor(text=labels, return_tensors="pt", padding=True, truncation=True)

        # Prepare image inputs
        image_inputs = processor(images=image, return_tensors="pt", padding=True)

        # Get model outputs
        outputs = clip(**text_inputs, pixel_values=image_inputs['pixel_values'])
        logits_per_image = outputs.logits_per_image

        # Compute probabilities
        probs = torch.nn.functional.softmax(logits_per_image[0], dim=0)

        # Prepare results
        results = {labels[i]: probs[i].item() * 100 for i in range(len(labels))}
        max_label = max(results, key=results.get)
        max_prob = results[max_label]

        # Conditions
        if max_label in ['a photo of a cat', 'a photo of a dog']:
            if max_prob <= 80:
                raise HTTPException(status_code=400, detail=f"Error: '{max_label}' must be more than 80%")
            if results['not a cat or dog'] > 50:
                raise HTTPException(status_code=400, detail="Error: 'not a cat or dog' is more than 50%")
            if results['an art of a cat'] > 50 or results['an art of a dog'] > 50:
                raise HTTPException(status_code=400, detail="Error: The image is classified as art with more than 50% confidence")

            # Return simplified label (cat or dog)
            return JSONResponse(content={label_mapping[max_label]: max_prob})

        else:
            raise HTTPException(status_code=400, detail="Error: The highest probability label must be 'a photo of a cat' or 'a photo of a dog'")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
