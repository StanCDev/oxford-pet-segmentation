from PIL import Image
import requests
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import matplotlib.pyplot as plt


#url = "https://github.com/timojl/clipseg/blob/master/example_image.jpg?raw=true"
image = Image.open('res/Abyssinian_1.jpg')
#image = Image.open(requests.get(url, stream=True).raw)

image.show()

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
#prompts = ["a glass", "something to fill", "wood", "a colander"]
prompts = ["a cat", "a dog", "a bird", "a fish"]

inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

# predict
with torch.no_grad():
  outputs = model(**inputs)

preds = outputs.logits.unsqueeze(1)

# visualize prediction
_, ax = plt.subplots(1, 5, figsize=(15, 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)];
[ax[i+1].text(0, -15, prompts[i]) for i in range(4)];
plt.show()