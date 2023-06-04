from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import base64
from PIL import Image
import io


app = FastAPI()
z = np.load('outputs/z.npy')
numbers = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
x = np.zeros((sum(numbers), 28 * 28))
labels = np.zeros(sum(numbers))
for i in range(len(numbers)):
    for j in range(numbers[i]):
        x[sum(numbers[:i]) + j] = np.array(Image.open('datasets/MNIST/test/{}/{}.png'.format(i, j))).reshape(-1)
        x[sum(numbers[:i]) + j] = x[sum(numbers[:i]) + j] / 255
        labels[sum(numbers[:i]) + j] = i

origins = [
    "http://localhost:3000",
    "https://mnist-ukr.web.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ClickPosition(BaseModel):
    x: float
    y: float
@app.post('/')
def create_user(clickposition: ClickPosition):
    input_z = np.array([[clickposition.x/500 * 2 -1, int(clickposition.y * -1) / 500 * 2 +1]])
    nearnum = np.argmin(np.sum((input_z - z)**2, axis=1))
    nearnum = int(nearnum)
    print(nearnum)
    print(type(nearnum))

    sigma = 0.1
    d = (input_z[:, None, :] - z[None, :, :])**2
    d = np.sum(d, axis=2)
    d = np.exp(-1 / (2 * sigma**2) * d)
    y = np.einsum('ij,jd->id', d, x)
    y = y / np.sum(d, axis=1, keepdims=True) * 255
    y = y.reshape(28, 28)
    print(np.max(y))
    pil_img = Image.fromarray(y.astype(np.uint8))
    stream = io.BytesIO()
    pil_img.save(stream, format='PNG')  # 画像をPNG形式で保存する場合
    image_base64 = base64.b64encode(stream.getvalue()).decode('utf-8')

    near_img = x[nearnum].reshape(28, 28)
    near_img = near_img * 255

    print(np.max(near_img))
    near_img = Image.fromarray(near_img.astype(np.uint8))
    stream = io.BytesIO()
    near_img.save(stream, format='PNG')  # 画像をPNG形式で保存する場合
    near_image_base64 = base64.b64encode(stream.getvalue()).decode('utf-8')
    return {"image": image_base64, 'nearImage': near_image_base64, 'nearNum': nearnum}
@app.get("/")
def Hello():
    z = np.load('outputs/z.npy')
    z = z.tolist()
    return {"z": z}

@app.get("/labels")
def Haello():
    labels = np.load('outputs/labels.npy')
    labels = labels.tolist()
    return {"labels": labels}

# @app.post("/click/{}")
# def nearimage(clickposition: ClickPosition):
#     input_z = np.array([[clickposition.x/500 * 2 -1, int(clickposition.y * -1) / 500 * 2 +1]])

#     sigma = 0.1
#     d = (input_z[:, None, :] - z[None, :, :])**2
#     d = np.sum(d, axis=2)
#     d = np.exp(-1 / (2 * sigma**2) * d)
#     y = np.einsum('ij,jd->id', d, x)
#     y = y / np.sum(d, axis=1, keepdims=True) * 255
#     y = y.reshape(28, 28)
#     pil_img = Image.fromarray(y.astype(np.uint8))
#     stream = io.BytesIO()
#     pil_img.save(stream, format='PNG')  # 画像をPNG形式で保存する場合
#     image_base64 = base64.b64encode(stream.getvalue()).decode('utf-8')
#     return {"image": image_base64}