from fastapi import FastAPI
from service.api.api import main_router
import onnxruntime as rt

app = FastAPI(project_name="emotion-detection")

app.include_router(main_router)

providers = ['CPUExecutionProvider']
m_q = rt.InferenceSession('lenet_quantized',providers=providers)
@app.get('/')
async def root():
    return {'hello':'word'}