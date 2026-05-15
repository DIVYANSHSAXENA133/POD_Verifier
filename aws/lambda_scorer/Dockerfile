FROM public.ecr.aws/lambda/python:3.12

# CPU-only PyTorch (+ torchvision for timm); reference pattern: PyTorch CPU index, not CUDA PyPI builds.
RUN pip install --no-cache-dir \
    torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY model/best.pt /opt/model/best.pt
COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
