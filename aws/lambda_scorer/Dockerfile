FROM public.ecr.aws/lambda/python:3.11

# Install system deps for OpenCV
RUN yum install -y mesa-libGL && yum clean all

# Install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy model source
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# Copy model checkpoint
COPY model/best.pt /opt/model/best.pt

# Copy handler
COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
