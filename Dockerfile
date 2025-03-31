FROM public.ecr.aws/lambda/python:3.9

# Set the working directory
WORKDIR /var/task

# Copy the requirements file from the backend folder
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('./model')"

# Verify the model folder exists
RUN ls -l ./model

# Copy only the contents of the backend folder
COPY backend/*.py .

# Expose port 8080 for local testing
EXPOSE 8080

# Set the Lambda handler for AWS Lambda
CMD ["lambda_handler.lambda_handler"]