import os

AWS_REGION      = os.getenv("AWS_REGION",  "us-east-1")
S3_BUCKET       = os.getenv("S3_BUCKET",   "chaen-nextflow-data")
RUNS_PREFIX     = os.getenv("RUNS_PREFIX", "runs/")
SIGNED_TTL      = int(os.getenv("SIGNED_TTL", "3600"))

S3_INPUT_BUCKET  = os.getenv("S3_INPUT_BUCKET",  S3_BUCKET)
S3_OUTPUT_BUCKET = os.getenv("S3_OUTPUT_BUCKET", S3_BUCKET)

# Lambda (옵션)
LAMBDA_NAME     = os.getenv("LAMBDA_NAME", "start")        # 전체 파이프라인 시작용

INPUT_PREFIX     = os.getenv("INPUT_PREFIX", "input/")  # 입력(FASTA)용
RUNS_PREFIX      = os.getenv("RUNS_PREFIX",  "runs/")    # 출력(run)용





