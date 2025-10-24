import json, os
from datetime import datetime
import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import (
    AWS_REGION,
    S3_INPUT_BUCKET, S3_OUTPUT_BUCKET,
    RUNS_PREFIX, INPUT_PREFIX,
    SIGNED_TTL, LAMBDA_NAME
)

# ── boto3 clients ────────────────────────────────────────────────────────────
session   = boto3.session.Session(region_name=AWS_REGION)
s3        = session.client("s3", config=BotoConfig(signature_version="s3v4"))
lambda_cl = session.client("lambda") if (LAMBDA_NAME) else None

# ── helpers ─────────────────────────────────────────────────────────────────
def kjoin(*parts) -> str:
    return "/".join([p.strip("/") for p in parts if p])

def gen_run_id() -> str:
    from datetime import datetime
    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    rnd   = os.urandom(3).hex()
    return f"{stamp}-{rnd}"

# 기본은 '출력 버킷' (UI에서 읽는 대상)
def presign_get(key: str, ttl: int = SIGNED_TTL, *, bucket: str = S3_OUTPUT_BUCKET) -> str | None:
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl
        )
    except ClientError:
        return None

def s3_exists(key: str, *, bucket: str = S3_OUTPUT_BUCKET) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False

def s3_read_json(key: str, *, bucket: str = S3_OUTPUT_BUCKET):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        import json as _json
        return _json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None

def s3_list_runs(prefix: str = RUNS_PREFIX, *, bucket: str = S3_OUTPUT_BUCKET) -> list[str]:
    runs = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            runs.append(cp["Prefix"].rstrip("/").split("/")[-1])
    return sorted(runs, reverse=True)

# 업로드 기본은 '입력 버킷'
def s3_upload_fileobj(fileobj, key: str, *, bucket: str = S3_INPUT_BUCKET):
    s3.upload_fileobj(fileobj, bucket, key)

# JSON 쓰기는 출력 버킷에
def s3_put_json(key: str, payload: dict, *, bucket: str = S3_OUTPUT_BUCKET):
    s3.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json"
    )

def lambda_invoke(function_name: str, payload: dict):
    if not lambda_cl:
        raise RuntimeError("lambda client is not configured.")
    return lambda_cl.invoke(
        FunctionName=function_name,
        InvocationType="Event",
        Payload=json.dumps(payload).encode("utf-8")
    )

def s3_get_text(key: str, *, bucket: str = S3_OUTPUT_BUCKET) -> str | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8")
    except Exception:
        return None
