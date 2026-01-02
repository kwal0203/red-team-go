# Artifact Storage (S3 + IAM)

This project can persist evaluation artifacts (requests, responses, provenance metadata) to local disk or S3.

## Configuration
Set the following environment variables:

- `ARTIFACT_STORAGE_MODE`: `disabled` (default), `local`, or `s3`
- `ARTIFACT_LOCAL_DIR`: local directory for artifacts (default: `artifacts`)
- `ARTIFACT_S3_BUCKET`: S3 bucket name (required for `s3` mode)
- `ARTIFACT_S3_PREFIX`: key prefix (default: `redteamgo`)
- `ARTIFACT_AWS_REGION`: optional AWS region for S3 client

## What Gets Stored
Each artifact is a JSON document containing:
- `metadata`: timestamp, endpoint, evaluation type, model info, client IP, user agent
- `request`: sanitized request payload
- `response`: sanitized response payload

Secrets (keys/tokens/passwords) are scrubbed before storage.

## IAM Policy Example (Least Privilege)
Attach a policy like the following to the ECS task role or instance profile:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowArtifactWrites",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:AbortMultipartUpload"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME/redteamgo/*"
    },
    {
      "Sid": "AllowBucketRead",
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLocation",
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME"
    }
  ]
}
```

## Example
```bash
export ARTIFACT_STORAGE_MODE=s3
export ARTIFACT_S3_BUCKET=redteamgo-artifacts
export ARTIFACT_S3_PREFIX=redteamgo
```

Artifacts will be stored under keys like:
```
redteamgo/toxicity_batch/2025-01-02/<artifact_id>.json
```
