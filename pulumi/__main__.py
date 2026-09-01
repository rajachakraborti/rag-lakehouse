"""
Pulumi Python Infrastructure-as-Code for RAG-Lakehouse Engine
Author: Raja Chakraborty

Provisions:
1. Google Cloud Storage (GCS) Bucket for document lakehouse storage.
2. IAM Service Account with least-privilege security policy.
3. GCP Cloud Run v2 Service (Serverless, scales to 0 for $0 idle cost).
"""

import pulumi
import pulumi_gcp as gcp

# Load Pulumi Configuration
config = pulumi.Config()
gcp_project = config.get("gcp:project") or "stellar-horizon-166823"
gcp_region = config.get("gcp:region") or "us-central1"
app_name = config.get("app_name") or "rag-lakehouse-app"
container_image = config.get("container_image") or f"gcr.io/{gcp_project}/rag-lakehouse:latest"

# -----------------------------------------------------------------------------
# 1. Google Cloud Storage Bucket (Document Lakehouse Data Store)
# -----------------------------------------------------------------------------
lakehouse_bucket = gcp.storage.Bucket(
    "lakehouse-bucket",
    name=f"{app_name}-docs-lakehouse",
    location=gcp_region,
    force_destroy=True,
    uniform_bucket_level_access=True,
    versioning=gcp.storage.BucketVersioningArgs(enabled=True),
)

# -----------------------------------------------------------------------------
# 2. IAM Service Account for Cloud Run Application
# -----------------------------------------------------------------------------
service_account = gcp.serviceaccount.Account(
    "cloud-run-sa",
    account_id="rag-lakehouse-sa",
    display_name="RAG Lakehouse Service Account",
)

# Grant Storage Object Viewer to Service Account
bucket_iam = gcp.storage.BucketIAMMember(
    "bucket-iam-viewer",
    bucket=lakehouse_bucket.name,
    role="roles/storage.objectViewer",
    member=service_account.email.apply(lambda email: f"serviceAccount:{email}"),
)

# -----------------------------------------------------------------------------
# 3. Google Cloud Run v2 Service (Scales to 0 for zero idle cost)
# -----------------------------------------------------------------------------
cloud_run_service = gcp.cloudrunv2.Service(
    "rag-lakehouse-service",
    name=app_name,
    location=gcp_region,
    template=gcp.cloudrunv2.ServiceTemplateArgs(
        service_account=service_account.email,
        scaling=gcp.cloudrunv2.ServiceTemplateScalingArgs(
            min_instance_count=0,  # Scales to 0 when idle ($0 cost)
            max_instance_count=3,
        ),
        containers=[
            {
                "image": container_image,
                "resources": {
                    "limits": {
                        "memory": "1Gi",
                        "cpu": "1000m",
                    }
                },
                "envs": [
                    {"name": "GCP_PROJECT", "value": gcp_project},
                    {"name": "GCP_REGION", "value": gcp_region},
                    {"name": "GCS_BUCKET_NAME", "value": lakehouse_bucket.name},
                    {"name": "VECTOR_DB_DIR", "value": "/tmp/chroma_storage"},
                ],
                "ports": [
                    {"container_port": 8000}
                ],
            }
        ],
    ),
)

# Allow Unauthenticated Public Access to Cloud Run (for demo API testing)
public_invoker = gcp.cloudrunv2.ServiceIamMember(
    "public-invoker",
    name=cloud_run_service.name,
    location=gcp_region,
    role="roles/run.invoker",
    member="allUsers",
)

# -----------------------------------------------------------------------------
# Exports / Infrastructure Outputs
# -----------------------------------------------------------------------------
pulumi.export("gcs_bucket_name", lakehouse_bucket.name)
pulumi.export("cloud_run_url", cloud_run_service.uri)
pulumi.export("service_account_email", service_account.email)
