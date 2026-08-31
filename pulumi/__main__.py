"""
Pulumi Infrastructure-as-Code (Python) for RAG-Lakehouse on GCP
Author: Raja Chakraborty

Deploys serverless Google Cloud Run v2 (scales to zero for $0 idle cost),
Google Cloud Storage Bucket for document data lake, and IAM service accounts.
"""

import pulumi
import pulumi_gcp as gcp

# Load Config
config = pulumi.Config()
gcp_config = pulumi.Config("gcp")
gcp_project = gcp_config.get("project") or "my-gcp-project"
gcp_region = gcp_config.get("region") or "us-central1"
app_name = config.get("app_name") or "rag-lakehouse"
container_image = config.get("container_image") or f"gcr.io/{gcp_project}/{app_name}:latest"

# -----------------------------------------------------------------------------
# 1. Google Cloud Storage Bucket (Document Data Lake)
# -----------------------------------------------------------------------------
lakehouse_bucket = gcp.storage.Bucket(
    "lakehouse-bucket",
    name=f"{app_name}-docs-lakehouse",
    location=gcp_region.upper(),
    uniform_bucket_level_access=True,
    versioning=gcp.storage.BucketVersioningArgs(enabled=True),
    force_destroy=True,  # Safe cleanup for demo envs
)

# -----------------------------------------------------------------------------
# 2. Service Account for Cloud Run Least-Privilege IAM
# -----------------------------------------------------------------------------
service_account = gcp.serviceaccount.Account(
    "cloud-run-sa",
    account_id=f"{app_name}-sa",
    display_name=f"Service Account for {app_name} Cloud Run",
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
            gcp.cloudrunv2.ServiceTemplateContainerArgs(
                image=container_image,
                resources=gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                    limits={
                        "memory": "1Gi",
                        "cpu": "1000m",
                    }
                ),
                envs=[
                    gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name="GCP_PROJECT", value=gcp_project),
                    gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name="GCP_REGION", value=gcp_region),
                    gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name="GCS_BUCKET_NAME", value=lakehouse_bucket.name),
                    gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name="VECTOR_DB_DIR", value="/tmp/chroma_storage"),
                ],
                ports=[
                    gcp.cloudrunv2.ServiceTemplateContainerPortArgs(container_port=8000)
                ],
            )
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
# Exports
# -----------------------------------------------------------------------------
pulumi.export("gcs_bucket_name", lakehouse_bucket.name)
pulumi.export("cloud_run_service_name", cloud_run_service.name)
pulumi.export("cloud_run_url", cloud_run_service.uri)
pulumi.export("service_account_email", service_account.email)
