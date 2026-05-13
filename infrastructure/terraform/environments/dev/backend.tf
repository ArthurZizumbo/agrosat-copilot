# Remote state — GCS with object versioning enabled.
# Bucket is provisioned by the gcp module on first apply (use -backend=false initially
# or pre-create via `gsutil mb -l europe-west1 -b on gs://agrosat-tfstate && gsutil versioning set on gs://agrosat-tfstate`).
terraform {
  backend "gcs" {
    bucket = "agrosat-tfstate"
    prefix = "dev"
  }
}
