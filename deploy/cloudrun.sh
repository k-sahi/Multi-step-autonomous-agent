#!/usr/bin/env bash
# Deploy the API and the MCP tool server to Cloud Run.
# Prereqs: gcloud auth login; a Cloud SQL Postgres instance; secrets created in Secret Manager:
#   anthropic-api-key, operator-token, checkpoint-url (postgresql://...), arize-api-key, arize-space-id
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT}
REGION=${REGION:-us-central1}
REPO=${REPO:-support-agent}
SQL_INSTANCE=${SQL_INSTANCE:?set SQL_INSTANCE (project:region:instance)}
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/support-agent:$(git rev-parse --short HEAD)"

gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION" --project "$PROJECT"

gcloud builds submit --tag "$IMAGE" --project "$PROJECT"

# 1) MCP tool server: internal ingress only, so it is unreachable from the internet; the API reaches
#    it through Direct VPC egress. (For defence in depth, also require IAM and send an ID token.)
gcloud run deploy support-tools --image "$IMAGE" --region "$REGION" --project "$PROJECT" \
  --command python --args=-m,supportagent.tools.mcp_server,--http,--port,8080 \
  --ingress internal --allow-unauthenticated --min-instances 0 --max-instances 5

TOOLS_URL=$(gcloud run services describe support-tools --region "$REGION" --project "$PROJECT" --format 'value(status.url)')

# 2) API: concurrency kept modest because each request may hold an LLM call for seconds.
#    Checkpoints live in Cloud SQL, so any instance can resume a paused approval.
gcloud run deploy support-agent-api --image "$IMAGE" --region "$REGION" --project "$PROJECT" \
  --add-cloudsql-instances "$SQL_INSTANCE" \
  --set-env-vars "TOOLS_MCP_URL=$TOOLS_URL/mcp,APPROVAL_MODE=human,ARIZE_PROJECT=support-agent" \
  --set-secrets "ANTHROPIC_API_KEY=anthropic-api-key:latest,OPERATOR_TOKEN=operator-token:latest,CHECKPOINT_URL=checkpoint-url:latest,ARIZE_API_KEY=arize-api-key:latest,ARIZE_SPACE_ID=arize-space-id:latest" \
  --network default --subnet default --vpc-egress all-traffic \
  --concurrency 20 --timeout 300 --cpu 1 --memory 1Gi --min-instances 1 --max-instances 10 \
  --allow-unauthenticated

echo "API: $(gcloud run services describe support-agent-api --region "$REGION" --project "$PROJECT" --format 'value(status.url)')"
