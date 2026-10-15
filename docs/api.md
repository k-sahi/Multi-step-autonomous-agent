# API reference

Base URL `http://localhost:8080`; OpenAPI docs at `/docs`.

| Method | Path | Auth | Body | Response |
|---|---|---|---|---|
| GET | `/healthz` | — | — | `{"ok", "anthropic", "sessions"}` |
| POST | `/v1/sessions` | — | `{"domain": "retail"}` | `201 {"thread_id"}` |
| POST | `/v1/sessions/{id}/messages` | — | `{"text"}` | `TurnResult`; `409` while an approval is pending |
| GET | `/v1/sessions/{id}` | — | — | status, customer-visible messages, approvals, pending approval |
| GET | `/v1/approvals` | operator | — | `{"pending": [{thread_id, actions[], context}]}` |
| POST | `/v1/approvals/{id}` | operator | `{"decisions": {call_id: {"approved", "reason"}}}` | `TurnResult`; `422` unless every pending action is decided |
| GET | `/v1/sessions/{id}/trace` | operator | — | trace records |

Operator auth: headers `X-Operator-Token` (matches `OPERATOR_TOKEN`) and `X-Reviewer` (recorded).

`TurnResult`:
```json
{"thread_id": "…", "status": "awaiting_user | awaiting_approval | transferred | escalated",
 "reply": "text for the customer or null",
 "approval_request": {"actions": [{"call_id", "tool", "arguments", "risk"}], "context": "…"},
 "new_messages": [...], "guard_events": [...]}
```

## Example

```bash
T=$(curl -s -XPOST localhost:8080/v1/sessions -H 'content-type: application/json' -d '{}' | jq -r .thread_id)
curl -s localhost:8080/v1/sessions/$T/messages -H 'content-type: application/json' \
  -d '{"text":"Hi, I am yusuf.rossi7301@example.com, cancel order #W2378156, I no longer need it"}' | jq .reply
curl -s localhost:8080/v1/sessions/$T/messages -H 'content-type: application/json' -d '{"text":"yes"}' | jq .status
# -> "awaiting_approval"
H='-H X-Operator-Token:change-me -H X-Reviewer:alice'
CID=$(curl -s $H localhost:8080/v1/approvals | jq -r '.pending[0].actions[0].call_id')
curl -s $H -XPOST localhost:8080/v1/approvals/$T -H 'content-type: application/json' \
  -d "{\"decisions\":{\"$CID\":{\"approved\":true,\"reason\":\"verified\"}}}" | jq .reply
```
