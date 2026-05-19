# csp-scanner-trigger (Cloudflare Worker)

Password-protected proxy that triggers the `scan-ondemand.yml` GitHub Actions workflow.
The GitHub PAT never leaves the Worker.

## Deploy

```bash
# one-time
npm install -g wrangler
wrangler login

# from this directory
wrangler secret put PASSWORD     # paste your chosen shared password
wrangler secret put GH_TOKEN     # paste a fine-grained PAT (see below)

wrangler deploy
```

After `wrangler deploy` you'll get a URL like
`https://csp-scanner-trigger.<your-subdomain>.workers.dev`.
Paste that into `static/trigger.html` (`WORKER_URL` constant) and push.

## GitHub PAT

Create a **fine-grained personal access token** at
`https://github.com/settings/personal-access-tokens/new`:

- Resource owner: your account
- Repository access: **Only select repositories** → `csp-scanner`
- Permissions → Repository permissions → **Actions: Read and write**
- Expiration: as long as you're comfortable with (rotate periodically)

## Test

```bash
curl -X POST https://csp-scanner-trigger.<sub>.workers.dev \
  -H 'content-type: application/json' \
  -d '{"password":"your-pass","workflow":"scan-ondemand.yml"}'
```

Expected: `{"ok":true,"workflow":"scan-ondemand.yml","ref":"main"}`

## Rotate password / token

```bash
wrangler secret put PASSWORD   # overwrites
wrangler secret put GH_TOKEN
```
