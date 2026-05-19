/**
 * Cloudflare Worker: password-protected proxy that triggers a GitHub Actions
 * workflow_dispatch.
 *
 * Secrets (set via `wrangler secret put`):
 *   PASSWORD  — shared secret for the trigger UI
 *   GH_TOKEN  — GitHub fine-grained PAT with "Actions: read & write" on the repo
 *
 * Vars (wrangler.toml):
 *   GH_REPO   — "owner/repo", e.g. "dinoel/csp-scanner"
 */

const ALLOWED_WORKFLOWS = new Set([
  "scan-ondemand.yml",
  "scan-scheduled.yml",
]);

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return cors(new Response(null, { status: 204 }));
    if (request.method !== "POST")    return json({ error: "POST only" }, 405);

    let body;
    try { body = await request.json(); }
    catch { return json({ error: "invalid JSON body" }, 400); }

    if (!env.PASSWORD || body.password !== env.PASSWORD) {
      return json({ error: "wrong password" }, 401);
    }

    const workflow = body.workflow || "scan-ondemand.yml";
    if (!ALLOWED_WORKFLOWS.has(workflow)) {
      return json({ error: `workflow not allowed: ${workflow}` }, 400);
    }

    const ref    = body.ref || "main";
    const inputs = body.inputs || {};

    const ghRes = await fetch(
      `https://api.github.com/repos/${env.GH_REPO}/actions/workflows/${workflow}/dispatches`,
      {
        method: "POST",
        headers: {
          "Authorization":         `Bearer ${env.GH_TOKEN}`,
          "Accept":                "application/vnd.github+json",
          "User-Agent":            "csp-scanner-trigger",
          "X-GitHub-Api-Version":  "2022-11-28",
        },
        body: JSON.stringify({ ref, inputs }),
      },
    );

    if (!ghRes.ok) {
      const text = await ghRes.text();
      return json({ error: "github api error", status: ghRes.status, body: text }, 502);
    }

    return json({ ok: true, workflow, ref });
  },
};

function json(obj, status = 200) {
  return cors(new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json" },
  }));
}

function cors(res) {
  res.headers.set("Access-Control-Allow-Origin",  "*");
  res.headers.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.headers.set("Access-Control-Allow-Headers", "content-type");
  return res;
}
