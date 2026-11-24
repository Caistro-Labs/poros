#!/usr/bin/env bash
# Apply the public repo's settings and rules with gh. A maintainer runs this on the new, empty
# repo before the first export is pushed, and again whenever the rules change; running it twice
# changes nothing.
#
#   scripts/repo_rules.sh [OWNER/REPO]            dry run: print every call and its JSON
#   scripts/repo_rules.sh [OWNER/REPO] --apply    make the calls
#
# OWNER/REPO defaults to $POROS_PUBLIC_REPO, then Caistro-Labs/poros. Needs gh logged in as a
# repo admin, and jq. A dry run only reads (the rulesets that exist, to update them by name).
set -euo pipefail
cd "$(dirname "$0")/.."

repo="${POROS_PUBLIC_REPO:-Caistro-Labs/poros}"
apply=false
for arg in "$@"; do
  case "$arg" in
    --apply) apply=true ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    */*) repo="$arg" ;;
    *) echo "usage: scripts/repo_rules.sh [OWNER/REPO] [--apply]" >&2; exit 2 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }
command -v jq >/dev/null || die "install jq"
command -v gh >/dev/null || die "install gh"

# The required checks are these jobs' names, as the workflows spell them.
REQUIRED=("Check" "Lint" "Workflow lint")
for check in "${REQUIRED[@]}"; do
  grep -qE "^    name: ${check}\$" .github/workflows/*.yml ||
    die "no job named '$check' in .github/workflows; update REQUIRED"
done
# GitHub Actions' app id: a required check passes only from an Actions run, never from a
# commit status anyone with write access could post.
ACTIONS_APP_ID=15368
# The repository admin role, as a ruleset bypass actor.
ADMIN='{"actor_id": 5, "actor_type": "RepositoryRole"}'

if $apply; then
  echo "Applying to $repo."
else
  echo "Dry run for $repo. Nothing changes; rerun with --apply to make these calls."
fi

# call METHOD PATH JSON [NOTE]: print the call; with --apply make it. NOTE, when given, is
# printed instead of failing when GitHub refuses (a setting the plan does not include).
call() {
  local method="$1" path="$2" body="$3" note="${4:-}"
  echo
  echo "$method /$path"
  [[ -n "$body" ]] && jq . <<< "$body"
  $apply || return 0
  local out
  if [[ -n "$body" ]]; then
    out="$(gh api -X "$method" "$path" --input - <<< "$body" 2>&1)" && { echo "ok"; return 0; }
  else
    out="$(gh api -X "$method" "$path" 2>&1)" && { echo "ok"; return 0; }
  fi
  if [[ -n "$note" ]]; then
    echo "skipped: $note"
    echo "  GitHub said: $(head -c 300 <<< "$out")"
  else
    die "$method /$path failed: $out"
  fi
}

# 1. Only collaborators can open pull requests: the public files issues, hardware reports and
# Discussions. Public main takes only the maintainers' export commits, pushed directly: no PRs and
# no merges, so no merge settings either.
call PATCH "repos/$repo" "$(jq -n '{
  pull_request_creation_policy: "collaborators_only"
}')"

# 2. Discussions on (gh's own flag; the REST description has no field for it).
echo
echo "gh repo edit $repo --enable-discussions"
$apply && gh repo edit "$repo" --enable-discussions && echo "ok"

# 3. Rulesets, each updated by name when it exists, so a second run adds none.
ruleset_id() {
  local ids
  ids="$(gh api "repos/$repo/rulesets?per_page=100" \
    --jq ".[] | select(.name == \"$1\") | .id" 2>/dev/null)" || return 0
  [[ "$(head -n1 <<< "$ids")" =~ ^[0-9]+$ ]] && head -n1 <<< "$ids"
  return 0
}
put_ruleset() {
  local name="$1" body="$2" id
  id="$(ruleset_id "$name")"
  if [[ -n "$id" ]]; then
    call PUT "repos/$repo/rulesets/$id" "$body"
  else
    $apply || echo "(no ruleset named '$name' found, so it is created)"
    call POST "repos/$repo/rulesets" "$body"
  fi
}

checks="$(printf '%s\n' "${REQUIRED[@]}" |
  jq -R --argjson app "$ACTIONS_APP_ID" '{context: ., integration_id: $app}' | jq -s .)"
# main is never rewritten: no deletion and no force push, by anyone, an admin included.
put_ruleset "main: never rewritten" "$(jq -n '{
  name: "main: never rewritten",
  target: "branch",
  enforcement: "active",
  bypass_actors: [],
  conditions: {ref_name: {include: ["refs/heads/main"], exclude: []}},
  rules: [{type: "deletion"}, {type: "non_fast_forward"}]
}')"

# The checks CI runs on main. A maintainer's export push lands before they run (an admin
# bypasses), and CI still runs on it; anyone else's push to main needs them to pass.
put_ruleset "main: checks" "$(jq -n --argjson admin "$ADMIN" --argjson checks "$checks" '{
  name: "main: checks",
  target: "branch",
  enforcement: "active",
  bypass_actors: [$admin + {bypass_mode: "always"}],
  conditions: {ref_name: {include: ["refs/heads/main"], exclude: []}},
  rules: [{type: "required_status_checks", parameters: {
    strict_required_status_checks_policy: false,
    required_status_checks: $checks
  }}]
}')"

put_ruleset "release tags" "$(jq -n --argjson admin "$ADMIN" '{
  name: "release tags",
  target: "tag",
  enforcement: "active",
  bypass_actors: [$admin + {bypass_mode: "always"}],
  conditions: {ref_name: {include: ["refs/tags/v*"], exclude: []}},
  rules: [{type: "creation"}, {type: "update"}, {type: "deletion"}]
}')"

# 4. Actions: the default token reads only, Actions can't create or approve PRs, and every
# outside contributor's fork PR waits for approval before its workflows run.
call PUT "repos/$repo/actions/permissions/workflow" "$(jq -n '{
  default_workflow_permissions: "read",
  can_approve_pull_request_reviews: false
}')"
call PUT "repos/$repo/actions/permissions/fork-pr-contributor-approval" "$(jq -n '{
  approval_policy: "all_external_contributors"
}')"

# 5. Secret scanning with push protection, and private vulnerability reporting.
call PATCH "repos/$repo" "$(jq -n '{security_and_analysis: {
  secret_scanning: {status: "enabled"},
  secret_scanning_push_protection: {status: "enabled"}
}}')" "secret scanning is not in this repo's plan; turn it on under Settings, Advanced Security"
call PUT "repos/$repo/private-vulnerability-reporting" "" \
  "private vulnerability reporting is not available here; turn it on under Settings, Advanced Security"

echo
$apply && echo "Done." || echo "Dry run done. Nothing was changed."
