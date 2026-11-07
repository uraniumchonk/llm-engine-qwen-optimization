#!/usr/bin/env bash
# Check that a version folder's patches apply with zero fuzz to the upstream tag.
#
# Usage: scripts/check-upstream.sh <version dir>
# Reads repo= and tag= from <version dir>/UPSTREAM, downloads every file the patches touch
# from raw.githubusercontent.com at that tag into a temporary directory, applies `series`
# there with scripts/apply.sh, then applies each test-only/*.patch on top. Nothing outside
# the temporary directory changes.
# Needs network access to raw.githubusercontent.com. A fetch that fails for any reason
# other than HTTP 404 prints NOT CHECKED and exits 3: report the series as not checked.
set -u
DIR=$(cd "${1:?version dir}" && pwd) || exit 2
HERE=$(cd "$(dirname "$0")" && pwd)
repo=$(sed -n 's/^repo=//p' "$DIR/UPSTREAM"); tag=$(sed -n 's/^tag=//p' "$DIR/UPSTREAM")
[ -n "$repo" ] && [ -n "$tag" ] || { echo "UPSTREAM needs repo= and tag="; exit 2; }
command -v curl >/dev/null || { echo "needs curl"; exit 2; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
patches=$(grep -v -E '^(#|$)' "$DIR/series" | sed "s#^#$DIR/#")
tests=$(ls "$DIR"/test-only/*.patch 2>/dev/null || true)

echo "== $repo@$tag"
# shellcheck disable=SC2086
for f in $(grep -h '^+++ b/' $patches $tests | sed 's#^+++ b/##' | sort -u); do
  mkdir -p "$TMP/tree/$(dirname "$f")"
  code=$(curl -sL -o "$TMP/tree/$f" -w '%{http_code}' "https://raw.githubusercontent.com/$repo/$tag/$f") || code=000
  case "$code" in
    200) ;;
    404) rm -f "$TMP/tree/$f"; echo "   (not in $tag, the patch creates it: $f)" ;;
    *) echo "NOT CHECKED: fetching $f returned HTTP $code (no network or GitHub unreachable)"; exit 3 ;;
  esac
done

"$HERE/apply.sh" "$DIR" "$TMP/tree" || exit 1
for t in $tests; do
  if patch -d "$TMP/tree" -p1 -F0 -N -s --dry-run <"$t" >/dev/null 2>&1; then echo "OK       test-only/$(basename "$t")"
  else echo "CONFLICT test-only/$(basename "$t")"; exit 1; fi
done
