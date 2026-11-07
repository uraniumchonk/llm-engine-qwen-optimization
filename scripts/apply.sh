#!/usr/bin/env bash
# Apply one version folder's patches to a tree, in `series` order, with zero fuzz.
#
# Usage: scripts/apply.sh <version dir> <tree root> [--check]
#   <version dir>  a folder such as vllm/v0.33.0 (holds `series` and the .patch files)
#   <tree root>    the directory that contains vllm/ or lmcache/: a source checkout at the
#                  tag named in UPSTREAM, or the site-packages directory of the installed wheel
#   --check        report the tree's state and change nothing
#
# The whole series is first tried on a scratch copy of the files it touches, because later
# patches build on earlier ones (E006c edits lines E006a adds). Outcomes:
#   CLEAN     the series applies in order; apply mode then applies it to the tree
#   APPLIED   the tree already holds the whole series; nothing to do
#   CONFLICT  the series does not apply; the first failing patch is named, the tree is
#             unchanged. Fix it by hand against that patch's card in the folder's README.md.
set -u
DIR=$(cd "${1:?version dir}" && pwd) || exit 2
ROOT=$(cd "${2:?tree root}" && pwd) || exit 2
MODE=${3:-apply}
command -v patch >/dev/null || { echo "needs GNU patch"; exit 2; }
[ -f "$DIR/series" ] || { echo "no series file in $DIR"; exit 2; }

mapfile -t SERIES < <(grep -v -E '^(#|$)' "$DIR/series")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

p() { local d=$1; shift; patch -d "$d" -p1 -F0 --no-backup-if-mismatch -s "$@"; }

stage() { # copy every existing file the series touches into a fresh scratch tree
  rm -rf "$TMP/t"; mkdir -p "$TMP/t"
  local f
  for f in $(cd "$DIR" && grep -h '^+++ b/' "${SERIES[@]}" | sed 's#^+++ b/##' | sort -u); do
    if [ -e "$ROOT/$f" ]; then mkdir -p "$TMP/t/$(dirname "$f")"; cp -p "$ROOT/$f" "$TMP/t/$f"; fi
  done
}

forward() { # apply the series in order; print the first failing patch
  local n
  for n in "${SERIES[@]}"; do p "$1" -N <"$DIR/$n" >/dev/null 2>&1 || { echo "$n"; return 1; }; done
}

reverse() { # undo the series in reverse order
  local i
  for ((i = ${#SERIES[@]} - 1; i >= 0; i--)); do
    p "$1" -R <"$DIR/${SERIES[$i]}" >/dev/null 2>&1 || return 1
  done
}

stage
if failed=$(forward "$TMP/t"); then
  state=CLEAN
else
  stage
  if reverse "$TMP/t"; then state=APPLIED; else state=CONFLICT; fi
fi

case "$state" in
  APPLIED) echo "APPLIED  the tree already holds all ${#SERIES[@]} patches of $DIR"; exit 0 ;;
  CONFLICT) echo "CONFLICT $failed does not apply, and the tree does not hold the whole series either (a partly patched tree lands here too). Tree unchanged; see the card in $DIR/README.md"; exit 1 ;;
esac
if [ "$MODE" = --check ]; then
  echo "CLEAN    all ${#SERIES[@]} patches apply in order: ${SERIES[*]}"; exit 0
fi
for n in "${SERIES[@]}"; do
  if p "$ROOT" -N <"$DIR/$n"; then echo "OK       $n"; else echo "FAILED   $n (tree is partly patched)"; exit 1; fi
done
