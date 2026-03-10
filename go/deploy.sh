#!/bin/bash
# deploy.sh - Build, version, deploy, commit, push. No excuses, no shortcuts.
#
# Usage:
#   ./deploy.sh patch "Fixed the thing"      # 2.4.0 -> 2.4.1
#   ./deploy.sh minor "Added new feature"    # 2.4.0 -> 2.5.0
#   ./deploy.sh major "Breaking change"      # 2.4.0 -> 3.0.0
#
# There is no "build only" option. Every deploy gets a version.

set -e

GOBIN="/usr/local/go/bin/go"
MAIN="main.go"
CHANGELOG="CHANGELOG.md"
OUT="../cogflow.exe"

BUMP="${1:-}"
DESC="$2"

if [ -z "$BUMP" ]; then
    echo "ERROR: Specify bump type."
    echo "Usage: ./deploy.sh patch|minor|major \"What changed\""
    exit 1
fi

if [ -z "$DESC" ]; then
    echo "ERROR: Need a description."
    echo "Usage: ./deploy.sh $BUMP \"What changed\""
    exit 1
fi

# Extract current version
CURRENT=$(grep 'version = "' "$MAIN" | head -1 | sed 's/.*"\(.*\)".*/\1/')
if [ -z "$CURRENT" ]; then
    echo "ERROR: Can't read version from $MAIN"
    exit 1
fi

echo "Current: v$CURRENT"

IFS='.' read -r MAJ MIN PAT <<< "$CURRENT"
case "$BUMP" in
    patch) PAT=$((PAT + 1)) ;;
    minor) MIN=$((MIN + 1)); PAT=0 ;;
    major) MAJ=$((MAJ + 1)); MIN=0; PAT=0 ;;
    *) echo "ERROR: Use patch, minor, or major"; exit 1 ;;
esac
NEW="$MAJ.$MIN.$PAT"

echo "Bumping: v$CURRENT -> v$NEW"

# Update version in main.go
sed -i "s/version = \"$CURRENT\"/version = \"$NEW\"/" "$MAIN"

# Prepend to CHANGELOG
HEADER="## v$NEW"
ENTRY="- $DESC"
sed -i "/^# Changelog$/a\\\\n$HEADER\n$ENTRY" "$CHANGELOG"

echo "Updated $MAIN and $CHANGELOG"

# Build
echo "Building v$NEW..."
GOOS=windows GOARCH=amd64 "$GOBIN" build -ldflags="-s -w" -o "$OUT" .
SIZE=$(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT" 2>/dev/null)
echo "Built: $OUT ($(( SIZE / 1024 ))KB)"

# Deploy
echo "Deploying..."
FAIL=0

deploy_to() {
    local host="$1"
    local name="$2"
    if scp "$OUT" "$host:C:/Users/jwill/Projects/cogflow-new.exe" 2>/dev/null; then
        if ssh "$host" "move /y C:\\Users\\jwill\\Projects\\cogflow-new.exe C:\\Users\\jwill\\Projects\\cogflow.exe" 2>/dev/null; then
            echo "  $name: OK"
        else
            echo "  $name: FAILED (file locked? kill cogflow first)"
            FAIL=1
        fi
    else
        echo "  $name: UNREACHABLE"
        FAIL=1
    fi
}

deploy_to "master-pc" "Master PC"
deploy_to "devbox" "Dev Laptop"

# SHA256 checksum
SHA_FILE="${OUT}.sha256"
sha256sum "$OUT" | awk '{print $1}' > "$SHA_FILE"
SHA=$(cat "$SHA_FILE")
echo "SHA256: $SHA"

# Commit + push
echo ""
echo "Committing..."
cd "$(dirname "$0")"
git add main.go CHANGELOG.md deploy.sh
git commit -m "v$NEW: $DESC

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
echo "Committed: v$NEW"

echo "Pushing..."
git push origin HEAD
echo "Pushed."

# GitHub release
echo ""
echo "Creating GitHub release v$NEW..."
REPO=$(git remote get-url origin 2>/dev/null | sed 's/.*github.com[:/]\(.*\)\.git/\1/' | sed 's/.*github.com[:/]\(.*\)/\1/')
if [ -n "$REPO" ] && command -v gh &>/dev/null; then
    git tag "v$NEW"
    git push origin "v$NEW" 2>/dev/null || true
    gh release create "v$NEW" \
        --title "v$NEW" \
        --notes "$DESC" \
        "$OUT#cogflow.exe" \
        "$SHA_FILE#cogflow.exe.sha256" \
        2>/dev/null && echo "Release created: v$NEW" || echo "Release creation failed (push manually)"
else
    echo "Skipped release (no gh CLI or no remote)"
fi

echo ""
if [ $FAIL -eq 0 ]; then
    echo "v$NEW deployed to all machines."
else
    echo "v$NEW deployed with errors (see above)."
fi
