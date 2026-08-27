#!/usr/bin/env bash
# Install the R package CLVTools into a project-local library (.Rlib/).
#
# The test suite does not need this: oracle fixtures are committed under
# tests/fixtures/. This script is only needed to *regenerate* those fixtures,
# or to check a new expectation against the reference implementation.
#
# The install is confined to .Rlib/ and never touches the user's R library.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLIB="$ROOT/.Rlib"
CACHE="$ROOT/.oracle-cache"
VERSION="${CLVTOOLS_VERSION:-0.12.1}"

mkdir -p "$RLIB" "$CACHE"

# CRAN's macOS binaries lag the newest R release, but a binary built for the
# previous minor version installs and runs correctly on the current one. Try
# each recent R series and take the first that resolves.
fetch_binary() {
  local arch series url
  arch="$(uname -m)"
  [ "$arch" = "arm64" ] && arch="big-sur-arm64" || arch="big-sur-x86_64"
  for series in 4.6 4.5 4.4; do
    url="https://cran.r-project.org/bin/macosx/$arch/contrib/$series/CLVTools_$VERSION.tgz"
    if curl -sfL --retry 3 --max-time 300 -o "$CACHE/CLVTools_bin.tgz" "$url"; then
      echo "$CACHE/CLVTools_bin.tgz"
      return 0
    fi
  done
  return 1
}

if [ ! -d "$RLIB/CLVTools" ]; then
  echo "==> fetching CLVTools $VERSION"
  if [ "$(uname -s)" = "Darwin" ] && PKG="$(fetch_binary)"; then
    echo "==> installing binary package"
  else
    # Source install needs GSL and OpenMP headers (see src/Makevars).
    echo "==> no binary available; building from source (needs gsl + libomp)"
    PKG="$CACHE/CLVTools_$VERSION.tar.gz"
    curl -fL --retry 3 --max-time 600 -o "$PKG" \
      "https://cran.r-project.org/src/contrib/CLVTools_$VERSION.tar.gz"
  fi
  R_LIBS="$RLIB" R CMD INSTALL -l "$RLIB" "$PKG"
fi

echo "==> resolving dependencies"
R_LIBS="$RLIB" Rscript -e '
  deps <- c("data.table","digest","Formula","ggplot2","lubridate",
            "numDeriv","Matrix","MASS","optimx","Rcpp")
  miss <- setdiff(deps, rownames(installed.packages()))
  if (length(miss)) {
    install.packages(miss, lib = Sys.getenv("R_LIBS"),
                     repos = "https://cloud.r-project.org", type = "binary")
  }
'

echo "==> verifying"
R_LIBS="$RLIB" Rscript -e '
  suppressMessages(library(CLVTools))
  cat("CLVTools", as.character(packageVersion("CLVTools")), "ready\n")
'
