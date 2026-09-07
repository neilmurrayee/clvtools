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

# Fetch first, so that what gets installed decides what has to be resolved: a
# macOS binary needs only CLVTools' runtime imports, while a source build also
# needs everything its `LinkingTo` and its tests name.
NEED_BUILD_DEPS=1
if [ ! -d "$RLIB/CLVTools" ]; then
  echo "==> fetching CLVTools $VERSION"
  if [ "$(uname -s)" = "Darwin" ] && PKG="$(fetch_binary)"; then
    echo "==> installing binary package"
    NEED_BUILD_DEPS=0
  else
    # Source install needs GSL and OpenMP headers (see src/Makevars).
    echo "==> no binary available; building from source (needs gsl + libomp)"
    PKG="$CACHE/CLVTools_$VERSION.tar.gz"
    curl -fL --retry 3 --max-time 600 -o "$PKG" \
      "https://cran.r-project.org/src/contrib/CLVTools_$VERSION.tar.gz"
  fi
fi

# Dependencies before the install, not after. `R CMD INSTALL` resolves nothing
# itself: it reads DESCRIPTION, finds a required package missing, stops, and
# deletes the half-installed library on its way out. On a machine where the
# dependencies happen to be present already the ordering never shows, which is
# how it survived here until a Linux runner ran this on an empty .Rlib and
# reported `dependencies ... are not available for package 'CLVTools'`.
echo "==> resolving dependencies"
R_LIBS="$RLIB" NEED_BUILD_DEPS="$NEED_BUILD_DEPS" Rscript -e '
  deps <- c("data.table","digest","Formula","ggplot2","lubridate",
            "numDeriv","Matrix","MASS","optimx","Rcpp")
  # A source build additionally needs what CLVTools LinkingTo and tests with;
  # `R CMD INSTALL` demands these even though nothing here calls them.
  if (Sys.getenv("NEED_BUILD_DEPS") == "1") {
    deps <- c(deps, "RcppArmadillo", "RcppGSL", "testthat")
  }

  # `repos` defaults to the sentinel "@CRAN@", which is not a URL and sends
  # install.packages looking for a mirror it has not been given -- so test for
  # the sentinel rather than for absence. A runner that set RSPM already has a
  # real value here and keeps it.
  repos <- getOption("repos")
  if (is.null(repos) || !("CRAN" %in% names(repos)) ||
      !nzchar(repos[["CRAN"]]) || identical(unname(repos[["CRAN"]]), "@CRAN@")) {
    repos <- c(CRAN = "https://cloud.r-project.org")
  }

  # `type = "binary"` is an error on Linux rather than a fallback, so ask for it
  # only where it exists. On a Linux runner the repository is Posit`s RSPM,
  # which serves precompiled builds through source URLs, so the default type is
  # no slower there. Braced because at top level R ends a statement at the
  # first complete line, and an `else` opening the next one is a syntax error.
  type <- if (Sys.info()[["sysname"]] %in% c("Darwin", "Windows")) {
    "binary"
  } else {
    getOption("pkgType")
  }

  miss <- setdiff(deps, rownames(installed.packages()))
  if (length(miss)) {
    cat("    installing:", paste(miss, collapse = " "), "\n")
    install.packages(miss, lib = Sys.getenv("R_LIBS"), repos = repos, type = type)
    still <- setdiff(deps, rownames(installed.packages()))
    if (length(still)) {
      stop("could not install: ", paste(still, collapse = ", "), call. = FALSE)
    }
  }
'

if [ ! -d "$RLIB/CLVTools" ]; then
  R_LIBS="$RLIB" R CMD INSTALL -l "$RLIB" "$PKG"
fi

echo "==> verifying"
R_LIBS="$RLIB" Rscript -e '
  suppressMessages(library(CLVTools))
  cat("CLVTools", as.character(packageVersion("CLVTools")), "ready\n")
'
