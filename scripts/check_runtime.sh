#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check_runtime.sh [--cpu|--qnn|--ort|--auto]

Checks the current shell environment for the Tachyon ANN runtime.

Modes:
  --auto  Infer what to check from the current environment (default)
  --cpu   Validate the CPU-only path
  --qnn   Validate the direct QNN/HTP path
  --ort   Validate the ONNX Runtime path
  --help  Show this help
USAGE
}

mode="auto"
case "${1-}" in
  ""|--auto) mode="auto" ;;
  --cpu) mode="cpu" ;;
  --qnn) mode="qnn" ;;
  --ort) mode="ort" ;;
  --help|-h) usage; exit 0 ;;
  *)
    echo "unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
esac

need_qnn=0
need_ort=0
case "$mode" in
  cpu) ;;
  qnn) need_qnn=1 ;;
  ort) need_ort=1 ;;
  auto)
    if [[ -n "${QNN_SDK_ROOT-}" || -n "${TACHANN_QNN_LIB-}" || -n "${ADSP_LIBRARY_PATH-}" || "${TACHANN_BACKEND-}" =~ ^(adaptive|qnn)$ ]]; then
      need_qnn=1
    fi
    if [[ -n "${ORT_DYLIB_PATH-}" || "${TACHANN_BACKEND-}" == "ort" ]]; then
      need_ort=1
    fi
    ;;
esac

failures=0
warnings=0

ok() { printf '[ok] %s\n' "$1"; }
warn() { printf '[warn] %s\n' "$1"; warnings=$((warnings + 1)); }
fail() { printf '[fail] %s\n' "$1"; failures=$((failures + 1)); }

require_env() {
  local key="$1"
  if [[ -n "${!key-}" ]]; then
    ok "$key is set"
  else
    fail "$key is not set"
  fi
}

require_path() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    ok "$label exists: $path"
  else
    fail "$label is missing: $path"
  fi
}

contains_path() {
  local haystack="$1"
  local needle="$2"
  [[ "$haystack" == *"$needle"* ]]
}

print_qnn_exports() {
  cat <<'HINT'
Suggested QNN exports:
  export QNN_SDK_ROOT=/path/to/qairt
  export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4:$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4/arch64
  export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v68/unsigned;/usr/lib/rfsa/adsp;/dsp"
  export TACHANN_QNN_LIB=./target/debug/libtachyon_qnnshim.so
  export TACHANN_QNN_BACKEND=htp
  export TACHANN_QNN_FP16=0
  export TACHANN_QNN_STATIC_A=1
HINT
}

echo "== Tachyon ANN runtime check =="
echo "mode: $mode"

if [[ "$mode" == "cpu" || ($need_qnn -eq 0 && $need_ort -eq 0) ]]; then
  ok "CPU path requires no external runtimes"
fi

if [[ $need_qnn -eq 1 ]]; then
  echo
  echo "-- QNN / HTP checks --"
  require_env QNN_SDK_ROOT
  require_env TACHANN_QNN_LIB
  require_path "QNN shim" "${TACHANN_QNN_LIB-}"

  qnn_root="${QNN_SDK_ROOT-}"
  host_lib_dir=""
  for candidate in \
    "$qnn_root/lib/aarch64-ubuntu-gcc9.4" \
    "$qnn_root/lib/aarch64-oe-linux-gcc9.3" \
    "$qnn_root/lib/aarch64-oe-linux-gcc11.2" \
    "$qnn_root/lib/aarch64-oe-linux-gcc8.2"; do
    if [[ -d "$candidate" ]]; then
      host_lib_dir="$candidate"
      break
    fi
  done

  if [[ -n "$host_lib_dir" ]]; then
    ok "QNN host library directory found: $host_lib_dir"
    for lib in libQnnSystem.so libQnnHtp.so libQnnHtpPrepare.so; do
      require_path "$lib" "$host_lib_dir/$lib"
    done
    stub="$(find "$host_lib_dir" -maxdepth 1 -name 'libQnnHtpV*Stub.so' -print -quit 2>/dev/null || true)"
    if [[ -n "$stub" ]]; then
      ok "HTP stub found: $stub"
    else
      fail "No libQnnHtpV*Stub.so found under $host_lib_dir"
    fi
    if [[ -n "${LD_LIBRARY_PATH-}" ]] && contains_path "$LD_LIBRARY_PATH" "$host_lib_dir"; then
      ok "LD_LIBRARY_PATH includes $host_lib_dir"
    else
      warn "LD_LIBRARY_PATH does not include $host_lib_dir"
    fi
  else
    fail "Could not find a supported QNN host library directory under QNN_SDK_ROOT"
  fi

  skel_dir="$qnn_root/lib/hexagon-v68/unsigned"
  require_path "DSP skel directory" "$skel_dir"
  skel="$(find "$skel_dir" -maxdepth 1 -name 'libQnnHtpV*Skel.so' -print -quit 2>/dev/null || true)"
  if [[ -n "$skel" ]]; then
    ok "HTP skel found: $skel"
  else
    fail "No libQnnHtpV*Skel.so found under $skel_dir"
  fi

  if [[ -n "${ADSP_LIBRARY_PATH-}" ]]; then
    ok "ADSP_LIBRARY_PATH is set"
    if contains_path "$ADSP_LIBRARY_PATH" "$skel_dir"; then
      ok "ADSP_LIBRARY_PATH includes $skel_dir"
    else
      fail "ADSP_LIBRARY_PATH does not include $skel_dir"
    fi
    if [[ -d /usr/lib/rfsa/adsp ]] && ! contains_path "$ADSP_LIBRARY_PATH" "/usr/lib/rfsa/adsp"; then
      warn "ADSP_LIBRARY_PATH is missing /usr/lib/rfsa/adsp"
    fi
    if [[ -d /dsp ]] && ! contains_path "$ADSP_LIBRARY_PATH" "/dsp"; then
      warn "ADSP_LIBRARY_PATH is missing /dsp"
    fi
  else
    fail "ADSP_LIBRARY_PATH is not set"
  fi

  require_path "/dev/adsprpc-smd" "/dev/adsprpc-smd"
  require_path "/dev/ion" "/dev/ion"
fi

if [[ $need_ort -eq 1 ]]; then
  echo
  echo "-- ONNX Runtime checks --"
  require_env ORT_DYLIB_PATH
  require_path "libonnxruntime.so" "${ORT_DYLIB_PATH-}"
fi

echo
if [[ $failures -gt 0 ]]; then
  echo "runtime check failed: $failures failure(s), $warnings warning(s)"
  if [[ $need_qnn -eq 1 ]]; then
    echo
    print_qnn_exports
  fi
  exit 1
fi

echo "runtime check passed: $warnings warning(s)"
if [[ $warnings -gt 0 ]]; then
  echo "Warnings do not block the selected path, but they are worth reviewing."
fi
