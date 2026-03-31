#!/usr/bin/env bash
set -euo pipefail

: "${TACHANN_QNN_LIB:=/app/lib/libtachyon_qnnshim.so}"
: "${QNN_SDK_ROOT:=/opt/qairt}"
: "${ORT_DYLIB_PATH:=/opt/onnx/lib/libonnxruntime.so}"

LD_PATHS=()

# Shim
if [ -f "${TACHANN_QNN_LIB}" ]; then
  LD_PATHS+=("$(dirname "${TACHANN_QNN_LIB}")")
fi

if [ -d "${QNN_SDK_ROOT}" ]; then
  for p in \
    "${QNN_SDK_ROOT}/lib/aarch64-ubuntu-gcc9.4" \
    "${QNN_SDK_ROOT}/lib/aarch64-oe-linux-gcc9.3" \
    "${QNN_SDK_ROOT}/lib/aarch64-oe-linux-gcc11.2" \
    "${QNN_SDK_ROOT}/lib/aarch64-oe-linux-gcc8.2" \
    "${QNN_SDK_ROOT}/lib" ; do
    [ -d "$p" ] && LD_PATHS+=("$p")
  done
fi

# ONNX Runtime
if [ -f "${ORT_DYLIB_PATH}" ]; then
  LD_PATHS+=("$(dirname "${ORT_DYLIB_PATH}")")
fi

# Apply LD_LIBRARY_PATH
if [ "${#LD_PATHS[@]}" -gt 0 ]; then
  export LD_LIBRARY_PATH="$(IFS=:; echo "${LD_PATHS[*]}"):${LD_LIBRARY_PATH-}"
fi

echo "[entrypoint] LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}"

# Hexagon DSP libraries (for HTP backend) — prefer v68 for YUPIK/QCM6490
ADSP_PATHS=()
if [ -d "${QNN_SDK_ROOT}/lib/hexagon-v68/unsigned" ]; then
  ADSP_PATHS+=("${QNN_SDK_ROOT}/lib/hexagon-v68/unsigned")
fi
if [ -d "${QNN_SDK_ROOT}/lib" ]; then
  for p in "${QNN_SDK_ROOT}"/lib/hexagon-*/unsigned; do
    [ -d "$p" ] && ADSP_PATHS+=("$p")
  done
fi
for p in /usr/lib/rfsa/adsp /dsp; do
  [ -d "$p" ] && ADSP_PATHS+=("$p")
done
if [ "${#ADSP_PATHS[@]}" -gt 0 ]; then
  if [ -n "${ADSP_LIBRARY_PATH-}" ]; then
    ADSP_PATHS+=("${ADSP_LIBRARY_PATH-}")
  fi
  # QNN expects ADSP_LIBRARY_PATH to be semicolon-delimited.
  export ADSP_LIBRARY_PATH="$(IFS=';'; echo "${ADSP_PATHS[*]}")"
  echo "[entrypoint] ADSP_LIBRARY_PATH=${ADSP_LIBRARY_PATH-}"
fi

# DSP device access check
if [ -e /dev/adsprpc-smd ]; then
  if [ ! -r /dev/adsprpc-smd ] || [ ! -w /dev/adsprpc-smd ]; then
    echo "[entrypoint] WARNING: insufficient permissions on /dev/adsprpc-smd (HTP may fail)"
  fi
else
  echo "[entrypoint] WARNING: /dev/adsprpc-smd not found (HTP unavailable)"
fi
if [ ! -e /dev/ion ]; then
  echo "[entrypoint] WARNING: /dev/ion not found (HTP may fail)"
fi

# Visibility checks
if [ -d "${QNN_SDK_ROOT}" ]; then
  echo "[entrypoint] QNN SDK detected at ${QNN_SDK_ROOT}"
  # Try to find libQnnCpu.so anywhere we added
  found_qnn=""
  IFS=: read -r -a _ldparts <<< "${LD_LIBRARY_PATH-}"
  for d in "${_ldparts[@]}"; do
    if [ -f "${d}/libQnnCpu.so" ]; then
      echo "[entrypoint] Found libQnnCpu.so at ${d}/libQnnCpu.so"
      found_qnn="${d}/libQnnCpu.so"
      break
    fi
  done
  [ -n "${found_qnn}" ] || echo "[entrypoint] WARNING: libQnnCpu.so not found on LD_LIBRARY_PATH"
fi
[ -f "${ORT_DYLIB_PATH}" ] && echo "[entrypoint] ONNX Runtime detected at ${ORT_DYLIB_PATH}"

# Run the app (binary and args come from docker-compose `command:`)
exec "$@"
