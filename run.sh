#!/usr/bin/env bash
set -euo pipefail

# Default-if-unset, but allow user override: MODELBIN_ROOT=/my/path ./run.sh run
: "${MODELBIN_ROOT:=/nfs/gpu_trainee/final-project/modelbin}"
export MODELBIN_ROOT
export OMP_NUM_THREADS=48

usage() {
  # Double-quoted to expand ${MODELBIN_ROOT} for clarity
  cat <<USAGE
Usage:
  ./run.sh build [default|fast|omp]         (default: omp)
  ./run.sh run [--checkpoint PATH|-c PATH] [-m MODE] [-i INPUT] [-o OUTPUT] [-z TOKENIZER] [-y SYS]
                [-t TEMP] [-p TOP_P] [-n STEPS] [-s SEED]
      - Default checkpoint (if not provided): ${MODELBIN_ROOT}/gpt-oss-20b.bin
      - In getp mode: -i defaults to tests/data/input.txt, -o defaults to tests/data/output.txt
  ./run.sh decode [-i OUTPUT_FILE]
  ./run.sh tokenizer export (builds tokenizer.bin)
  ./run.sh tokenizer test [-t TOKENIZER_BIN] [-i PROMPT] (builds and runs the C++ tokenizer test)
  ./run.sh tokenizer verify [--bin BIN_PATH] [--tok TOK_PATH] [--prompt PROMPT] [--quiet]
USAGE
}

now() { date +"%Y-%m-%d %H:%M:%S"; }

print_kv() {
  # $1=key, $2=value, $3=note(optional)
  if [[ -n "${3-}" ]]; then
    printf "  %-14s: %s %s\n" "$1" "$2" "$3"
  else
    printf "  %-14s: %s\n" "$1" "$2"
  fi
}

find_checkpoint() {
  local ckpt="${1:-}"

  # explicit
  if [[ -n "${ckpt}" ]]; then
    echo "${ckpt}"; return 0
  fi

  # preferred default
  if [[ -f "${MODELBIN_ROOT}/gpt-oss-20b.bin" ]]; then
    echo "${MODELBIN_ROOT}/gpt-oss-20b.bin"; return 0
  fi

  echo ""  # not found, caller handles
}

cmd_build() {
  local flavor="${1:-omp}"
  echo "[BUILD] $(now)"
  print_kv "flavor" "${flavor}"
  case "${flavor}" in
    default) echo "+ make run"; make run ;;
    fast)    echo "+ make runfast"; make runfast ;;
    omp)     echo "+ make runomp"; make runomp ;;
    *) echo "Unknown build flavor: ${flavor}. Use: default|fast|omp" >&2; exit 1 ;;
  esac
}

cmd_run() {
  local ckpt=""
  local mode=""
  local inp=""
  local out=""
  local tok=""
  local sys=""
  local temp=""
  local top_p=""
  local steps=""
  local seed=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -c|--checkpoint) ckpt="$2"; shift 2 ;;
      -m) mode="$2"; shift 2 ;;
      -i) inp="$2"; shift 2 ;;
      -o) out="$2"; shift 2 ;;
      -z) tok="$2"; shift 2 ;;
      -y) sys="$2"; shift 2 ;;
      -t) temp="$2"; shift 2 ;;
      -p) top_p="$2"; shift 2 ;;
      -n) steps="$2"; shift 2 ;;
      -s) seed="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
  done

  ckpt="$(find_checkpoint "${ckpt}")"
  if [[ -z "${ckpt}" ]]; then
    echo "Checkpoint not found. Set MODELBIN_ROOT or pass --checkpoint." >&2
    exit 1
  fi

  # Mode-specific defaults for getp
  if [[ "${mode:-}" == "getp" ]]; then
    [[ -z "${inp}" ]] && inp="tests/data/input.txt"
    [[ -z "${out}" ]] && out="tests/data/output.txt"
  fi

  echo "[RUN] $(now)"
  print_kv "cwd"           "$(pwd)"
  print_kv "MODELBIN_ROOT" "${MODELBIN_ROOT:-<unset>}"
  print_kv "checkpoint"    "${ckpt}"
  print_kv "mode"          "${mode:-generate}" $([[ -z "${mode:-}" ]] && echo "(run.cpp default)" || echo "")

  if [[ "${mode:-}" == "getp" ]]; then
    print_kv "input(-i)"  "${inp}"  $([[ "${inp}" == "tests/data/input.txt" ]] && echo "(run.sh default for getp)" || echo "(provided)")
    print_kv "output(-o)" "${out}"  $([[ "${out}" == "tests/data/output.txt" ]] && echo "(run.sh default for getp)" || echo "(provided)")
  else
    [[ -n "${inp:-}"  ]] && print_kv "input(-i)"  "${inp}" "(provided)"
    [[ -n "${out:-}"  ]] && print_kv "output(-o)" "${out}" "(provided)"
  fi
  print_kv "tokenizer(-z)" "${tok:-<unset>}"
  print_kv "system(-y)"    "${sys:-<unset>}"
  print_kv "temp(-t)"      "${temp:-0.0}"   $([[ -z "${temp:-}" ]] && echo "(run.cpp default)" || echo "(provided)")
  print_kv "top_p(-p)"     "${top_p:-0.9}"  $([[ -z "${top_p:-}" ]] && echo "(run.cpp default)" || echo "(provided)")
  print_kv "steps(-n)"     "${steps:-1024}" $([[ -z "${steps:-}" ]] && echo "(run.cpp default)" || echo "(provided)")
  print_kv "seed(-s)"      "${seed:-time(NULL)}" $([[ -z "${seed:-}" ]] && echo "(run.cpp default)" || echo "(provided)")

  # Optional helpful hint if build/run doesn't exist or isn't executable
  if [[ ! -x build/run ]]; then
    echo "[hint] build/run not found or not executable. Build it via: ./run.sh build" >&2
  fi

  # Build command line (only pass provided flags; plus getp defaults)
  local args=()
  [[ -n "${mode}" ]] && args+=(-m "${mode}")
  [[ -n "${inp}"  ]] && args+=(-i "${inp}")
  [[ -n "${out}"  ]] && args+=(-o "${out}")
  [[ -n "${tok}"  ]] && args+=(-z "${tok}")
  [[ -n "${sys}"  ]] && args+=(-y "${sys}")
  [[ -n "${temp}" ]] && args+=(-t "${temp}")
  [[ -n "${top_p}" ]] && args+=(-p "${top_p}")
  [[ -n "${steps}" ]] && args+=(-n "${steps}")
  [[ -n "${seed}" ]] && args+=(-s "${seed}")

  echo "+ build/run \"${ckpt}\" ${args[*]:-}"
  build/run "${ckpt}" "${args[@]:-}"
}

cmd_decode() {
  local infile="tests/data/output.txt"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -i) infile="$2"; shift 2 ;;
      -h|--help) echo "Usage: ./run.sh decode [-i OUTPUT_FILE]"; exit 0 ;;
      *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
  done
  echo "[DECODE] $(now)"
  print_kv "input(-i)" "${infile}" $([[ "${infile}" == "tests/data/output.txt" ]] && echo "(run.sh default)" || echo "(provided)")
  echo "+ make decode"
  make decode
  echo "+ build/decode -1 -i \"${infile}\""
  build/decode -1 -i "${infile}"
}

cmd_tokenizer() {
  local action="${1:-}"
  shift || true
  case "${action}" in
    export)
      echo "[TOKENIZER export] $(now)"
      echo "+ make tokenizer-bin"
      make tokenizer-bin
      ;;
    test)
      local tokbin="build/tokenizer.bin"
      local prompt="Hello world"
      while [[ $# -gt 0 ]]; do
        case "$1" in
          -t) tokbin="$2"; shift 2 ;;
          -i) prompt="$2"; shift 2 ;;
          -h|--help) echo "Usage: ./run.sh tokenizer test [-t TOKENIZER_BIN] [-i PROMPT]"; exit 0 ;;
          *) echo "Unknown argument: $1" >&2; exit 1 ;;
        esac
      done
      echo "[TOKENIZER test] $(now)"
      print_kv "tokbin" "${tokbin}" $([[ "${tokbin}" == "build/tokenizer.bin" ]] && echo "(default)" || echo "(provided)")
      print_kv "prompt" "${prompt}" $([[ "${prompt}" == "Hello world" ]] && echo "(default)" || echo "(provided)")
      echo "+ make tokenizer-test"
      make tokenizer-test
      echo "+ build/test_tokenizer -t \"${tokbin}\" -i \"${prompt}\""
      build/test_tokenizer -t "${tokbin}" -i "${prompt}"
      ;;
    verify)
      local bin="build/test_tokenizer"
      local tok="build/tokenizer.bin"
      local prompt="tests/data/input.txt"
      local verbose="--verbose"
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --bin) bin="$2"; shift 2 ;;
          --tok) tok="$2"; shift 2 ;;
          --prompt) prompt="$2"; shift 2 ;;
          --quiet) verbose=""; shift 1 ;;
          -h|--help) echo "Usage: ./run.sh tokenizer verify [--bin BIN_PATH] [--tok TOK_PATH] [--prompt PROMPT] [--quiet]"; exit 0 ;;
          *) echo "Unknown argument: $1" >&2; exit 1 ;;
        esac
      done
      echo "[TOKENIZER verify] $(now)"
      print_kv "bin"    "${bin}" $([[ "${bin}" == "build/test_tokenizer" ]] && echo "(default)" || echo "(provided)")
      print_kv "tok"    "${tok}" $([[ "${tok}" == "build/tokenizer.bin" ]] && echo "(default)" || echo "(provided)")
      print_kv "prompt" "${prompt}" $([[ "${prompt}" == "tests/data/input.txt" ]] && echo "(default)" || echo "(provided)")
      print_kv "flags"  "${verbose:-<none>}"
      echo "+ python3 tests/test_tokenizer.py --bin \"${bin}\" --tok \"${tok}\" ${verbose} --prompt \"${prompt}\""
      python3 tests/test_tokenizer.py --bin "${bin}" --tok "${tok}" ${verbose} --prompt "${prompt}"
      ;;
    *)
      echo "Usage: ./run.sh tokenizer {export|test|verify}"; exit 1 ;;
  esac
}

main() {
  local sub="${1:-}"
  if [[ -z "${sub}" ]]; then usage; exit 1; fi
  shift || true
  case "${sub}" in
    build)     cmd_build "${1:-omp}" ;;
    run)       cmd_run "$@" ;;
    decode)    cmd_decode "$@" ;;
    tokenizer) cmd_tokenizer "$@" ;;
    -h|--help|help) usage ;;
    *) echo "Unknown subcommand: ${sub}"; usage; exit 1 ;;
  esac
}

main "$@"
