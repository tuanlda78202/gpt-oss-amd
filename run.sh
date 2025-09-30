#!/usr/bin/env bash
set -euo pipefail

: "${MODELBIN_ROOT:=/gpu_trainee/final-project/modelbin}"
export MODELBIN_ROOT
export OMP_NUM_THREADS=96

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'              # No Color

FOREST1='\033[38;5;22m'   # Dark forest green
FOREST2='\033[38;5;28m'   # Medium forest green
FOREST3='\033[38;5;34m'   # Bright forest green
FOREST4='\033[38;5;40m'   # Light forest green
FOREST5='\033[38;5;46m'   # Very light forest green

print_fancy_header() {
  echo -e "${FOREST1}                      __"
  echo -e "${FOREST2}                     |  \\"
  echo -e "${FOREST3}  ______    ______  _| \$\$_           ______    _______   _______"
  echo -e "${FOREST4} /      \\  /      \\|   \$\$ \\ ______  /      \\  /       \\ /       \\"
  echo -e "${FOREST5}|  \$\$\$\$\$\$\\|  \$\$\$\$\$\$\\\\\$\$\$\$\$\$|      \\|  \$\$\$\$\$\$\\|  \$\$\$\$\$\$\$\$"
  echo -e "${FOREST4}| \$\$  | \$\$| \$\$  | \$\$ | \$\$ __\\\$\$\$\$\$\$| \$\$  | \$\$ \\\$\$    \\  \\\$\$\$    \\"
  echo -e "${FOREST3}| \$\$__| \$\$| \$\$__/ \$\$ | \$\$|  \\      | \$\$__/ \$\$ _\\\$\$\$\$\$\$\\ _\\\$\$\$\$\$\$\\"
  echo -e "${FOREST2} \\\$\$    \$\$| \$\$    \$\$  \\\$\$  \$\$       \\\$\$    \$\$|       \$\$|       \$\$"
  echo -e "${FOREST1} _\\\$\$\$\$\$\$\$| \$\$\$\$\$\$\$\$    \\\$\$\$\$         \\\$\$\$\$\$\$  \\\$\$\$\$\$\$\$  \\\$\$\$\$\$\$\$"
  echo -e "${FOREST2}|  \\__| \$\$| \$\$"
  echo -e "${FOREST3} \\\$\$    \$\$| \$\$"
  echo -e "${FOREST4}  \\\$\$\$\$\$\$  \\\$\$"
  echo -e "${FOREST5}  ════════════════════════════════════════════════════════════════"
  echo -e "${FOREST4}                         gpt-oss-c (moreh)"
  echo -e "${FOREST3}              https://github.com/tuanlda78202/gpt-oss-c"
  echo -e "${FOREST2}  ════════════════════════════════════════════════════════════════"
  echo -e "${NC}"
}

# Color functions
print_header() {
  local color="$1"
  local title="$2"
  echo -e "${color}==================================================================${NC}"
  echo -e "${color}[$2]${NC} ${WHITE}$(now)${NC}"
}

print_step() {
  echo -e "${GREEN}+${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
  echo -e "${CYAN}[INFO]${NC} $1"
}

print_command() {
  echo -e "${PURPLE}>>>${NC} $1"
}

print_executing() {
  echo -e "${YELLOW} EXECUTING:${NC} $1"
}

usage() {
  echo -e "${YELLOW}🚀 QUICK START:${NC}"
  echo -e "  ${GREEN}./run.sh build${NC}                    # Build the project (default: omp flavor)"
  echo -e "  ${GREEN}./run.sh run${NC}                      # Run with default settings (getp mode, 1 GPU)"
  echo -e "  ${GREEN}./run.sh all${NC}                      # Build and run in one command"
  echo -e "  ${GREEN}./run.sh -h${NC}                       # Show this help message"
  echo ""
  echo -e "${YELLOW}🔨 BUILD COMMANDS:${NC}"
  echo -e "  ${GREEN}./run.sh build [default|fast|omp]${NC}         (default: omp)"
  echo ""
  echo -e "  ${CYAN}Build Flavors:${NC}"
  echo -e "    ${WHITE}default${NC}  - Uses 'make run'"
  echo -e "    ${WHITE}fast${NC}     - Uses 'make runfast'"
  echo -e "    ${WHITE}omp${NC}      - Uses 'make runomp' (recommended)"
  echo ""
  echo -e "${RED}🏃‍♂️ RUN COMMANDS:${NC}"
  echo -e "  ${GREEN}./run.sh run [--checkpoint PATH|-c PATH] [-m MODE] [-i INPUT] [-o OUTPUT] [-z TOKENIZER] [-y SYS]${NC}"
  echo -e "                ${GREEN}[-T TEMP] [-t TRUNCATE] [-p TOP_P] [-n STEPS] [-s SEED] [-l] [-g N_GPUS] [-b BATCH_SIZE] [-f] [-v VERIFY_FILE]${NC}"
  echo ""
  echo -e "  ${CYAN}Key Features:${NC}"
  echo -e "    • Default checkpoint: ${WHITE}${MODELBIN_ROOT}/gpt-oss-20b.bin${NC}"
  echo -e "    • Default mode: ${WHITE}getp${NC} (uses 20b model)"
  echo -e "    • GPU allocation: Automatically uses ${WHITE}'srun --gres=gpu:N_GPUS'${NC}"
  echo -e "    • Logging: Save output to ${WHITE}log.txt${NC} with -l flag"
  echo -e "    • Profiling: Enable forward timing with ${WHITE}-f${NC} flag"
  echo ""
  echo -e "  ${CYAN}Model Selection Shortcuts:${NC}"
  echo -e "    ${WHITE}-m 20${NC}   - Auto-select 20B model (${MODELBIN_ROOT}/gpt-oss-20b.bin, tests/gt/output_20b.txt)"
  echo -e "    ${WHITE}-m 120${NC}  - Auto-select 120B model (${MODELBIN_ROOT}/gpt-oss-120b.bin, tests/gt/output_120b.txt)"
  echo ""
  echo -e "  ${CYAN}Mode-specific defaults for getp:${NC}"
  echo -e "    • Input defaults to: ${WHITE}tests/data/input.txt${NC}"
  echo -e "    • Output defaults to: ${WHITE}tests/data/output.txt${NC}"
  echo ""
  echo -e "  ${CYAN}Parameters:${NC}"
  echo -e "    ${WHITE}-c, --checkpoint PATH${NC}  Specify model checkpoint"
  echo -e "    ${WHITE}-m MODE${NC}                Set run mode (default: getp) or model size (20/120)"
  echo -e "    ${WHITE}-i INPUT${NC}               Input file path"
  echo -e "    ${WHITE}-o OUTPUT${NC}              Output file path"
  echo -e "    ${WHITE}-z TOKENIZER${NC}           Tokenizer path"
  echo -e "    ${WHITE}-y SYS${NC}                 System prompt"
  echo -e "    ${WHITE}-T TEMP${NC}                Temperature (default: 0.0)"
  echo -e "    ${WHITE}-t TRUNCATE${NC}            Truncate input to first N lines (getp mode only)"
  echo -e "    ${WHITE}-p TOP_P${NC}               Top-p sampling (default: 0.9)"
  echo -e "    ${WHITE}-n STEPS${NC}               Number of steps (default: 1024)"
  echo -e "    ${WHITE}-s SEED${NC}                Random seed"
  echo -e "    ${WHITE}-l${NC}                     Log output to log.txt"
  echo -e "    ${WHITE}-g N_GPUS${NC}              Number of GPUs to request (default: 1)"
  echo -e "    ${WHITE}-b BATCH_SIZE${NC}          Batch size for getp mode (default: 32)"
  echo -e "    ${WHITE}-f${NC}                     Enable forward pass profiling (shows timing breakdown)"
  echo -e "    ${WHITE}-v VERIFY_FILE${NC}         Ground truth file for verification (default: tests/gt/output_20b.txt)"
  echo -e "    ${WHITE}--kv16${NC}                 Use 16-bit KV cache (bfloat16, default)"
  echo -e "    ${WHITE}--kv32${NC}                 Use 32-bit KV cache (override kv16)"
  echo ""
  echo -e "${PURPLE}🔄 ALL-IN-ONE COMMANDS:${NC}"
  echo -e "  ${GREEN}./run.sh all [-c] [--checkpoint PATH|-c PATH] [-m MODE] [-i INPUT] [-o OUTPUT] [-z TOKENIZER] [-y SYS]${NC}"
  echo -e "                ${GREEN}[-T TEMP] [-t TRUNCATE] [-p TOP_P] [-n STEPS] [-s SEED] [-l] [-g N_GPUS] [-b BATCH_SIZE] [-f] [-v VERIFY_FILE] [--kv16|--kv32]${NC}"
  echo ""
  echo -e "  ${CYAN}Features:${NC}"
  echo -e "    • Combines: ${WHITE}./run.sh build && ./run.sh run${NC}"
  echo -e "    • ${WHITE}-c${NC} flag runs 'make clean' before building"
  echo -e "    • All other flags are passed to the run command"
  echo ""
  echo -e "${BLUE} DECODE COMMANDS:${NC}"
  echo -e "  ${GREEN}./run.sh decode [-i OUTPUT_FILE] [-l]${NC}"
  echo ""
  echo -e "  ${CYAN}Features:${NC}"
  echo -e "    • Default input: ${WHITE}tests/gt/output_20b.txt${NC} (GT file)"
  echo -e "    • ${WHITE}-l${NC} flag saves decoded output to ${WHITE}gt_decoded.txt${NC}"
  echo ""
  echo -e "${CYAN} TOKENIZER COMMANDS:${NC}"
  echo -e "  ${GREEN}./run.sh tokenizer export${NC}                               # Builds tokenizer.bin"
  echo -e "  ${GREEN}./run.sh tokenizer test [-t TOKENIZER_BIN] [-i PROMPT]${NC}  # Builds and runs C++ tokenizer test"
  echo -e "  ${GREEN}./run.sh tokenizer verify [--bin BIN_PATH] [--tok TOK_PATH] [--prompt PROMPT] [--quiet]${NC}"
  echo ""
  echo -e "${WHITE}📁 FILE STRUCTURE:${NC}"
  echo -e "  ${CYAN}tests/${NC}"
  echo -e "  ├── ${CYAN}data/${NC}"
  echo -e "  │   ├── ${WHITE}input.txt${NC}          # Default input for getp mode"
  echo -e "  │   └── ${WHITE}output.txt${NC}         # Default output for getp mode"
  echo -e "  └── ${CYAN}gt/${NC}"
  echo -e "      └── ${WHITE}output.txt${NC}         # Default GT file for decode"
  echo -e "  ${CYAN}build/${NC}                     # Build artifacts"
  echo -e "  ${WHITE}log.txt${NC}                    # Output log (when using -l flag)"
  echo ""
  echo -e "${GREEN}📊 EXAMPLE USAGE:${NC}"
  echo -e "  ${CYAN}# Basic getp mode with 1 GPU (default 20B model)${NC}"
  echo -e "  ${GREEN}./run.sh run${NC}"
  echo ""
  echo -e "  ${CYAN}# Use 20B model explicitly${NC}"
  echo -e "  ${GREEN}./run.sh run -m 20${NC}"
  echo ""
  echo -e "  ${CYAN}# Use 120B model with 4 GPUs${NC}"
  echo -e "  ${GREEN}./run.sh run -m 120 -g 4${NC}"
  echo ""
  echo -e "  ${CYAN}# Getp mode with profiling and logging${NC}"
  echo -e "  ${GREEN}./run.sh run -f -l${NC}"
  echo ""
  echo -e "  ${CYAN}# Getp mode with 2 GPUs, batch size 32, and profiling${NC}"
  echo -e "  ${GREEN}./run.sh run -g 2 -b 32 -f${NC}"
  echo ""
  echo -e "  ${CYAN}# Truncate input to first 16 lines${NC}"
  echo -e "  ${GREEN}./run.sh run -t 16${NC}"
  echo ""
  echo -e "  ${CYAN}# Custom checkpoint with 4 GPUs${NC}"
  echo -e "  ${GREEN}./run.sh run -c /path/to/model.bin -g 4${NC}"
  echo ""
  echo -e "  ${CYAN}# Build and run with clean${NC}"
  echo -e "  ${GREEN}./run.sh all -c -g 2${NC}"
  echo ""
  echo -e "  ${CYAN}# Decode GT file and save output${NC}"
  echo -e "  ${GREEN}./run.sh decode -l${NC}"
  echo ""
  echo -e "  ${CYAN}# Test tokenizer with custom prompt${NC}"
  echo -e "  ${GREEN}./run.sh tokenizer test -i \"Hello, world!\"${NC}"
  echo ""
  echo -e "${YELLOW}⚙️ ENVIRONMENT VARIABLES:${NC}"
  echo -e "  ${WHITE}MODELBIN_ROOT${NC}     - Path to model binaries (default: ${CYAN}${MODELBIN_ROOT}${NC})"
  echo -e "  ${WHITE}OMP_NUM_THREADS${NC}   - OpenMP thread count (default: ${CYAN}96${NC})"
  echo ""
  echo -e "${BLUE}🔧 DEPENDENCIES:${NC}"
  echo -e "  ${WHITE}make${NC}      - For building the project"
  echo -e "  ${WHITE}srun${NC}      - For GPU allocation (SLURM)"
  echo -e "  ${WHITE}python3${NC}   - For tokenizer verification tests"
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

  echo "${MODELBIN_ROOT}/gpt-oss-20b.bin"; return 0
}

cmd_build() {
  local flavor="${1:-omp}"
  print_header "${YELLOW}" "BUILD"
  print_kv "flavor" "${flavor}"
  case "${flavor}" in
    default)
      print_step "make run"
      print_executing "make run"
      make run
      print_success "Build completed successfully"
      ;;
    fast)
      print_step "make runfast"
      print_executing "make runfast"
      make runfast
      print_success "Build completed successfully"
      ;;
    omp)
      print_step "make runomp"
      print_executing "make runomp"
      make runomp
      print_success "Build completed successfully"
      ;;
    *)
      print_error "Unknown build flavor: ${flavor}. Use: default|fast|omp"
      exit 1
      ;;
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
  local log_output=""
  local n_gpus="1"
  local batch_size=""
  local enable_profiling=""
  local verify_file=""
  local truncate_lines=""
  local kv16_flag=""
  local kv32_flag=""
  local odd_win=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -c|--checkpoint) ckpt="$2"; shift 2 ;;
      -m) mode="$2"; shift 2 ;;
      -i) inp="$2"; shift 2 ;;
      -o) out="$2"; shift 2 ;;
      -z) tok="$2"; shift 2 ;;
      -y) sys="$2"; shift 2 ;;
      -T) temp="$2"; shift 2 ;;
      -p) top_p="$2"; shift 2 ;;
      -n) steps="$2"; shift 2 ;;
      -s) seed="$2"; shift 2 ;;
      -l) log_output="1"; shift 1 ;;
      -g) n_gpus="$2"; shift 2 ;;
      -b) batch_size="$2"; shift 2 ;;
      -f) enable_profiling="1"; shift 1 ;;
      -v) verify_file="$2"; shift 2 ;;
      -t) truncate_lines="$2"; shift 2 ;;
      --kv16) kv16_flag="1"; shift 1 ;;
      --kv32) kv32_flag="1"; shift 1 ;;
      --odd_win) odd_win="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
  done

  # Handle model size shortcuts (20/120) for mode
  local model_size=""
  if [[ "${mode}" == "20" ]]; then
    model_size="20b"
    mode="getp"
    [[ -z "${ckpt}" ]] && ckpt="${MODELBIN_ROOT}/gpt-oss-20b.bin"
    [[ -z "${verify_file}" ]] && verify_file="tests/gt/output_20b.txt"
  elif [[ "${mode}" == "120" ]]; then
    model_size="120b"
    mode="getp"
    [[ -z "${ckpt}" ]] && ckpt="${MODELBIN_ROOT}/gpt-oss-120b.bin"
    [[ -z "${verify_file}" ]] && verify_file="tests/gt/output_120b.txt"
  fi

  ckpt="$(find_checkpoint "${ckpt}")"
  if [[ -z "${ckpt}" ]]; then
    print_error "Checkpoint not found. Set MODELBIN_ROOT or pass --checkpoint."
    exit 1
  fi

  # Default mode is getp if not specified
  [[ -z "${mode}" ]] && mode="getp"

  # Mode-specific defaults for getp
  if [[ "${mode}" == "getp" ]]; then
    [[ -z "${inp}" ]] && inp="tests/data/input.txt"
    [[ -z "${out}" ]] && out="tests/data/output.txt"
    [[ -z "${verify_file}" ]] && verify_file="tests/gt/output_20b.txt"
  fi

  print_header "${RED}" "RUN"
  print_kv "cwd"           "$(pwd)"
  print_kv "MODELBIN_ROOT" "${MODELBIN_ROOT:-<unset>}"
  print_kv "checkpoint"    "${ckpt}"
  print_kv "mode"          "${mode}" "$([[ "${mode}" == "getp" && -z "${model_size}" ]] && echo "(default)" || [[ -n "${model_size}" ]] && echo "(${model_size} model)" || echo "(provided)")"
  print_kv "gpus(-g)"      "${n_gpus}" "$([[ "${n_gpus}" == "1" ]] && echo "(default)" || echo "(requested)")"

  if [[ "${mode}" == "getp" ]]; then
    print_kv "input(-i)"  "${inp}"  "$([[ "${inp}" == "tests/data/input.txt" ]] && echo "(run.sh default for getp)" || echo "(provided)")"
    print_kv "output(-o)" "${out}"  "$([[ "${out}" == "tests/data/output.txt" ]] && echo "(run.sh default for getp)" || echo "(provided)")"
    if [[ -n "${model_size}" ]]; then
      print_kv "verify(-v)" "${verify_file}"  "(${model_size} model default)"
    else
      print_kv "verify(-v)" "${verify_file}"  "$([[ "${verify_file}" == "tests/gt/output_20b.txt" ]] && echo "(run.sh default for getp)" || echo "(provided)")"
    fi
  else
    [[ -n "${inp:-}"  ]] && print_kv "input(-i)"  "${inp}" "(provided)"
    [[ -n "${out:-}"  ]] && print_kv "output(-o)" "${out}" "(provided)"
    [[ -n "${verify_file:-}" ]] && print_kv "verify(-v)" "${verify_file}" "(provided)"
  fi
  print_kv "tokenizer(-z)" "${tok:-<unset>}"
  print_kv "system(-y)"    "${sys:-<unset>}"
  print_kv "temp(-T)"      "${temp:-0.0}"   "$([[ -z "${temp:-}" ]] && echo "(run.cpp default)" || echo "(provided)")"
  print_kv "top_p(-p)"     "${top_p:-0.9}"  "$([[ -z "${top_p:-}" ]] && echo "(run.cpp default)" || echo "(provided)")"
  print_kv "steps(-n)"     "${steps:-1024}" "$([[ -z "${steps:-}" ]] && echo "(run.cpp default)" || echo "(provided)")"
  print_kv "seed(-s)"      "${seed:-time(NULL)}" "$([[ -z "${seed:-}" ]] && echo "(run.cpp default)" || echo "(provided)")"
  print_kv "batch_size(-b)" "${batch_size:-32}" "$([[ -z "${batch_size:-}" ]] && echo "(run.cpp default)" || echo "(provided)")"
  print_kv "profiling(-f)" "${enable_profiling:+enabled}" "$([[ -n "${enable_profiling}" ]] && echo "(forward timing)" || echo "(disabled)")"
  print_kv "logging(-l)"   "${log_output:+enabled}" "$([[ -n "${log_output}" ]] && echo "(to log.txt)" || echo "(disabled)")"
  print_kv "truncate(-t)" "${truncate_lines:-<none>}" "$([[ -n "${truncate_lines}" ]] && echo "(limit to first ${truncate_lines} lines)" || echo "(process all lines)")"
  if [[ -n "${kv32_flag}" ]]; then
    print_kv "kv_cache" "fp32" "(--kv32)"
  elif [[ -n "${kv16_flag}" ]]; then
    print_kv "kv_cache" "bf16" "(--kv16)"
  else
    print_kv "kv_cache" "bf16" "(default)"
  fi
  [[ -n "${odd_win}" ]] && print_kv "odd_win" "${odd_win}" "(--odd_win)"

  # Optional helpful hint if build/run doesn't exist or isn't executable
  if [[ ! -x build/run ]]; then
    print_warning "build/run not found or not executable. Build it via: ./run.sh build"
  fi

  # Build command line (only pass provided flags; plus getp defaults)
  local args=()
  [[ -n "${mode}" ]] && args+=(-m "${mode}")
  [[ -n "${inp}"  ]] && args+=(-i "${inp}")
  [[ -n "${out}"  ]] && args+=(-o "${out}")
  [[ -n "${tok}"  ]] && args+=(-z "${tok}")
  [[ -n "${sys}"  ]] && args+=(-y "${sys}")
  [[ -n "${temp}" ]] && args+=(-T "${temp}")
  [[ -n "${top_p}" ]] && args+=(-p "${top_p}")
  [[ -n "${steps}" ]] && args+=(-n "${steps}")
  [[ -n "${seed}" ]] && args+=(-s "${seed}")
  [[ -n "${batch_size}" ]] && args+=(-b "${batch_size}")
  [[ -n "${enable_profiling}" ]] && args+=(-f "1")
  [[ -n "${verify_file}" ]] && args+=(-v "${verify_file}")
  [[ -n "${truncate_lines}" ]] && args+=(-t "${truncate_lines}")
  [[ -n "${kv16_flag}" ]] && args+=(--kv16)
  [[ -n "${kv32_flag}" ]] && args+=(--kv32)
  [[ -n "${odd_win}" ]] && args+=(--odd_win "${odd_win}")

  local srun_cmd="srun --gres=gpu:${n_gpus}" # --exclude MV-DZ-MI250-01
  print_command "${srun_cmd} build/run \"${ckpt}\" ${args[*]:-}"

  if [[ -n "${log_output}" ]]; then
    print_info "logging output to log.txt"
    print_executing "${srun_cmd} build/run \"${ckpt}\" ${args[*]:-} | tee log.txt"
    ${srun_cmd} build/run "${ckpt}" "${args[@]:-}" 2>&1 | tee log.txt
  else
    print_executing "${srun_cmd} build/run \"${ckpt}\" ${args[*]:-}"
    ${srun_cmd} build/run "${ckpt}" "${args[@]:-}"
  fi
}

cmd_all() {
  local clean_flag=""
  local run_args=()

  # Parse arguments - separate clean flag from run arguments
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -c) clean_flag="1"; shift 1 ;;
      *) run_args+=("$1"); shift 1 ;;
    esac
  done

  print_header "${PURPLE}" "ALL"
  print_kv "clean" "${clean_flag:+enabled}" "$([[ -n "${clean_flag}" ]] && echo "(-c flag)" || echo "(disabled)")"

  # Run make clean if -c flag is provided
  if [[ -n "${clean_flag}" ]]; then
    print_step "make clean"
    print_executing "make clean"
    make clean
    print_success "Clean completed"
  fi

  # Build
  print_info "building..."
  cmd_build "omp"

  # Run
  print_info "running..."
  cmd_run "${run_args[@]}"
}

cmd_decode() {
  local infile="tests/gt/output_20b.txt"
  local log_output=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -i) infile="$2"; shift 2 ;;
      -l) log_output="1"; shift 1 ;;
      -h|--help) echo "Usage: ./run.sh decode [-i OUTPUT_FILE] [-l]"; exit 0 ;;
      *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
  done
  print_header "${BLUE}" "DECODE"
  print_kv "input(-i)" "${infile}" "$([[ "${infile}" == "tests/gt/output_20b.txt" ]] && echo "(run.sh default GT)" || echo "(provided)")"
  print_kv "logging(-l)" "${log_output:+enabled}" "$([[ -n "${log_output}" ]] && echo "(to gt_decoded.txt)" || echo "(disabled)")"
  print_step "make decode"
  print_executing "make decode"
  make decode
  print_command "build/decode -1 -i \"${infile}\""

  if [[ -n "${log_output}" ]]; then
    print_info "saving decoded output to gt_decoded.txt"
    print_executing "build/decode -1 -i \"${infile}\" > gt_decoded.txt"
    build/decode -1 -i "${infile}" > gt_decoded.txt 2>&1
    print_success "decoded output saved to gt_decoded.txt"
  else
    print_executing "build/decode -1 -i \"${infile}\""
    build/decode -1 -i "${infile}"
  fi
}

cmd_tokenizer() {
  local action="${1:-}"
  shift || true
  case "${action}" in
    export)
      print_header "${CYAN}" "TOKENIZER export"
      print_step "make tokenizer-bin"
      print_executing "make tokenizer-bin"
      make tokenizer-bin
      print_success "Tokenizer export completed"
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
      print_header "${CYAN}" "TOKENIZER test"
      print_kv "tokbin" "${tokbin}" "$([[ "${tokbin}" == "build/tokenizer.bin" ]] && echo "(default)" || echo "(provided)")"
      print_kv "prompt" "${prompt}" "$([[ "${prompt}" == "Hello world" ]] && echo "(default)" || echo "(provided)")"
      print_step "make tokenizer-test"
      print_executing "make tokenizer-test"
      make tokenizer-test
      print_command "build/test_tokenizer -t \"${tokbin}\" -i \"${prompt}\""
      print_executing "build/test_tokenizer -t \"${tokbin}\" -i \"${prompt}\""
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
      print_header "${CYAN}" "TOKENIZER verify"
      print_kv "bin"    "${bin}" "$([[ "${bin}" == "build/test_tokenizer" ]] && echo "(default)" || echo "(provided)")"
      print_kv "tok"    "${tok}" "$([[ "${tok}" == "build/tokenizer.bin" ]] && echo "(default)" || echo "(provided)")"
      print_kv "prompt" "${prompt}" "$([[ "${prompt}" == "tests/data/input.txt" ]] && echo "(default)" || echo "(provided)")"
      print_kv "flags"  "${verbose:-<none>}"
      print_command "python3 tests/test_tokenizer.py --bin \"${bin}\" --tok \"${tok}\" ${verbose} --prompt \"${prompt}\""
      print_executing "python3 tests/test_tokenizer.py --bin \"${bin}\" --tok \"${tok}\" ${verbose} --prompt \"${prompt}\""
      python3 tests/test_tokenizer.py --bin "${bin}" --tok "${tok}" ${verbose} --prompt "${prompt}"
      ;;
    *)
      echo "Usage: ./run.sh tokenizer {export|test|verify}"; exit 1 ;;
  esac
}

main() {
  print_fancy_header

  local sub="${1:-}"
  if [[ -z "${sub}" ]]; then usage; exit 1; fi
  shift || true
  case "${sub}" in
    build)     cmd_build "${1:-omp}" ;;
    run)       cmd_run "$@" ;;
    all)       cmd_all "$@" ;;
    decode)    cmd_decode "$@" ;;
    tokenizer) cmd_tokenizer "$@" ;;
    -h|--help|help) usage ;;
    *) echo "Unknown subcommand: ${sub}"; usage; exit 1 ;;
  esac
}

main "$@"
