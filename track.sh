#!/bin/bash

# alias track='FORCE_COLOR=1 watch --color -n 0.1 /path/to/track.sh'

TEAM_DATA="
Name,User ID,Team ID
Thai Minh Dung,getp01,team01
Vu Tung Duong,getp20,team01
Chu Huu Dang Truong,getp08,team01
Nguyen Phi Phuc,getp21,team02
Truong Minh Tri,getp22,team02
Trinh An Hai,getp05,team03
Vu Trung Hieu,getp25,team03
Pham Van Trong,getp28,team04
Phan Minh Anh Tuan,getp07,team04
Nguyen Thi Hoai Linh,getp19,team05
Tran Thanh Nhan,getp24,team05
Nguyen Duc Trung,getp16,team06
Vu Quoc Tuan,getp29,team06
Vu Huy Hoang,getp03,team07
Nguyen Xuan Thanh,getp02,team07
Huynh Tien Dung,getp06,team08
Hoang Nhu Vinh,getp23,team08
Nguyen Quy Dang,getp26,team09
Nguyen Huy Thai,getp18,team09
Dam Anh Duc,getp27,team10
Pham Duc Minh,getp04,team10
Tran Thi Thu Hue,getp14,team11
Le Thanh Nam,getp11,team11
Tong Duc Khai,getp15,team12
Vu Quang Nam,getp10,team12
Cao Van The Anh,getp12,team13
Ngo Dinh Luyen,getp17,team13
Tran Hong Quan,getp13,team15
Le Duc Anh Tuan,getp09,team15
"

# ---------- colors ----------
if { [[ -t 1 ]] || [[ -n "$FORCE_COLOR" ]]; } && [[ -z "$NO_COLOR" ]]; then
  RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
  BLUE=$'\033[34m'; CYAN=$'\033[36m'; GREY=$'\033[90m'
  BOLD=$'\033[1m'; RESET=$'\033[0m'
else
  RED=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""; GREY=""; BOLD=""; RESET=""
fi

# ---------- helpers ----------
strip_ansi() { sed -E 's/\x1B\[[0-9;]*[[:alpha:]]//g'; }
truncate_field() {
  local txt="$1" w="$2" plain; plain="$(printf '%s' "$txt" | strip_ansi)"
  local len=${#plain}
  if (( len > w )); then printf '%s' "${plain:0:w-1}…"; else printf '%s' "$plain"; fi
}
print_cell() {  # $1=color $2=width $3=text
  local color="$1" w="$2" txt; txt="$(truncate_field "$3" "$w")"
  printf '%s' "$color"; printf "%-${w}s " "$txt"; printf '%s' "$RESET"
}

# Lookup "Full Name|TeamID" by User ID (getpXX) from embedded team data
lookup_person() {
  local user_id="$1"
  echo "$TEAM_DATA" | awk -F',' -v id="$user_id" '
    BEGIN { gsub_spc="^[ \t]+|[ \t]+$" }
    NR==1 { next }  # skip header
    { n=$1; u=$2; t=$3;
      gsub(gsub_spc,"",n); gsub(gsub_spc,"",u); gsub(gsub_spc,"",t);
      if (u==id) { print n "|" t; exit }
    }
  '
}

# Get GPU count for a job using scontrol
get_gpu_count() {
  local jobid="$1"
  local gpu_count="0"

  # Try to get GPU count from scontrol show job
  if command -v scontrol >/dev/null 2>&1; then
    local gpu_info=$(scontrol show job "$jobid" 2>/dev/null | grep -i "gres/gpu" | head -1)
    if [[ -n "$gpu_info" ]]; then
      # Extract GPU count from gres/gpu=X format
      gpu_count=$(echo "$gpu_info" | sed -n 's/.*gres\/gpu:\([0-9]*\).*/\1/p')
      [[ -z "$gpu_count" ]] && gpu_count="0"
    fi
  fi

  echo "$gpu_count"
}

# ---------- widths ----------
W_JOBID=6; W_NAME=22; W_USER=8; W_TEAM=6; W_ST=2; W_TIME=8; W_NODES=5; W_GPU=3; W_NODELIST=18

# ---------- header ----------
printf "%-${W_JOBID}s %-${W_NAME}s %-${W_USER}s %-${W_TEAM}s %-${W_ST}s %-${W_TIME}s %-${W_NODES}s %-${W_GPU}s %-${W_NODELIST}s\n" \
  "JOBID" "NAME" "USER" "TEAM" "ST" "TIME" "NODES" "GPU" "NODELIST"
printf "%-${W_JOBID}s %-${W_NAME}s %-${W_USER}s %-${W_TEAM}s %-${W_ST}s %-${W_TIME}s %-${W_NODES}s %-${W_GPU}s %-${W_NODELIST}s\n" \
  "$(printf -- '%.0s-' $(seq 1 $W_JOBID))" \
  "$(printf -- '%.0s-' $(seq 1 $W_NAME))" \
  "$(printf -- '%.0s-' $(seq 1 $W_USER))" \
  "$(printf -- '%.0s-' $(seq 1 $W_TEAM))" \
  "$(printf -- '%.0s-' $(seq 1 $W_ST))" \
  "$(printf -- '%.0s-' $(seq 1 $W_TIME))" \
  "$(printf -- '%.0s-' $(seq 1 $W_NODES))" \
  "$(printf -- '%.0s-' $(seq 1 $W_GPU))" \
  "$(printf -- '%.0s-' $(seq 1 $W_NODELIST))"

# ---------- main ----------
# squeue fields now exclude PARTITION and JOBNAME
# %i id | %u user | %t state | %M time | %D nodes | %R nodelist
squeue -h -o "%i %u %t %M %D %R" \
| awk '{
    state=$3; r=9
    if(state=="R")r=0; else if(state=="PD")r=1; else if(state=="CG")r=2; else if(state=="CD")r=3
    printf("%d %s\n", r, $0)
  }' \
| sort -k1,1n -k2,2n \
| cut -d' ' -f2- \
| while IFS= read -r line; do
    # JOBID USER ST TIME NODES NODELIST...
    read -r jobid user st time nodes rest <<< "$line" || continue
    nodelist="$rest"

    info="$(lookup_person "$user")"
    fullname="${info%%|*}"; team="${info##*|}"
    [[ -z "$fullname" ]] && fullname="Unknown($user)"
    [[ -z "$team" ]] && team="?"

    # Get GPU count for this job
    gpu_count="$(get_gpu_count "$jobid")"

    case "$st" in
      R)  STC="$BOLD$RED" ;;
      PD) STC="$BOLD$GREY" ;;
      CG|CD) STC="$BOLD$GREEN" ;;
      *)  STC="$BOLD$RESET" ;;
    esac
    [[ "$user" == "getp09" ]] && ROW="$YELLOW" || ROW=""

    print_cell "$ROW"        "$W_JOBID"    "$jobid"
    print_cell "$BOLD$ROW"   "$W_NAME"     "$fullname"
    print_cell "$ROW"        "$W_USER"     "$user"
    print_cell "$ROW"        "$W_TEAM"     "$team"
    print_cell "$STC"        "$W_ST"       "$st"
    print_cell "$ROW"        "$W_TIME"     "$time"
    print_cell "$ROW"        "$W_NODES"    "$nodes"
    print_cell "$BLUE$ROW"   "$W_GPU"      "$gpu_count"
    print_cell "$CYAN$ROW"   "$W_NODELIST" "$nodelist"
    printf "\n"
  done
