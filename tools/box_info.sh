#!/usr/bin/env bash
set -u

JSON=0
MEMBW_BIN="${MEMBW_BIN:-}"      # binary from tests/membw.c; without it the MEASURED bandwidth is skipped
OUT="${OUT:-}"                  # when set, the JSON is written there too (besides the readable report)
MEMBW_REPS="${MEMBW_REPS:-5}"
while [ $# -gt 0 ]; do
    case "$1" in
        --json)   JSON=1 ;;
        --membw)  MEMBW_BIN="${2:-}"; shift ;;
        --out)    OUT="${2:-}"; shift ;;
        -h|--help) sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (use --json | --membw BIN | --out FILE)" >&2; exit 2 ;;
    esac
    shift
done

have() { command -v "$1" >/dev/null 2>&1; }
rd()   { [ -r "$1" ] && tr -d '\n' < "$1" 2>/dev/null; }        # a /sys file, one line
rdm()  { [ -r "$1" ] && cat "$1" 2>/dev/null; }                 # multi-line file

DEGRADED=""
degraded() { DEGRADED="$DEGRADED$1
"; }
WARN=""
warn() { WARN="$WARN$1
"; }

sz_to_mb() {
    [ -z "${1:-}" ] && return
    awk -v s="$1" 'BEGIN{
        u = substr(s, length(s), 1); n = s + 0;
        if (u=="K"||u=="k")      n = n/1024;
        else if (u=="M"||u=="m") n = n;
        else if (u=="G"||u=="g") n = n*1024;
        else                     n = n/1048576;   # no suffix = bytes
        printf "%.4f", n;                          # 2 decimals would round the L1 caches away
    }'
}
mb_h() {
    [ -z "${1:-}" ] && { printf 'n/d'; return; }
    awk -v m="$1" 'BEGIN{
        if (m == 0)      printf "0";
        else if (m >= 1024) printf "%.2f GiB", m/1024;
        else if (m >= 10) printf "%.0f MiB", m;
        else if (m >= 1) printf "%.1f MiB", m;
        else             printf "%.0f KiB", m*1024;    # sotto il MiB, "0 MiB" sarebbe un dato falso
    }'
}

jstr() {
    case "${1:-}" in "") printf 'null'; return ;; esac
    printf '"%s"' "$(printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | tr '\n\t' '  ')"
}
jnum() { case "${1:-}" in ''|*[!0-9.]*) printf 'null' ;; *) printf '%s' "$1" ;; esac; }
jarr() {
    local out="" line
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        [ -n "$out" ] && out="$out,"
        out="$out$(jstr "$line")"
    done <<EOF
${1:-}
EOF
    printf '[%s]' "$out"
}
jobj() {
    local out=""
    while [ $# -ge 2 ]; do
        [ -n "$out" ] && out="$out,"
        out="$out$(printf '"%s":%s' "$1" "$2")"
        shift 2
    done
    printf '{%s}' "$out"
}

B_OS="$(uname -s 2>/dev/null)"
B_KERNEL="$(uname -r 2>/dev/null)"
B_ARCH="$(uname -m 2>/dev/null)"
B_HOST="$(hostname 2>/dev/null)"
B_DATE="$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null)"

B_GCP_MT=""; B_GCP_ZONE=""; B_GCP_PREEMPT=""
if have curl; then
    _md="http://metadata.google.internal/computeMetadata/v1/instance"
    _mt=$(curl -s --max-time 1 -H "Metadata-Flavor: Google" "$_md/machine-type" 2>/dev/null)
    if [ -n "$_mt" ]; then
        B_GCP_MT="${_mt##*/}"
        B_GCP_ZONE="$(curl -s --max-time 1 -H 'Metadata-Flavor: Google' "$_md/zone" 2>/dev/null)"
        B_GCP_ZONE="${B_GCP_ZONE##*/}"
        B_GCP_PREEMPT="$(curl -s --max-time 1 -H 'Metadata-Flavor: Google' "$_md/scheduling/preemptible" 2>/dev/null)"
    fi
fi

B_CPU_MODEL=""; B_CPU_VENDOR=""; B_CPU_FAMILY=""; B_CPU_MODELID=""; B_CPU_STEPPING=""
B_SOCKETS=""; B_CORES_PHYS=""; B_CPUS_LOG=""; B_TPC=""; B_SMT=""
B_FREQ_BASE=""; B_FREQ_MAX=""
B_FLAGS_HAVE=""; B_FLAGS_MISS=""; B_FLAGS_RAW=""
B_L1D=""; B_L1I=""; B_L2=""; B_L3=""
B_L3_INSTANCES=""; B_L3_SHARED=""; B_L3_PER_CORE=""; B_L3_TOTAL_MB=""
B_LLC_MB=""; B_LLC_WHAT=""
B_NUMA_NODES=""; B_NUMA_DETAIL=""; B_NUMA_DIST=""; B_NUMA_RECO=""
B_MEM_TOTAL_MB=""; B_MEM_AVAIL_MB=""; B_SWAP_MB=""; B_THP=""; B_HUGEPAGES=""
B_CG_CPU=""; B_CG_MEM=""
B_GOVERNOR=""; B_SCALING_DRIVER=""
B_PERFLEVELS=""
B_BW_THEO=""; B_BW_THEO_HOW=""; B_BW_THEO_SRC=""; B_DIMMS=""
B_MEMBW_JSON=""; B_MEMBW_TXT=""; B_MEMBW_NUMA=""

X86_FLAGS="avx avx2 fma f16c avx512f avx512bw avx512vl avx512dq avx512cd avx512_vnni avx512_bf16 amx_tile amx_int8 amx_bf16"
ARM_FLAGS="asimd asimdhp asimddp i8mm bf16 sve sve2 svei8mm svebf16 sme sme2"

flag_probe() {   # $1 = normalized haystack (spaces at the edges), $2 = flag list
    local hay="$1" f
    for f in $2; do
        case "$hay" in
            *" $f "*) B_FLAGS_HAVE="$B_FLAGS_HAVE$f
" ;;
            *) B_FLAGS_MISS="$B_FLAGS_MISS$f
" ;;
        esac
    done
}

collect_linux() {
    if have lscpu; then
        _ls="$(lscpu 2>/dev/null)"
        B_CPU_MODEL=$(printf '%s\n'  "$_ls" | awk -F: '/^Model name/     {sub(/^[ \t]+/,"",$2); print $2; exit}')
        B_CPU_VENDOR=$(printf '%s\n' "$_ls" | awk -F: '/^Vendor ID/      {sub(/^[ \t]+/,"",$2); print $2; exit}')
        B_CPU_FAMILY=$(printf '%s\n' "$_ls" | awk -F: '/^CPU family/     {sub(/^[ \t]+/,"",$2); print $2; exit}')
        B_CPU_MODELID=$(printf '%s\n' "$_ls" | awk -F: '/^Model:/        {sub(/^[ \t]+/,"",$2); print $2; exit}')
        B_CPU_STEPPING=$(printf '%s\n' "$_ls" | awk -F: '/^Stepping/     {sub(/^[ \t]+/,"",$2); print $2; exit}')
        B_SOCKETS=$(printf '%s\n'    "$_ls" | awk -F: '/^Socket\(s\)/    {sub(/^[ \t]+/,"",$2); print $2; exit}')
        B_CPUS_LOG=$(printf '%s\n'   "$_ls" | awk -F: '/^CPU\(s\):/      {sub(/^[ \t]+/,"",$2); print $2; exit}')
        B_TPC=$(printf '%s\n'        "$_ls" | awk -F: '/^Thread\(s\) per core/ {sub(/^[ \t]+/,"",$2); print $2; exit}')
        _cps=$(printf '%s\n'         "$_ls" | awk -F: '/^Core\(s\) per socket/ {sub(/^[ \t]+/,"",$2); print $2; exit}')
        B_FREQ_MAX=$(printf '%s\n'   "$_ls" | awk -F: '/^CPU max MHz/    {printf "%.0f", $2; exit}')
        B_FREQ_BASE=$(printf '%s\n'  "$_ls" | awk -F: '/^CPU min MHz/    {printf "%.0f", $2; exit}')
        [ -n "${_cps:-}" ] && [ -n "${B_SOCKETS:-}" ] && B_CORES_PHYS=$((_cps * B_SOCKETS))
    else
        degraded "lscpu absent -> CPU identity read from /proc/cpuinfo (fewer fields)"
    fi
    [ -z "$B_CPU_MODEL" ] && B_CPU_MODEL=$(awk -F: '/^model name|^Model|^CPU implementer/{sub(/^[ \t]+/,"",$2); print $2; exit}' /proc/cpuinfo 2>/dev/null)
    [ -z "$B_CPUS_LOG" ]  && B_CPUS_LOG=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null)
    if [ -z "$B_CORES_PHYS" ] && [ -d /sys/devices/system/cpu/cpu0/topology ]; then
        B_CORES_PHYS=$(cat /sys/devices/system/cpu/cpu*/topology/core_cpus_list \
                            /sys/devices/system/cpu/cpu*/topology/thread_siblings_list 2>/dev/null | sort -u | wc -l | tr -d ' ')
    fi

    _smtctl=$(rd /sys/devices/system/cpu/smt/control)
    _smtact=$(rd /sys/devices/system/cpu/smt/active)
    if [ -n "$_smtctl" ]; then
        case "$_smtctl" in
            on)                 B_SMT="on" ;;
            off|forceoff)       B_SMT="off ($_smtctl)" ;;
            notsupported|notimplemented) B_SMT="not supported" ;;
            *)                  B_SMT="$_smtctl" ;;
        esac
    elif [ -n "$_smtact" ]; then
        [ "$_smtact" = "1" ] && B_SMT="on" || B_SMT="off"
    elif [ -n "${B_TPC:-}" ]; then
        [ "$B_TPC" -gt 1 ] 2>/dev/null && B_SMT="on (from threads-per-core)" || B_SMT="off (from threads-per-core)"
    else
        B_SMT="unknown"; degraded "/sys/devices/system/cpu/smt absent -> SMT state inferred or unknown"
    fi
    [ -z "$B_TPC" ] && [ -n "$B_CORES_PHYS" ] && [ -n "$B_CPUS_LOG" ] && [ "$B_CORES_PHYS" -gt 0 ] 2>/dev/null \
        && B_TPC=$((B_CPUS_LOG / B_CORES_PHYS))

    _bf=$(rd /sys/devices/system/cpu/cpu0/cpufreq/base_frequency)
    [ -n "$_bf" ] && B_FREQ_BASE=$(awk -v k="$_bf" 'BEGIN{printf "%.0f", k/1000}')
    _mf=$(rd /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
    [ -n "$_mf" ] && [ -z "$B_FREQ_MAX" ] && B_FREQ_MAX=$(awk -v k="$_mf" 'BEGIN{printf "%.0f", k/1000}')

    B_FLAGS_RAW=$(awk -F: '/^flags|^Features/{sub(/^[ \t]+/,"",$2); print $2; exit}' /proc/cpuinfo 2>/dev/null)
    _hay=" $(printf '%s' "$B_FLAGS_RAW" | tr 'A-Z' 'a-z') "
    case "$B_ARCH" in
        x86_64|amd64) flag_probe "$_hay" "$X86_FLAGS" ;;
        aarch64|arm64) flag_probe "$_hay" "$ARM_FLAGS" ;;
        *) degraded "arch $B_ARCH not covered -> no reference flag list" ;;
    esac

    for d in /sys/devices/system/cpu/cpu0/cache/index*; do
        [ -r "$d/level" ] || continue
        _lv=$(rd "$d/level"); _ty=$(rd "$d/type"); _sz=$(rd "$d/size")
        case "$_lv:$_ty" in
            1:Data)        B_L1D="$_sz" ;;
            1:Instruction) B_L1I="$_sz" ;;
            2:*)           B_L2="$_sz" ;;
            3:*)           B_L3="$_sz" ;;
        esac
    done
    _l3u=$(for d in /sys/devices/system/cpu/cpu*/cache/index*; do
               [ -r "$d/level" ] || continue
               [ "$(rd "$d/level")" = "3" ] || continue
               printf '%s|%s\n' "$(rd "$d/size")" "$(rd "$d/shared_cpu_list")"
           done | sort -u)
    if [ -n "$_l3u" ]; then
        B_L3_INSTANCES=$(printf '%s\n' "$_l3u" | wc -l | tr -d ' ')
        B_L3_SHARED=$(printf '%s\n' "$_l3u" | awk -F'|' '{printf "%s shared by cpu %s\n", $1, $2}')
        B_L3_TOTAL_MB=$(printf '%s\n' "$_l3u" | awk -F'|' -v OFS='' '
            { s=$1; u=substr(s,length(s),1); n=s+0;
              if (u=="K"||u=="k") n=n/1024; else if (u=="G"||u=="g") n=n*1024;
              else if (u!="M"&&u!="m") n=n/1048576;
              t+=n } END{ printf "%.2f", t }')
        [ -z "$B_L3" ] && B_L3=$(printf '%s\n' "$_l3u" | head -1 | cut -d'|' -f1)
        B_LLC_MB="$B_L3_TOTAL_MB"
        B_LLC_WHAT="L3, sum of $B_L3_INSTANCES instance$([ "$B_L3_INSTANCES" = 1 ] && echo "" || echo s)"
    else
        [ -d /sys/devices/system/cpu/cpu0/cache ] && degraded "no L3 exposed in sysfs (a VM that does not publish the cache topology?)"
        B_LLC_MB=$(sz_to_mb "$B_L2")
        B_LLC_WHAT="L2; no L3 exposed, so the comparison against the working set is optimistic"
    fi

    if have numactl; then
        B_NUMA_DETAIL=$(numactl --hardware 2>/dev/null)
        B_NUMA_NODES=$(printf '%s\n' "$B_NUMA_DETAIL" | awk '/available:/{print $2; exit}')
        B_NUMA_DIST=$(printf '%s\n' "$B_NUMA_DETAIL" | sed -n '/node distances/,$p')
    fi
    if [ -z "$B_NUMA_NODES" ] && [ -d /sys/devices/system/node ]; then
        degraded "numactl absent -> NUMA read from /sys/devices/system/node (and without numactl you cannot PIN either)"
        B_NUMA_NODES=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l | tr -d ' ')
        B_NUMA_DETAIL=$(for n in /sys/devices/system/node/node[0-9]*; do
                            printf 'node %s  cpus: %s  mem: %s\n' "${n##*/node}" \
                                   "$(rd "$n/cpulist")" \
                                   "$(awk '/MemTotal/{print $4" "$5}' "$n/meminfo" 2>/dev/null)"
                        done)
        B_NUMA_DIST=$(for n in /sys/devices/system/node/node[0-9]*; do
                          printf 'node %s -> %s\n' "${n##*/node}" "$(rd "$n/distance")"
                      done)
    fi
    [ -z "$B_NUMA_NODES" ] && B_NUMA_NODES="1"

    B_MEM_TOTAL_MB=$(awk '/^MemTotal:/{printf "%.0f", $2/1024}'     /proc/meminfo 2>/dev/null)
    B_MEM_AVAIL_MB=$(awk '/^MemAvailable:/{printf "%.0f", $2/1024}' /proc/meminfo 2>/dev/null)
    B_SWAP_MB=$(awk '/^SwapTotal:/{printf "%.0f", $2/1024}'         /proc/meminfo 2>/dev/null)
    B_THP=$(rd /sys/kernel/mm/transparent_hugepage/enabled)
    _hp=$(awk '/^HugePages_Total:/{t=$2} /^Hugepagesize:/{s=$2" "$3} END{if(t!="")printf "%s x %s", t, s}' /proc/meminfo 2>/dev/null)
    B_HUGEPAGES="$_hp"

    _cg=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
    for _p in "/sys/fs/cgroup${_cg}" "/sys/fs/cgroup"; do
        [ -z "$B_CG_CPU" ] && [ -r "$_p/cpu.max" ] && B_CG_CPU="$(rd "$_p/cpu.max")"
        [ -z "$B_CG_MEM" ] && [ -r "$_p/memory.max" ] && B_CG_MEM="$(rd "$_p/memory.max")"
    done
    if [ -z "$B_CG_CPU" ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then   # cgroup v1
        B_CG_CPU="$(rd /sys/fs/cgroup/cpu/cpu.cfs_quota_us) $(rd /sys/fs/cgroup/cpu/cpu.cfs_period_us) (v1)"
    fi

    B_GOVERNOR=$(rd /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    B_SCALING_DRIVER=$(rd /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver)
    if [ -z "$B_GOVERNOR" ]; then
        B_GOVERNOR="not exposed"
        degraded "no cpufreq in sysfs: the hypervisor decides the frequency, not you (normal on many cloud VMs)"
    fi
}

collect_macos() {
    B_CPU_MODEL=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
    B_CPU_VENDOR=$(sysctl -n machdep.cpu.vendor 2>/dev/null)
    [ -z "$B_CPU_VENDOR" ] && B_CPU_VENDOR="Apple"
    B_CPU_FAMILY=$(sysctl -n hw.cpufamily 2>/dev/null)
    B_SOCKETS=1
    B_CPUS_LOG=$(sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null)
    B_CORES_PHYS=$(sysctl -n hw.physicalcpu 2>/dev/null)
    if [ -n "$B_CORES_PHYS" ] && [ -n "$B_CPUS_LOG" ] && [ "$B_CORES_PHYS" -gt 0 ] 2>/dev/null; then
        B_TPC=$((B_CPUS_LOG / B_CORES_PHYS))
        [ "$B_TPC" -gt 1 ] && B_SMT="on" || B_SMT="off (Apple Silicon has no SMT)"
    fi
    B_FREQ_MAX=$(awk -v h="$(sysctl -n hw.cpufrequency_max 2>/dev/null)" 'BEGIN{ if (h!="" && h+0>0) printf "%.0f", h/1000000 }')
    [ -z "$B_FREQ_MAX" ] && degraded "Apple Silicon does not publish hw.cpufrequency: base/max frequency unknown (not a problem: MHz are not compared across machines here)"

    _np=$(sysctl -n hw.nperflevels 2>/dev/null)
    if [ -n "$_np" ]; then
        _i=0
        while [ "$_i" -lt "$_np" ]; do
            B_PERFLEVELS="$B_PERFLEVELS$(printf '%s: %s physical cores, L1d %s, L2 %s (shared by %s cores)' \
                "$(sysctl -n hw.perflevel$_i.name 2>/dev/null)" \
                "$(sysctl -n hw.perflevel$_i.physicalcpu 2>/dev/null)" \
                "$(mb_h "$(sz_to_mb "$(sysctl -n hw.perflevel$_i.l1dcachesize 2>/dev/null)")")" \
                "$(mb_h "$(sz_to_mb "$(sysctl -n hw.perflevel$_i.l2cachesize 2>/dev/null)")")" \
                "$(sysctl -n hw.perflevel$_i.cpusperl2 2>/dev/null)")
"
            _i=$((_i + 1))
        done
    fi

    B_L1D=$(sysctl -n hw.l1dcachesize 2>/dev/null)
    B_L1I=$(sysctl -n hw.l1icachesize 2>/dev/null)
    B_L2=$(sysctl -n hw.l2cachesize 2>/dev/null)
    B_L3=$(sysctl -n hw.l3cachesize 2>/dev/null)
    [ "${B_L3:-0}" = "0" ] && B_L3=""
    if [ -n "$B_L3" ]; then
        B_L3_TOTAL_MB=$(sz_to_mb "$B_L3")
        B_LLC_MB="$B_L3_TOTAL_MB"
        B_LLC_WHAT="L3 (Mac Intel)"
    else
        B_L3_SHARED="Apple Silicon exposes no L3 (the SLC is shared with the GPU and not readable from sysctl); the useful level is the per-cluster L2."
        _l2max=0; _i=0
        while [ "$_i" -lt "${_np:-0}" ]; do
            _v=$(sysctl -n hw.perflevel$_i.l2cachesize 2>/dev/null)
            [ -n "$_v" ] && [ "$_v" -gt "$_l2max" ] 2>/dev/null && _l2max="$_v"
            _i=$((_i + 1))
        done
        [ "$_l2max" = "0" ] && _l2max="$B_L2"
        B_LLC_MB=$(sz_to_mb "$_l2max")
        B_LLC_WHAT="Performance-cluster L2; Apple Silicon has no L3"
    fi

    case "$B_ARCH" in
        arm64)
            _map="asimd:AdvSIMD asimdhp:FEAT_FP16 asimddp:FEAT_DotProd i8mm:FEAT_I8MM bf16:FEAT_BF16 sme:FEAT_SME sme2:FEAT_SME2"
            for _pair in $_map; do
                _f=${_pair%%:*}; _s=${_pair#*:}
                if [ "$(sysctl -n hw.optional.arm.$_s 2>/dev/null)" = "1" ]; then
                    B_FLAGS_HAVE="$B_FLAGS_HAVE$_f
"
                else
                    B_FLAGS_MISS="$B_FLAGS_MISS$_f
"
                fi
            done
            B_FLAGS_MISS="${B_FLAGS_MISS}sve
"
            degraded "SVE is not exposed by Apple in any form: it is absent by design, not 'not detected'"
            B_FLAGS_RAW="hw.optional.arm.FEAT_*"
            ;;
        x86_64)
            B_FLAGS_RAW="$(sysctl -n machdep.cpu.features 2>/dev/null) $(sysctl -n machdep.cpu.leaf7_features 2>/dev/null)"
            _hay=" $(printf '%s' "$B_FLAGS_RAW" | tr 'A-Z.' 'a-z_') "
            flag_probe "$_hay" "$X86_FLAGS"
            ;;
    esac

    B_MEM_TOTAL_MB=$(awk -v b="$(sysctl -n hw.memsize 2>/dev/null)" 'BEGIN{ if (b!="") printf "%.0f", b/1048576 }')
    if have vm_stat; then
        B_MEM_AVAIL_MB=$(vm_stat 2>/dev/null | awk '
            /page size of/ { for(i=1;i<=NF;i++) if ($i=="of") ps=$(i+1) }
            /Pages free/       { gsub(/\./,"",$3); f=$3 }
            /Pages inactive/   { gsub(/\./,"",$3); iv=$3 }
            /Pages speculative/{ gsub(/\./,"",$3); s=$3 }
            END { if (ps!="") printf "%.0f", (f+iv+s)*ps/1048576 }')
    fi
    B_SWAP_MB=$(sysctl -n vm.swapusage 2>/dev/null | awk '{ s=$3; sub(/M$/,"",s); printf "%.0f", s+0 }')

    B_NUMA_NODES="1"
    B_NUMA_DETAIL="NUMA not applicable on macOS (unified memory, a single domain)"
    B_NUMA_RECO="no pinning: unified memory, a single domain"
    B_GOVERNOR="n/a (macOS exposes no governor; the SoC manages the frequency)"
    B_THP="n/a (macOS has no configurable THP)"
}

collect_bw_theoretical() {
    if have dmidecode; then
        _dmi=$(dmidecode -t memory 2>/dev/null)
        if [ -n "$_dmi" ]; then
            B_DIMMS=$(printf '%s\n' "$_dmi" | awk '
                /^Memory Device/       { insl=1; sz=""; sp=""; ty=""; dw=""; next }
                insl && /^\tSize:/      { sub(/^\tSize: /,""); sz=$0 }
                insl && /^\tType:/      { sub(/^\tType: /,""); ty=$0 }
                insl && /^\tSpeed:/     { sub(/^\tSpeed: /,""); if (sp=="") sp=$0 }
                insl && /^\tData Width:/{ sub(/^\tData Width: /,""); dw=$0 }
                insl && /^$/           { if (sz!="" && sz !~ /No Module/) printf "%s %s @ %s (%s)\n", sz, ty, sp, dw; insl=0 }
                END { if (insl && sz!="" && sz !~ /No Module/) printf "%s %s @ %s (%s)\n", sz, ty, sp, dw }')
            if [ -n "$B_DIMMS" ]; then
                B_BW_THEO=$(printf '%s\n' "$_dmi" | awk '
                    /^Memory Device/  { d=1; sz=""; mts=0; dw=64; next }
                    d && /Size:/      { if ($0 ~ /No Module/) sz=""; else sz=$2 }
                    d && /Speed:/     { if (mts==0) for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) { mts=$i+0; break } }
                    d && /Data Width:/{ for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) { dw=$i+0; break } }
                    d && /^$/         { if (sz!="" && mts>0) tot += mts*dw/8/1000; d=0 }
                    END { if (d && sz!="" && mts>0) tot += mts*dw/8/1000
                          if (tot>0) printf "%.0f", tot }')
                B_BW_THEO_SRC="smbios"
                B_BW_THEO_HOW="from dmidecode: $(printf '%s\n' "$B_DIMMS" | wc -l | tr -d ' ') populated modules (channels x MT/s x width)"
            fi
        else
            degraded "dmidecode present but not readable (needs root): theoretical bandwidth inferred, not read"
        fi
    fi
    if [ -z "$B_BW_THEO" ] && [ -d /sys/devices/system/edac/mc ]; then
        _nch=$(ls -d /sys/devices/system/edac/mc/mc*/dimm* 2>/dev/null | wc -l | tr -d ' ')
        [ "${_nch:-0}" -gt 0 ] && B_BW_THEO_HOW="EDAC sees $_nch populated DIMMs but does not publish their speed"
    fi
    if [ -z "$B_BW_THEO" ] && [ -n "$B_GCP_MT" ]; then
        case "$B_GCP_MT" in
            c4a-*)        B_BW_THEO="";    B_BW_THEO_HOW="c4a (Axion/Neoverse-V2): DDR5, channel count not published by Google -> no honest estimate" ;;
            c4-*)         B_BW_THEO="358"; B_BW_THEO_HOW="c4 (Emerald Rapids): DDR5-5600 x 8 canali per socket" ;;
            c3d-*)        B_BW_THEO="460"; B_BW_THEO_HOW="c3d (Genoa/Zen4): DDR5-4800 x 12 canali per socket" ;;
            c3-*)         B_BW_THEO="307"; B_BW_THEO_HOW="c3 (Sapphire Rapids): DDR5-4800 x 8 canali per socket" ;;
            n2d-*|c2d-*)  B_BW_THEO="204"; B_BW_THEO_HOW="n2d/c2d (Milan/Zen3): DDR4-3200 x 8 canali per socket" ;;
            t2a-*)        B_BW_THEO="204"; B_BW_THEO_HOW="t2a (Ampere Altra): DDR4-3200 x 8 canali per socket" ;;
            n2-*|c2-*)    B_BW_THEO="205"; B_BW_THEO_HOW="n2/c2 (Cascade/Ice Lake): DDR4-3200 x ~8 canali per socket" ;;
            *)            B_BW_THEO_HOW="family '$B_GCP_MT' has no published memory configuration -> no estimate" ;;
        esac
        [ -n "$B_BW_THEO" ] && B_BW_THEO_SRC="cloud-family"
    fi
    if [ -z "$B_BW_THEO" ] && [ "$B_OS" = "Darwin" ]; then
        case "$B_CPU_MODEL" in
            *"M1 Ultra"*) B_BW_THEO="800" ;; *"M1 Max"*) B_BW_THEO="400" ;; *"M1 Pro"*) B_BW_THEO="200" ;; *"M1"*) B_BW_THEO="68" ;;
            *"M2 Ultra"*) B_BW_THEO="800" ;; *"M2 Max"*) B_BW_THEO="400" ;; *"M2 Pro"*) B_BW_THEO="200" ;; *"M2"*) B_BW_THEO="100" ;;
            *"M3 Ultra"*) B_BW_THEO="800" ;; *"M3 Max"*) B_BW_THEO="400" ;; *"M3 Pro"*) B_BW_THEO="150" ;; *"M3"*) B_BW_THEO="100" ;;
            *"M4 Max"*)   B_BW_THEO="546" ;; *"M4 Pro"*) B_BW_THEO="273" ;; *"M4"*) B_BW_THEO="120" ;;
        esac
        [ -n "$B_BW_THEO" ] && { B_BW_THEO_SRC="soc-table"; B_BW_THEO_HOW="Apple SoC table ($B_CPU_MODEL): Apple does not expose the bandwidth through sysctl"; }
    fi
    [ -z "$B_BW_THEO_SRC" ] && B_BW_THEO_SRC="unknown"
    [ -z "$B_BW_THEO_HOW" ] && B_BW_THEO_HOW="no readable source (neither SMBIOS nor a known cloud family)"
}

collect_membw() {
    [ -n "$MEMBW_BIN" ] && [ -x "$MEMBW_BIN" ] || {
        B_MEMBW_TXT="not measured (pass --membw <binary from tests/membw.c>, or use 'make server-hw-check', which builds it)"
        return
    }
    _l3arg=$(awk -v m="${B_LLC_MB:-32}" 'BEGIN{ printf "%d", (m<1?32:m) }')
    B_MEMBW_JSON="$("$MEMBW_BIN" --l3-mb "$_l3arg" --reps "$MEMBW_REPS" --json 2>/dev/null)"
    if [ -n "$B_MEMBW_JSON" ] && have python3; then
        B_MEMBW_TXT=$(printf '%s' "$B_MEMBW_JSON" | python3 -c '
import json, sys
d = json.load(sys.stdin)
out = ["array %d MiB x3 (>= 4x L3), best of %d, %d cpu viste"
       % (d["array_mib_per_buffer"], d["reps"], d["cpus_seen"]),
       "  %-8s %12s %12s" % ("threads", "Copy GB/s", "Triad GB/s")]
for r in d["sweep"]:
    out.append("  %-8d %12.1f %12.1f" % (r["threads"], r["copy_gbs"], r["triad_gbs"]))
out.append("  peak Triad %.1f GB/s at %d threads; KNEE at %d threads (95%% of the peak)"
           % (d["peak_triad_gbs"], d["peak_triad_threads"], d["knee_threads"]))
print("\n".join(out))' 2>/dev/null)
    fi
    [ -z "$B_MEMBW_TXT" ] && B_MEMBW_TXT="$("$MEMBW_BIN" --l3-mb "$_l3arg" --reps "$MEMBW_REPS" 2>&1)"

    if [ "${B_NUMA_NODES:-1}" -gt 1 ] 2>/dev/null && have numactl; then
        _loc=$(numactl --cpunodebind=0 --membind=0 "$MEMBW_BIN" --l3-mb "$_l3arg" --reps "$MEMBW_REPS" --json --label numa-local 2>/dev/null)
        _rem=$(numactl --cpunodebind=0 --membind=1 "$MEMBW_BIN" --l3-mb "$_l3arg" --reps "$MEMBW_REPS" --json --label numa-cross 2>/dev/null)
        B_MEMBW_NUMA="$_loc
$_rem"
    elif [ "${B_NUMA_NODES:-1}" -gt 1 ] 2>/dev/null; then
        B_MEMBW_NUMA=""
        degraded "several NUMA nodes but numactl is absent: local-vs-cross cannot be measured (install numactl: it is also the only way to pin)"
    fi
}

case "$B_OS" in
    Linux)  collect_linux ;;
    Darwin) collect_macos ;;
    *)      degraded "OS '$B_OS' not handled: minimal collection"; B_CPUS_LOG=$(getconf _NPROCESSORS_ONLN 2>/dev/null) ;;
esac
collect_bw_theoretical
collect_membw

WS_CP_06B_MB=60
WS_CP_17B_MB=120

REC_THREADS="$B_CORES_PHYS"
if [ "$B_OS" = "Darwin" ] && [ -n "$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null)" ]; then
    REC_THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null)   # P-cores only
fi
[ -z "$REC_THREADS" ] && REC_THREADS="$B_CPUS_LOG"

GATE_SMT="PASS"; GATE_GOV="PASS"; GATE_CG="PASS"; GATE_ALL="PASS"
case "${B_SMT:-}" in on*) GATE_SMT="FAIL" ;; esac
case "${B_GOVERNOR:-}" in performance|n/a*|"not exposed") : ;; *) GATE_GOV="FAIL" ;; esac
if [ -n "${B_CG_CPU:-}" ] && [ "${B_CG_CPU%% *}" != "max" ] && [ "${B_CG_CPU%% *}" != "-1" ]; then GATE_CG="FAIL"; fi
[ "$GATE_SMT$GATE_GOV$GATE_CG" = "PASSPASSPASS" ] || GATE_ALL="FAIL"

case "${B_SMT:-}" in
    on*) warn "SMT IS ON: $B_CPUS_LOG vCPU = ${B_CORES_PHYS:-?} real cores. On GCP recreate the box with --threads-per-core=1, or use -j${REC_THREADS} and declare it — otherwise the measurement describes two hyperthreads contending for one FMA." ;;
esac
case "${B_GOVERNOR:-}" in
    performance|n/a*|"not exposed") : ;;
    *) warn "governor = '$B_GOVERNOR' (not 'performance'): the frequency scales under load and the timings wander. sudo cpupower frequency-set -g performance, or declare that the numbers carry that noise." ;;
esac
if [ -n "${B_NUMA_NODES:-}" ] && [ "$B_NUMA_NODES" -gt 1 ] 2>/dev/null; then
    B_NUMA_RECO="$B_NUMA_NODES nodes: PIN the process to one node — numactl --cpunodebind=0 --membind=0 ./qwen_tts ... — otherwise half the weight accesses cross the interconnect and the variance eats the effect you are measuring."
    warn "NUMA with $B_NUMA_NODES nodes: without pinning every cell of the matrix carries noise you do not control."
elif [ -z "$B_NUMA_RECO" ]; then
    B_NUMA_RECO="1 node: no pinning needed."
fi
if [ -n "${B_CG_CPU:-}" ] && [ "${B_CG_CPU%% *}" != "max" ] && [ "${B_CG_CPU%% *}" != "-1" ]; then
    warn "a cgroup CPU quota is active ('$B_CG_CPU'): you are measuring the QUOTA, not the machine. On a dedicated box it should not be there."
fi
if [ -n "${B_SWAP_MB:-}" ] && [ "$B_SWAP_MB" -gt 0 ] 2>/dev/null; then
    warn "swap is on (${B_SWAP_MB} MiB): if the server RSS approaches RAM, a slow cell can be paging rather than the kernel under test."
fi

if [ -n "$B_LLC_MB" ] && [ -n "${B_CORES_PHYS:-}" ] && [ "$B_CORES_PHYS" -gt 0 ] 2>/dev/null; then
    B_L3_PER_CORE=$(awk -v t="$B_LLC_MB" -v c="$B_CORES_PHYS" 'BEGIN{printf "%.2f", t/c}')
fi
BW_PEAK=""; BW_KNEE=""
if [ -n "$B_MEMBW_JSON" ] && have python3; then
    BW_PEAK=$(printf '%s' "$B_MEMBW_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["peak_triad_gbs"])' 2>/dev/null)
    BW_KNEE=$(printf '%s' "$B_MEMBW_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["knee_threads"])' 2>/dev/null)
fi
if [ -n "$BW_PEAK" ] && [ -n "$B_BW_THEO" ]; then
    _ratio=$(awk -v m="$BW_PEAK" -v t="$B_BW_THEO" 'BEGIN{ if (t>0) printf "%.0f", 100*m/t }')
    [ -n "$_ratio" ] && [ "$_ratio" -lt 60 ] 2>/dev/null && \
        warn "measured bandwidth ${BW_PEAK} GB/s = ${_ratio}% of the theoretical estimate ($B_BW_THEO GB/s): normal on a VM that sees a slice of the socket, but from here on use the MEASURED one — the estimate is the ceiling of the physical server, not of your instance."
fi

FIT_06B="unknown"; FIT_17B="unknown"
if [ -n "$B_LLC_MB" ]; then
    FIT_06B=$(awk -v l="$B_LLC_MB" -v w="$WS_CP_06B_MB" 'BEGIN{print (l>=w) ? "FITS" : "does NOT fit"}')
    FIT_17B=$(awk -v l="$B_LLC_MB" -v w="$WS_CP_17B_MB" 'BEGIN{print (l>=w) ? "FITS" : "does NOT fit"}')
fi

JOUT=$(jobj \
        schema        "$(jstr 'box_info/1')" \
        collected_at  "$(jstr "$B_DATE")" \
        host          "$(jobj hostname "$(jstr "$B_HOST")" os "$(jstr "$B_OS")" kernel "$(jstr "$B_KERNEL")" arch "$(jstr "$B_ARCH")")" \
        gcp           "$(jobj machine_type "$(jstr "$B_GCP_MT")" zone "$(jstr "$B_GCP_ZONE")" preemptible "$(jstr "$B_GCP_PREEMPT")")" \
        cpu           "$(jobj model "$(jstr "$B_CPU_MODEL")" vendor "$(jstr "$B_CPU_VENDOR")" family "$(jstr "$B_CPU_FAMILY")" \
                              model_id "$(jstr "$B_CPU_MODELID")" stepping "$(jstr "$B_CPU_STEPPING")" \
                              sockets "$(jnum "$B_SOCKETS")" cores_physical "$(jnum "$B_CORES_PHYS")" \
                              cpus_logical "$(jnum "$B_CPUS_LOG")" threads_per_core "$(jnum "$B_TPC")" \
                              smt "$(jstr "$B_SMT")" freq_base_mhz "$(jnum "$B_FREQ_BASE")" freq_max_mhz "$(jnum "$B_FREQ_MAX")" \
                              perflevels "$(jarr "$B_PERFLEVELS")")" \
        flags         "$(jobj have "$(jarr "$B_FLAGS_HAVE")" missing "$(jarr "$B_FLAGS_MISS")")" \
        cache         "$(jobj l1d "$(jstr "$B_L1D")" l1i "$(jstr "$B_L1I")" l2 "$(jstr "$B_L2")" l3 "$(jstr "$B_L3")" \
                              l1d_mb "$(jnum "$(sz_to_mb "$B_L1D")")" l2_mb "$(jnum "$(sz_to_mb "$B_L2")")" \
                              l3_instances "$(jnum "$B_L3_INSTANCES")" l3_total_mb "$(jnum "$B_L3_TOTAL_MB")" \
                              llc_mb "$(jnum "$B_LLC_MB")" llc_what "$(jstr "$B_LLC_WHAT")" \
                              llc_per_core_mb "$(jnum "$B_L3_PER_CORE")" l3_shared "$(jarr "$B_L3_SHARED")")" \
        numa          "$(jobj nodes "$(jnum "$B_NUMA_NODES")" detail "$(jarr "$B_NUMA_DETAIL")" \
                              distances "$(jarr "$B_NUMA_DIST")" recommendation "$(jstr "$B_NUMA_RECO")")" \
        memory        "$(jobj total_mb "$(jnum "$B_MEM_TOTAL_MB")" available_mb "$(jnum "$B_MEM_AVAIL_MB")" \
                              swap_mb "$(jnum "$B_SWAP_MB")" thp "$(jstr "$B_THP")" hugepages "$(jstr "$B_HUGEPAGES")")" \
        limits        "$(jobj cgroup_cpu_max "$(jstr "$B_CG_CPU")" cgroup_memory_max "$(jstr "$B_CG_MEM")")" \
        cpufreq       "$(jobj governor "$(jstr "$B_GOVERNOR")" driver "$(jstr "$B_SCALING_DRIVER")")" \
        memory_bandwidth "$(jobj \
                              theoretical_gbs "$(jnum "$B_BW_THEO")" \
                              theoretical_source "$(jstr "$B_BW_THEO_SRC")" \
                              theoretical_note "$(jstr "$B_BW_THEO_HOW")" \
                              theoretical_is_estimate true \
                              dimms "$(jarr "$B_DIMMS")" \
                              measured "${B_MEMBW_JSON:-null}" \
                              measured_peak_triad_gbs "$(jnum "$BW_PEAK")" \
                              measured_knee_threads "$(jnum "$BW_KNEE")" \
                              numa_cells "[$(printf '%s' "$B_MEMBW_NUMA" | awk 'NF{ printf "%s%s", (n++?",":""), $0 }')]")" \
        gates         "$(jobj smt_off "$(jstr "$GATE_SMT")" governor_performance "$(jstr "$GATE_GOV")" \
                              no_cgroup_quota "$(jstr "$GATE_CG")" all "$(jstr "$GATE_ALL")" \
                              note "$(jstr 'if any of these is FAIL, every server/batching number taken afterwards describes a different machine')")" \
        bench_advice  "$(jobj threads "$(jnum "$REC_THREADS")" numa "$(jstr "$B_NUMA_RECO")" \
                              membw_knee_threads "$(jnum "$BW_KNEE")" \
                              membw_peak_triad_gbs "$(jnum "$BW_PEAK")" \
                              llc_per_core_mb "$(jnum "$B_L3_PER_CORE")" \
                              cp_ws_06b_int8_mb "$(jnum "$WS_CP_06B_MB")" cp_ws_17b_int8_mb "$(jnum "$WS_CP_17B_MB")" \
                              llc_fits_06b "$(jstr "$FIT_06B")" llc_fits_17b "$(jstr "$FIT_17B")")" \
        warnings      "$(jarr "$WARN")" \
        degraded      "$(jarr "$DEGRADED")")
if have python3; then
    OUT_PRETTY=$(printf '%s' "$JOUT" | python3 -m json.tool 2>/dev/null) || OUT_PRETTY="$JOUT"
else
    OUT_PRETTY="$JOUT"
fi
[ -n "$OUT" ] && printf '%s\n' "$OUT_PRETTY" > "$OUT"
if [ "$JSON" = "1" ]; then
    printf '%s\n' "$OUT_PRETTY"
    exit 0
fi

hr()  { printf '%.0s─' $(seq 1 76); echo; }
sec() { echo; printf '── %s\n' "$1"; }
kv()  { printf '   %-22s %s\n' "$1" "${2:-n/a}"; }

hr
printf '  BOX INFO   %s   %s %s (%s)   %s\n' "${B_HOST:-?}" "$B_OS" "$B_KERNEL" "$B_ARCH" "$B_DATE"
hr

echo
printf '   GATE  SMT off ................. %s   %s\n' "$GATE_SMT" \
    "$([ "$GATE_SMT" = PASS ] && echo 'vCPU = core: -j reads directly' || echo "$B_CPUS_LOG vCPU = ${B_CORES_PHYS:-?} cores -> recreate the box with --threads-per-core=1")"
printf '   GATE  governor performance .... %s   %s\n' "$GATE_GOV" \
    "$([ "$GATE_GOV" = PASS ] && echo 'frequency stays put under load' || echo "it is '$B_GOVERNOR': the frequency scales and the timings wander")"
printf '   GATE  no cgroup quota ......... %s   %s\n' "$GATE_CG" \
    "$([ "$GATE_CG" = PASS ] && echo 'you are measuring the machine' || echo "cpu.max='$B_CG_CPU': you are measuring the QUOTA")"
if [ "$GATE_ALL" = "PASS" ]; then
    echo "   → the three invalidating conditions are clear: what follows describes THIS machine."
else
    echo "   → ⛔ at least one invalidating condition is red. Fix it BEFORE measuring batching or a server:"
    echo "        a number taken now is not slightly wrong, it describes a different machine."
fi

if [ -n "$WARN" ]; then
    echo
    printf '%s\n' "$WARN" | while IFS= read -r l; do [ -n "$l" ] && printf '   ⚠️  %s\n' "$l"; done
else
    echo
    echo "   ✅ no warnings: SMT, governor, NUMA and quotas are in the state that makes a measurement readable."
fi

if [ -n "$B_GCP_MT" ]; then
    sec "GCP"
    kv "machine-type" "$B_GCP_MT"
    kv "zone"         "$B_GCP_ZONE"
    kv "preemptible"  "${B_GCP_PREEMPT:-false}  (spot = fine for capacity, NOT for a soak)"
fi

sec "CPU"
kv "model"          "$B_CPU_MODEL"
kv "vendor"         "$B_CPU_VENDOR"
kv "family/model"   "${B_CPU_FAMILY:-n/a}/${B_CPU_MODELID:-n/a}  stepping ${B_CPU_STEPPING:-n/a}"
kv "sockets"        "$B_SOCKETS"
kv "PHYSICAL cores" "${B_CORES_PHYS:-n/a}          <-- this is the number to use for -j"
kv "logical cpus"   "${B_CPUS_LOG:-n/a}          (nproc; on GCP a vCPU is a hyperthread)"
kv "threads/core"   "$B_TPC"
kv "SMT"            "$B_SMT"
kv "freq base/max"  "${B_FREQ_BASE:-n/a} / ${B_FREQ_MAX:-n/a} MHz"
if [ -n "$B_PERFLEVELS" ]; then
    printf '%s\n' "$B_PERFLEVELS" | while IFS= read -r l; do [ -n "$l" ] && printf '   %-22s %s\n' "" "$l"; done
fi

sec "ISA — the extensions that decide WHICH GEMM runs"
printf '   present:   %s\n' "$(printf '%s' "$B_FLAGS_HAVE" | tr '\n' ' ')"
printf '   absent:    %s\n' "$(printf '%s' "$B_FLAGS_MISS" | tr '\n' ' ')"
echo   "   (note: in /proc/cpuinfo VNNI and BF16 carry the underscore — avx512_vnni, avx512_bf16 —"
echo   "    while avx512f/bw/vl/dq do not. The gcc flags are the opposite. Do not mix them up.)"
echo   "   → whether the BINARY actually uses them is answered by ./qwen_tts --caps, not by this list."

sec "Cache — the bottleneck is the Code Predictor re-reading the weights 16x per frame"
kv "L1d / L1i" "$(mb_h "$(sz_to_mb "$B_L1D")") / $(mb_h "$(sz_to_mb "$B_L1I")")"
kv "L2"        "$(mb_h "$(sz_to_mb "$B_L2")")"
[ -n "$B_L3" ] && kv "L3" "$(mb_h "$(sz_to_mb "$B_L3")")${B_L3_INSTANCES:+  x $B_L3_INSTANCES}"
kv "usable LLC"  "$(mb_h "$B_LLC_MB")  (${B_LLC_WHAT:-n/a})"
[ -n "$B_L3_PER_CORE" ] && kv "LLC per core" "$(mb_h "$B_L3_PER_CORE") per physical core"
if [ -n "$B_L3_SHARED" ]; then
    printf '%s\n' "$B_L3_SHARED" | while IFS= read -r l; do [ -n "$l" ] && printf '   %-22s %s\n' "" "$l"; done
fi

sec "NUMA"
printf '%s' "${B_NUMA_DETAIL:-n/a}" | sed 's/^/   /'
echo
[ -n "$B_NUMA_DIST" ] && { printf '%s' "$B_NUMA_DIST" | sed 's/^/   /'; echo; }
printf '   → %s\n' "$B_NUMA_RECO"

sec "Memory"
kv "total"      "$(mb_h "$B_MEM_TOTAL_MB")"
kv "available"  "$(mb_h "$B_MEM_AVAIL_MB")"
kv "swap"       "$(mb_h "$B_SWAP_MB")"
kv "THP"        "$B_THP"
kv "hugepages"  "$B_HUGEPAGES"

sec "Memory bandwidth — the workload is memory-bound, so THIS is the number"
kv "theoretical (ESTIMATE)" "${B_BW_THEO:-n/a}${B_BW_THEO:+ GB/s}"
printf '   %-22s %s\n' "" "source: $B_BW_THEO_HOW"
printf '   %-22s %s\n' "" "⚠️ it is a CEILING of the physical server, not a measurement of your instance."
if [ -n "$B_DIMMS" ]; then
    printf '%s\n' "$B_DIMMS" | while IFS= read -r l; do [ -n "$l" ] && printf '   %-22s %s\n' "" "$l"; done
fi
echo
if [ -n "$B_MEMBW_TXT" ]; then
    printf '%s' "$B_MEMBW_TXT" | sed 's/^/   /'
    echo
fi
if [ -n "$B_MEMBW_NUMA" ] && have python3; then
    echo "   NUMA-local against cross-NUMA (the difference IS the cost of the interconnect):"
    printf '%s\n' "$B_MEMBW_NUMA" | while IFS= read -r l; do
        [ -z "$l" ] && continue
        printf '%s' "$l" | python3 -c '
import json,sys
d = json.load(sys.stdin)
print("     %-12s peak Triad %6.1f GB/s at %d threads (knee %d)" %
      (d["label"], d["peak_triad_gbs"], d["peak_triad_threads"], d["knee_threads"]))' 2>/dev/null
    done
elif [ "${B_NUMA_NODES:-1}" -le 1 ] 2>/dev/null; then
    echo "   NUMA-local vs cross-NUMA: not applicable, there is a single node."
fi

sec "Limits (container / VM) and frequency"
kv "cgroup cpu.max"    "${B_CG_CPU:-none}"
kv "cgroup memory.max" "${B_CG_MEM:-none}"
kv "governor"          "$B_GOVERNOR"
kv "scaling driver"    "$B_SCALING_DRIVER"

if [ -n "$DEGRADED" ]; then
    sec "Degraded (missing commands/files — a field left empty, not invented)"
    printf '%s\n' "$DEGRADED" | while IFS= read -r l; do [ -n "$l" ] && printf '   · %s\n' "$l"; done
fi

sec "WHAT THIS MEANS FOR THE BENCH"
case "${B_SMT:-}" in
    on*) ADV_SMT="ON -> $B_CPUS_LOG vCPU are ${B_CORES_PHYS:-?} cores. Recreate the box with --threads-per-core=1: that is the difference between measuring the machine and measuring hyperthreading." ;;
    *)   ADV_SMT="off/absent -> vCPU = core, the -j numbers read directly." ;;
esac
case "${B_GOVERNOR:-}" in
    performance|n/a*|"not exposed") ADV_FREQ="governor $B_GOVERNOR -> stable timings." ;;
    *) ADV_FREQ="governor $B_GOVERNOR -> NOT performance: change it (cpupower frequency-set -g performance) or declare the noise." ;;
esac
ADV_CORE="PHYSICAL cores"
[ "$B_OS" = "Darwin" ] && ADV_CORE="Performance cores (the E-cores are ~3x slower and become the frame time themselves)"

if [ -n "$BW_KNEE" ]; then
    ADV_BW="measured bandwidth: peak ${BW_PEAK} GB/s, KNEE at ${BW_KNEE} threads.
               Past ${BW_KNEE} threads the bandwidth stops rising: on a workload that re-reads
               the weights 16x per frame, give ~${BW_KNEE} threads to MORE requests (B=2..8)
               rather than every core to one. That is the -j x --batch-size choice."
else
    ADV_BW="bandwidth NOT measured (pass --membw, or use make server-hw-check): without the
               knee, the choice between a high -j and a high batch stays guesswork."
fi

cat <<ADVICE
   1. threads  use -j$REC_THREADS = the $ADV_CORE.
               Beyond that, two threads contend for the same FMA: per-request throughput
               drops without Q rising, and it looks like a defect in the batching.
   1.bis bw    $ADV_BW
   2. SMT      $ADV_SMT
   3. NUMA     $B_NUMA_RECO
   4. cache    LLC $(mb_h "$B_LLC_MB") ($B_LLC_WHAT), $(mb_h "$B_L3_PER_CORE") per core.
               The int8 Code Predictor re-reads ~$WS_CP_06B_MB MB per frame on the 0.6B ($FIT_06B)
               and ~$WS_CP_17B_MB MB on the 1.7B ($FIT_17B). Where it does NOT fit, bandwidth is
               the limit: expect batching to pay more than the ISA, and int4 to cost more than AVX-512.
   5. freq     $ADV_FREQ
   6. order    this report, then ./qwen_tts --caps and --self-test (make server-hw-check), and ONLY
               then any server number. A --caps that does not declare the expected primitive
               active means everything after it measures a fallback, not this machine. And a
               red --self-test invalidates the whole matrix.
ADVICE
echo
hr
