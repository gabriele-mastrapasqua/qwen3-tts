#!/usr/bin/env bash
set -u

JSON=0
MEMBW_BIN="${MEMBW_BIN:-}"      # binario di tests/membw.c; se assente la banda MISURATA si salta
OUT="${OUT:-}"                  # se valorizzato, ci scrive il JSON (oltre al report leggibile)
MEMBW_REPS="${MEMBW_REPS:-5}"
while [ $# -gt 0 ]; do
    case "$1" in
        --json)   JSON=1 ;;
        --membw)  MEMBW_BIN="${2:-}"; shift ;;
        --out)    OUT="${2:-}"; shift ;;
        -h|--help) sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "opzione sconosciuta: $1 (usa --json | --membw BIN | --out FILE)" >&2; exit 2 ;;
    esac
    shift
done

have() { command -v "$1" >/dev/null 2>&1; }
rd()   { [ -r "$1" ] && tr -d '\n' < "$1" 2>/dev/null; }        # file di /sys, una riga
rdm()  { [ -r "$1" ] && cat "$1" 2>/dev/null; }                 # file multi-riga

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
        else                     n = n/1048576;   # senza suffisso = byte
        printf "%.4f", n;                          # 2 decimali arrotondano via le L1
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

flag_probe() {   # $1 = haystack normalizzato (spazi ai bordi), $2 = elenco flag
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
        degraded "lscpu assente -> identita' CPU letta da /proc/cpuinfo (meno campi)"
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
            notsupported|notimplemented) B_SMT="non supportato" ;;
            *)                  B_SMT="$_smtctl" ;;
        esac
    elif [ -n "$_smtact" ]; then
        [ "$_smtact" = "1" ] && B_SMT="on" || B_SMT="off"
    elif [ -n "${B_TPC:-}" ]; then
        [ "$B_TPC" -gt 1 ] 2>/dev/null && B_SMT="on (da thread-per-core)" || B_SMT="off (da thread-per-core)"
    else
        B_SMT="ignoto"; degraded "/sys/devices/system/cpu/smt assente -> stato SMT dedotto o ignoto"
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
        *) degraded "arch $B_ARCH non prevista -> nessun elenco flag di riferimento" ;;
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
        B_L3_SHARED=$(printf '%s\n' "$_l3u" | awk -F'|' '{printf "%s condivisa da cpu %s\n", $1, $2}')
        B_L3_TOTAL_MB=$(printf '%s\n' "$_l3u" | awk -F'|' -v OFS='' '
            { s=$1; u=substr(s,length(s),1); n=s+0;
              if (u=="K"||u=="k") n=n/1024; else if (u=="G"||u=="g") n=n*1024;
              else if (u!="M"&&u!="m") n=n/1048576;
              t+=n } END{ printf "%.2f", t }')
        [ -z "$B_L3" ] && B_L3=$(printf '%s\n' "$_l3u" | head -1 | cut -d'|' -f1)
        B_LLC_MB="$B_L3_TOTAL_MB"
        B_LLC_WHAT="L3, somma di $B_L3_INSTANCES istanz$([ "$B_L3_INSTANCES" = 1 ] && echo a || echo e)"
    else
        [ -d /sys/devices/system/cpu/cpu0/cache ] && degraded "nessuna L3 esposta in sysfs (VM che non pubblica la topologia di cache?)"
        B_LLC_MB=$(sz_to_mb "$B_L2")
        B_LLC_WHAT="L2; nessuna L3 esposta, quindi il confronto col working set e' ottimistico"
    fi

    if have numactl; then
        B_NUMA_DETAIL=$(numactl --hardware 2>/dev/null)
        B_NUMA_NODES=$(printf '%s\n' "$B_NUMA_DETAIL" | awk '/available:/{print $2; exit}')
        B_NUMA_DIST=$(printf '%s\n' "$B_NUMA_DETAIL" | sed -n '/node distances/,$p')
    fi
    if [ -z "$B_NUMA_NODES" ] && [ -d /sys/devices/system/node ]; then
        degraded "numactl assente -> NUMA letta da /sys/devices/system/node (e senza numactl non si puo' nemmeno PINNARE)"
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
        B_GOVERNOR="non esposto"
        degraded "nessun cpufreq in sysfs: la frequenza la decide l'hypervisor, non tu (normale su molte VM cloud)"
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
        [ "$B_TPC" -gt 1 ] && B_SMT="on" || B_SMT="off (Apple Silicon non ha SMT)"
    fi
    B_FREQ_MAX=$(awk -v h="$(sysctl -n hw.cpufrequency_max 2>/dev/null)" 'BEGIN{ if (h!="" && h+0>0) printf "%.0f", h/1000000 }')
    [ -z "$B_FREQ_MAX" ] && degraded "Apple Silicon non pubblica hw.cpufrequency: frequenza base/max ignota (non e' un problema: qui non si confrontano MHz fra macchine)"

    _np=$(sysctl -n hw.nperflevels 2>/dev/null)
    if [ -n "$_np" ]; then
        _i=0
        while [ "$_i" -lt "$_np" ]; do
            B_PERFLEVELS="$B_PERFLEVELS$(printf '%s: %s core fisici, L1d %s, L2 %s (condivisa da %s core)' \
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
        B_L3_SHARED="Apple Silicon non espone una L3 (SLC condivisa con la GPU, non leggibile da sysctl); il livello utile e' la L2 per cluster."
        _l2max=0; _i=0
        while [ "$_i" -lt "${_np:-0}" ]; do
            _v=$(sysctl -n hw.perflevel$_i.l2cachesize 2>/dev/null)
            [ -n "$_v" ] && [ "$_v" -gt "$_l2max" ] 2>/dev/null && _l2max="$_v"
            _i=$((_i + 1))
        done
        [ "$_l2max" = "0" ] && _l2max="$B_L2"
        B_LLC_MB=$(sz_to_mb "$_l2max")
        B_LLC_WHAT="L2 del cluster Performance; Apple Silicon non ha L3"
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
            degraded "SVE non e' esposto da Apple in nessuna forma: e' assente per progetto, non 'non rilevato'"
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
    B_NUMA_DETAIL="NUMA non applicabile su macOS (memoria unificata, un solo dominio)"
    B_NUMA_RECO="nessun pinning: memoria unificata, un solo dominio"
    B_GOVERNOR="n/a (macOS non espone un governor; la frequenza la gestisce il SoC)"
    B_THP="n/a (macOS non ha THP configurabile)"
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
                B_BW_THEO_HOW="da dmidecode: $(printf '%s\n' "$B_DIMMS" | wc -l | tr -d ' ') moduli popolati (canali x MT/s x larghezza)"
            fi
        else
            degraded "dmidecode presente ma non leggibile (serve root): banda teorica dedotta, non letta"
        fi
    fi
    if [ -z "$B_BW_THEO" ] && [ -d /sys/devices/system/edac/mc ]; then
        _nch=$(ls -d /sys/devices/system/edac/mc/mc*/dimm* 2>/dev/null | wc -l | tr -d ' ')
        [ "${_nch:-0}" -gt 0 ] && B_BW_THEO_HOW="EDAC vede $_nch DIMM popolati, ma non ne pubblica la velocita'"
    fi
    if [ -z "$B_BW_THEO" ] && [ -n "$B_GCP_MT" ]; then
        case "$B_GCP_MT" in
            c4a-*)        B_BW_THEO="";    B_BW_THEO_HOW="c4a (Axion/Neoverse-V2): DDR5, canali non pubblicati da Google -> nessuna stima onesta" ;;
            c4-*)         B_BW_THEO="358"; B_BW_THEO_HOW="c4 (Emerald Rapids): DDR5-5600 x 8 canali per socket" ;;
            c3d-*)        B_BW_THEO="460"; B_BW_THEO_HOW="c3d (Genoa/Zen4): DDR5-4800 x 12 canali per socket" ;;
            c3-*)         B_BW_THEO="307"; B_BW_THEO_HOW="c3 (Sapphire Rapids): DDR5-4800 x 8 canali per socket" ;;
            n2d-*|c2d-*)  B_BW_THEO="204"; B_BW_THEO_HOW="n2d/c2d (Milan/Zen3): DDR4-3200 x 8 canali per socket" ;;
            t2a-*)        B_BW_THEO="204"; B_BW_THEO_HOW="t2a (Ampere Altra): DDR4-3200 x 8 canali per socket" ;;
            n2-*|c2-*)    B_BW_THEO="205"; B_BW_THEO_HOW="n2/c2 (Cascade/Ice Lake): DDR4-3200 x ~8 canali per socket" ;;
            *)            B_BW_THEO_HOW="famiglia '$B_GCP_MT' senza configurazione di memoria pubblicata -> nessuna stima" ;;
        esac
        [ -n "$B_BW_THEO" ] && B_BW_THEO_SRC="famiglia-cloud"
    fi
    if [ -z "$B_BW_THEO" ] && [ "$B_OS" = "Darwin" ]; then
        case "$B_CPU_MODEL" in
            *"M1 Ultra"*) B_BW_THEO="800" ;; *"M1 Max"*) B_BW_THEO="400" ;; *"M1 Pro"*) B_BW_THEO="200" ;; *"M1"*) B_BW_THEO="68" ;;
            *"M2 Ultra"*) B_BW_THEO="800" ;; *"M2 Max"*) B_BW_THEO="400" ;; *"M2 Pro"*) B_BW_THEO="200" ;; *"M2"*) B_BW_THEO="100" ;;
            *"M3 Ultra"*) B_BW_THEO="800" ;; *"M3 Max"*) B_BW_THEO="400" ;; *"M3 Pro"*) B_BW_THEO="150" ;; *"M3"*) B_BW_THEO="100" ;;
            *"M4 Max"*)   B_BW_THEO="546" ;; *"M4 Pro"*) B_BW_THEO="273" ;; *"M4"*) B_BW_THEO="120" ;;
        esac
        [ -n "$B_BW_THEO" ] && { B_BW_THEO_SRC="tabella-soc"; B_BW_THEO_HOW="tabella dei SoC Apple ($B_CPU_MODEL): Apple non espone la banda via sysctl"; }
    fi
    [ -z "$B_BW_THEO_SRC" ] && B_BW_THEO_SRC="ignota"
    [ -z "$B_BW_THEO_HOW" ] && B_BW_THEO_HOW="nessuna fonte leggibile (ne' SMBIOS, ne' famiglia cloud nota)"
}

collect_membw() {
    [ -n "$MEMBW_BIN" ] && [ -x "$MEMBW_BIN" ] || {
        B_MEMBW_TXT="non misurata (passa --membw <binario di tests/membw.c>, oppure usa 'make server-hw-check' che lo compila)"
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
       "  %-8s %12s %12s" % ("thread", "Copy GB/s", "Triad GB/s")]
for r in d["sweep"]:
    out.append("  %-8d %12.1f %12.1f" % (r["threads"], r["copy_gbs"], r["triad_gbs"]))
out.append("  picco Triad %.1f GB/s a %d thread; GINOCCHIO a %d thread (95%% del picco)"
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
        degraded "piu' nodi NUMA ma numactl assente: local-vs-cross non misurabile (installa numactl: e' anche l'unico modo di pinnare)"
    fi
}

case "$B_OS" in
    Linux)  collect_linux ;;
    Darwin) collect_macos ;;
    *)      degraded "OS '$B_OS' non gestito: raccolta minima"; B_CPUS_LOG=$(getconf _NPROCESSORS_ONLN 2>/dev/null) ;;
esac
collect_bw_theoretical
collect_membw

WS_CP_06B_MB=60
WS_CP_17B_MB=120

REC_THREADS="$B_CORES_PHYS"
if [ "$B_OS" = "Darwin" ] && [ -n "$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null)" ]; then
    REC_THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null)   # solo i P-core
fi
[ -z "$REC_THREADS" ] && REC_THREADS="$B_CPUS_LOG"

GATE_SMT="PASS"; GATE_GOV="PASS"; GATE_CG="PASS"; GATE_ALL="PASS"
case "${B_SMT:-}" in on*) GATE_SMT="FAIL" ;; esac
case "${B_GOVERNOR:-}" in performance|n/a*|"non esposto") : ;; *) GATE_GOV="FAIL" ;; esac
if [ -n "${B_CG_CPU:-}" ] && [ "${B_CG_CPU%% *}" != "max" ] && [ "${B_CG_CPU%% *}" != "-1" ]; then GATE_CG="FAIL"; fi
[ "$GATE_SMT$GATE_GOV$GATE_CG" = "PASSPASSPASS" ] || GATE_ALL="FAIL"

case "${B_SMT:-}" in
    on*) warn "SMT ACCESO: $B_CPUS_LOG vCPU = ${B_CORES_PHYS:-?} core veri. Su GCP ricrea il box con --threads-per-core=1, oppure usa -j${REC_THREADS} e dichiaralo — altrimenti la misura descrive due hyperthread che si contendono una FMA." ;;
esac
case "${B_GOVERNOR:-}" in
    performance|n/a*|"non esposto") : ;;
    *) warn "governor = '$B_GOVERNOR' (non 'performance'): la frequenza scala sotto carico e i tempi ballano. sudo cpupower frequency-set -g performance, oppure dichiara che i numeri hanno quel rumore dentro." ;;
esac
if [ -n "${B_NUMA_NODES:-}" ] && [ "$B_NUMA_NODES" -gt 1 ] 2>/dev/null; then
    B_NUMA_RECO="$B_NUMA_NODES nodi: PINNA il processo a un nodo — numactl --cpunodebind=0 --membind=0 ./qwen_tts ... — altrimenti meta' degli accessi ai pesi attraversa l'interconnessione e la varianza mangia l'effetto che stai misurando."
    warn "NUMA a $B_NUMA_NODES nodi: senza pinning ogni cella della matrice ha dentro un rumore che non controlli."
elif [ -z "$B_NUMA_RECO" ]; then
    B_NUMA_RECO="1 nodo: nessun pinning necessario."
fi
if [ -n "${B_CG_CPU:-}" ] && [ "${B_CG_CPU%% *}" != "max" ] && [ "${B_CG_CPU%% *}" != "-1" ]; then
    warn "quota CPU cgroup attiva ('$B_CG_CPU'): stai misurando la QUOTA, non la macchina. Su un box dedicato non dovrebbe esserci."
fi
if [ -n "${B_SWAP_MB:-}" ] && [ "$B_SWAP_MB" -gt 0 ] 2>/dev/null; then
    warn "swap attiva (${B_SWAP_MB} MiB): se l'RSS del server sfiora la RAM, una cella lenta puo' essere paging e non il kernel sotto test."
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
        warn "banda misurata ${BW_PEAK} GB/s = ${_ratio}% della stima teorica ($B_BW_THEO GB/s): normale su una VM che vede una fetta del socket, ma da qui in avanti usa il MISURATO — la stima e' il tetto del server fisico, non della tua istanza."
fi

FIT_06B="ignoto"; FIT_17B="ignoto"
if [ -n "$B_LLC_MB" ]; then
    FIT_06B=$(awk -v l="$B_LLC_MB" -v w="$WS_CP_06B_MB" 'BEGIN{print (l>=w) ? "CI STA" : "NON ci sta"}')
    FIT_17B=$(awk -v l="$B_LLC_MB" -v w="$WS_CP_17B_MB" 'BEGIN{print (l>=w) ? "CI STA" : "NON ci sta"}')
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
                              note "$(jstr 'se uno di questi e FAIL, ogni numero di server/batching preso dopo descrive unaltra macchina')")" \
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
kv()  { printf '   %-22s %s\n' "$1" "${2:-n/d}"; }

hr
printf '  BOX INFO   %s   %s %s (%s)   %s\n' "${B_HOST:-?}" "$B_OS" "$B_KERNEL" "$B_ARCH" "$B_DATE"
hr

echo
printf '   GATE  SMT spento .............. %s   %s\n' "$GATE_SMT" \
    "$([ "$GATE_SMT" = PASS ] && echo 'vCPU = core: -j si legge direttamente' || echo "$B_CPUS_LOG vCPU = ${B_CORES_PHYS:-?} core -> ricrea con --threads-per-core=1")"
printf '   GATE  governor performance .... %s   %s\n' "$GATE_GOV" \
    "$([ "$GATE_GOV" = PASS ] && echo 'frequenza stabile sotto carico' || echo "e '$B_GOVERNOR': la frequenza scala e i tempi ballano")"
printf '   GATE  nessuna quota cgroup .... %s   %s\n' "$GATE_CG" \
    "$([ "$GATE_CG" = PASS ] && echo 'stai misurando la macchina' || echo "cpu.max='$B_CG_CPU': stai misurando la QUOTA")"
if [ "$GATE_ALL" = "PASS" ]; then
    echo "   → i tre invalidanti sono a posto: i numeri che seguono descrivono QUESTA macchina."
else
    echo "   → ⛔ almeno un invalidante e' rosso. Sistemalo PRIMA di misurare batching o server:"
    echo "        un numero preso adesso non e' sbagliato per poco, descrive un'altra macchina."
fi

if [ -n "$WARN" ]; then
    echo
    printf '%s\n' "$WARN" | while IFS= read -r l; do [ -n "$l" ] && printf '   ⚠️  %s\n' "$l"; done
else
    echo
    echo "   ✅ nessun avviso: SMT, governor, NUMA e quote sono nello stato che rende la misura leggibile."
fi

if [ -n "$B_GCP_MT" ]; then
    sec "GCP"
    kv "machine-type" "$B_GCP_MT"
    kv "zona"         "$B_GCP_ZONE"
    kv "preemptible"  "${B_GCP_PREEMPT:-false}  (spot = OK per capacita', NON per il soak)"
fi

sec "CPU"
kv "modello"        "$B_CPU_MODEL"
kv "vendor"         "$B_CPU_VENDOR"
kv "family/model"   "${B_CPU_FAMILY:-n/d}/${B_CPU_MODELID:-n/d}  stepping ${B_CPU_STEPPING:-n/d}"
kv "socket"         "$B_SOCKETS"
kv "core FISICI"    "${B_CORES_PHYS:-n/d}          <-- questo e' il numero da usare per -j"
kv "cpu logiche"    "${B_CPUS_LOG:-n/d}          (nproc; su GCP un vCPU e' un hyperthread)"
kv "thread/core"    "$B_TPC"
kv "SMT"            "$B_SMT"
kv "freq base/max"  "${B_FREQ_BASE:-n/d} / ${B_FREQ_MAX:-n/d} MHz"
if [ -n "$B_PERFLEVELS" ]; then
    printf '%s\n' "$B_PERFLEVELS" | while IFS= read -r l; do [ -n "$l" ] && printf '   %-22s %s\n' "" "$l"; done
fi

sec "ISA — le estensioni che decidono QUALE GEMM gira"
printf '   presenti:  %s\n' "$(printf '%s' "$B_FLAGS_HAVE" | tr '\n' ' ')"
printf '   assenti:   %s\n' "$(printf '%s' "$B_FLAGS_MISS" | tr '\n' ' ')"
echo   "   (nota: in /proc/cpuinfo VNNI e BF16 hanno l'underscore — avx512_vnni, avx512_bf16 —"
echo   "    mentre avx512f/bw/vl/dq no. I flag di gcc sono l'opposto. Non confonderli.)"
echo   "   → la conferma che il BINARIO li usa davvero la da' ./qwen_tts --caps, non questo elenco."

sec "Cache — il collo e' il Code Predictor che rilegge i pesi 16x per frame"
kv "L1d / L1i" "$(mb_h "$(sz_to_mb "$B_L1D")") / $(mb_h "$(sz_to_mb "$B_L1I")")"
kv "L2"        "$(mb_h "$(sz_to_mb "$B_L2")")"
[ -n "$B_L3" ] && kv "L3" "$(mb_h "$(sz_to_mb "$B_L3")")${B_L3_INSTANCES:+  x $B_L3_INSTANCES}"
kv "LLC utile"   "$(mb_h "$B_LLC_MB")  (${B_LLC_WHAT:-n/d})"
[ -n "$B_L3_PER_CORE" ] && kv "LLC per core" "$(mb_h "$B_L3_PER_CORE") per core fisico"
if [ -n "$B_L3_SHARED" ]; then
    printf '%s\n' "$B_L3_SHARED" | while IFS= read -r l; do [ -n "$l" ] && printf '   %-22s %s\n' "" "$l"; done
fi

sec "NUMA"
printf '%s' "${B_NUMA_DETAIL:-n/d}" | sed 's/^/   /'
echo
[ -n "$B_NUMA_DIST" ] && { printf '%s' "$B_NUMA_DIST" | sed 's/^/   /'; echo; }
printf '   → %s\n' "$B_NUMA_RECO"

sec "Memoria"
kv "totale"      "$(mb_h "$B_MEM_TOTAL_MB")"
kv "disponibile" "$(mb_h "$B_MEM_AVAIL_MB")"
kv "swap"        "$(mb_h "$B_SWAP_MB")"
kv "THP"         "$B_THP"
kv "hugepages"   "$B_HUGEPAGES"

sec "Banda di memoria — il carico e' memory-bound, quindi e' QUESTO il numero"
kv "teorica (STIMA)" "${B_BW_THEO:-n/d}${B_BW_THEO:+ GB/s}"
printf '   %-22s %s\n' "" "fonte: $B_BW_THEO_HOW"
printf '   %-22s %s\n' "" "⚠️ e' un TETTO del server fisico, non una misura della tua istanza."
if [ -n "$B_DIMMS" ]; then
    printf '%s\n' "$B_DIMMS" | while IFS= read -r l; do [ -n "$l" ] && printf '   %-22s %s\n' "" "$l"; done
fi
echo
if [ -n "$B_MEMBW_TXT" ]; then
    printf '%s' "$B_MEMBW_TXT" | sed 's/^/   /'
    echo
fi
if [ -n "$B_MEMBW_NUMA" ] && have python3; then
    echo "   NUMA-local contro cross-NUMA (la differenza E' il costo dell'interconnessione):"
    printf '%s\n' "$B_MEMBW_NUMA" | while IFS= read -r l; do
        [ -z "$l" ] && continue
        printf '%s' "$l" | python3 -c '
import json,sys
d = json.load(sys.stdin)
print("     %-12s picco Triad %6.1f GB/s a %d thread (ginocchio %d)" %
      (d["label"], d["peak_triad_gbs"], d["peak_triad_threads"], d["knee_threads"]))' 2>/dev/null
    done
elif [ "${B_NUMA_NODES:-1}" -le 1 ] 2>/dev/null; then
    echo "   NUMA-local vs cross-NUMA: non applicabile, c'e' un solo nodo."
fi

sec "Limiti (container / VM) e frequenza"
kv "cgroup cpu.max"    "${B_CG_CPU:-nessuno}"
kv "cgroup memory.max" "${B_CG_MEM:-nessuno}"
kv "governor"          "$B_GOVERNOR"
kv "scaling driver"    "$B_SCALING_DRIVER"

if [ -n "$DEGRADED" ]; then
    sec "Degradato (comandi/file assenti — campo mancante, non inventato)"
    printf '%s\n' "$DEGRADED" | while IFS= read -r l; do [ -n "$l" ] && printf '   · %s\n' "$l"; done
fi

sec "COSA SIGNIFICA PER IL BENCH"
case "${B_SMT:-}" in
    on*) ADV_SMT="ACCESO -> $B_CPUS_LOG vCPU sono ${B_CORES_PHYS:-?} core. Ricrea il box con --threads-per-core=1: e la differenza fra misurare la macchina e misurare l hyperthreading." ;;
    *)   ADV_SMT="spento/assente -> vCPU = core, i numeri di -j si leggono direttamente." ;;
esac
case "${B_GOVERNOR:-}" in
    performance|n/a*|"non esposto") ADV_FREQ="governor $B_GOVERNOR -> tempi stabili." ;;
    *) ADV_FREQ="governor $B_GOVERNOR -> NON e performance: cambialo (cpupower frequency-set -g performance) o dichiara il rumore." ;;
esac
ADV_CORE="core FISICI"
[ "$B_OS" = "Darwin" ] && ADV_CORE="core Performance (gli E-core sono ~3x piu lenti e diventano loro il tempo del frame)"

if [ -n "$BW_KNEE" ]; then
    ADV_BW="banda misurata: picco ${BW_PEAK} GB/s, GINOCCHIO a ${BW_KNEE} thread.
               Oltre ${BW_KNEE} thread la banda non sale: su un carico che rilegge i pesi
               16x per frame conviene dare ~${BW_KNEE} thread a PIU' richieste (B=2..8)
               piuttosto che tutti i core a una sola. E' la scelta -j x --batch-size."
else
    ADV_BW="banda NON misurata (passa --membw, o usa make server-hw-check): senza il
               ginocchio, la scelta fra -j alto e batch alto resta a tentoni."
fi

cat <<ADVICE
   1. thread   usa -j$REC_THREADS = i $ADV_CORE.
               Oltre, due thread si contendono la stessa FMA: il throughput per
               richiesta scende senza che Q salga, e sembra un difetto del batching.
   1.bis banda $ADV_BW
   2. SMT      $ADV_SMT
   3. NUMA     $B_NUMA_RECO
   4. cache    LLC $(mb_h "$B_LLC_MB") ($B_LLC_WHAT), $(mb_h "$B_L3_PER_CORE") per core.
               Il Code Predictor a int8 rilegge ~$WS_CP_06B_MB MB per frame sul 0.6B ($FIT_06B)
               e ~$WS_CP_17B_MB MB sul 1.7B ($FIT_17B). Dove NON ci sta, il limite e la banda:
               aspettati che il batching renda piu dell ISA, e che int4 paghi piu di AVX-512.
   5. freq     $ADV_FREQ
   6. ordine   questo report, poi ./qwen_tts --caps e --self-test (make server-hw-check), e SOLO
               dopo qualunque numero di server. Un --caps che non dichiara attiva la
               primitiva attesa significa che tutto cio che segue misura un fallback,
               non questa macchina. E un --self-test rosso invalida la matrice intera.
ADVICE
echo
hr
