#!/usr/bin/env python3
"""LOKI circular-orbit Extreme Pruning (EP) search driver.

Runs inside the vishnubk/loki:latest container on an already-dedispersed
single-DM time series (.dat + .inf, or .tim).

Fixed campaign recipe (validated end-to-end on injected reference data):
    circular-orbit basis (ParamLimits.from_circular), nbins=64, eta=2.0,
    bseg_ffa=2**18, prune_poly_order=5, p_orb_min=3600 s (do NOT raise:
    inj_000004 has P_orb=3981 s), m_c_max=4.0, m_p_min=1.2, min_pd=0.05.

Example
-------
docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
    -v /data:/data -v $PWD:/work vishnubk/loki:latest \
    python /work/ep_search.py \
        --tim /data/inj_000004_DM50.00.dat \
        --outdir /work/results/inj4_fourier \
        --backend fourier --f-target 200.2965759 --device 0 --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np

# ----------------------------------------------------------------------------
# Fixed campaign constants (do not change without re-validating)
# ----------------------------------------------------------------------------
NBINS = 64
ETA = 2.0
BSEG_FFA = 2**18
PRUNE_POLY_ORDER = 5
P_ORB_MIN = 3600.0  # seconds -- do NOT raise (inj_000004 P_orb = 3981 s)
M_C_MAX = 4.0
M_P_MIN = 1.2
MIN_PD = 0.05
NSAMPS_PAD = 2**24  # pipeline pads the 30-min series to 2^24
EXPECTED_TSAMP = 120.470592e-6

LOG = logging.getLogger("ep_search")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOKI circular-orbit EP search on a dedispersed time series.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- required interface (as specified) ---
    p.add_argument("--tim", required=True, help="dedispersed .dat (+ .inf) or .tim")
    p.add_argument("--outdir", required=True, help="results directory")
    p.add_argument(
        "--backend",
        required=True,
        choices=["fourier", "time"],
        help="EPMultiPassCUDA variant: Fourier-domain or time-domain folds",
    )
    p.add_argument(
        "--f-target",
        required=True,
        type=float,
        help="apparent (mid-observation) spin frequency in Hz; the shard MUST contain it",
    )
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument(
        "--seed", type=int, default=42, help="pinned seed for the dynamic threshold scheme"
    )
    # --- knobs with campaign defaults (overridable, all logged) ---
    p.add_argument(
        "--shard-halfwidth",
        type=float,
        default=1.0,
        help="requested frequency shard is f_target +/- this (Hz) before the "
        "Doppler widening that ParamLimits.from_circular applies",
    )
    p.add_argument("--n-runs", type=int, default=16, help="number of EP reference segments")
    p.add_argument("--max-sugg", type=int, default=2**20, help="max leaves held per run")
    p.add_argument("--batch-size", type=int, default=1024, help="EP branching batch size")
    p.add_argument("--nthreads", type=int, default=8, help="CPU threads (FFA stage)")
    p.add_argument("--ducy-max", type=float, default=0.3, help="max duty cycle searched")
    p.add_argument(
        "--ref-ducy",
        type=float,
        default=0.25,
        help="reference duty cycle for the threshold scheme",
    )
    p.add_argument("--wtsp", type=float, default=1.0, help="width-trial spacing factor")
    p.add_argument(
        "--thresh-ntrials", type=int, default=1024, help="MC trials per threshold stage"
    )
    p.add_argument(
        "--thresh-snr-final", type=float, default=8.0, help="target final S/N of the scheme"
    )
    p.add_argument(
        "--thresholds-cache",
        default=None,
        help="npz path: loaded if it exists (and matches nstages/seed), else written",
    )
    p.add_argument("--file-prefix", default=None, help="output file prefix (default: auto)")
    p.add_argument("--n-top", type=int, default=20, help="number of top candidates to report")
    p.add_argument(
        "--nsamps-pad", type=int, default=NSAMPS_PAD, help="pad/truncate the series to this"
    )
    p.add_argument(
        "--tim-type",
        default=None,
        choices=["dat", "tim"],
        help="override the reader chosen from the file extension",
    )
    return p.parse_args(argv)


def setup_logging(outdir: str, tag: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    logpath = os.path.join(outdir, f"ep_search_{tag}.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    fh = logging.FileHandler(logpath)
    fh.setFormatter(fmt)
    LOG.addHandler(fh)
    return logpath


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------
def log_provenance(args: argparse.Namespace) -> dict:
    import loki
    import pyloki

    commit = getattr(loki, "__commit__", "<unstamped>")
    prov = {
        "loki_commit": commit,
        "loki_version": getattr(loki, "__version__", "?"),
        "loki_file": loki.__file__,
        "pyloki_file": pyloki.__file__,
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "hostname": os.uname().nodename,
    }
    try:  # GPU identity is worth having in the log; failure is non-fatal
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={args.device}",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        prov["gpu"] = out.stdout.strip() or out.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        prov["gpu"] = f"<nvidia-smi failed: {exc}>"

    LOG.info("=" * 78)
    LOG.info("LOKI circular-orbit EP search")
    LOG.info("loki.__commit__ = %s", commit)
    for k, v in prov.items():
        if k != "loki_commit":
            LOG.info("  %-14s %s", k, v)
    LOG.info("=" * 78)
    return prov


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_and_pad(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Read the time series and zero-pad to args.nsamps_pad.

    pyloki's TimeSeries.from_tim() deredden+normalises and returns
    ts_v = ones_like(ts_e).  The pipeline convention for padding (as used by
    the validated repro scripts) is: ts_e = 0 and ts_v = 0 in the pad region,
    so the folder sees zero variance/weight there.
    """
    from pyloki.io.timeseries import TimeSeries

    tim_type = args.tim_type
    if tim_type is None:
        tim_type = "tim" if args.tim.lower().endswith(".tim") else "dat"
    if not os.path.exists(args.tim):
        raise FileNotFoundError(args.tim)
    if tim_type == "dat":
        inf = os.path.splitext(args.tim)[0] + ".inf"
        if not os.path.exists(inf):
            raise FileNotFoundError(f"PRESTO .dat needs its header: {inf}")

    LOG.info("Reading time series %s (tim_type=%s)", args.tim, tim_type)
    ts = TimeSeries.from_tim(args.tim, tim_type=tim_type)
    n_real, tsamp = ts.nsamps, float(ts.dt)
    LOG.info("  nsamps(real) = %d, tsamp = %.9e s, tobs = %.3f s", n_real, tsamp, ts.tobs)
    if abs(tsamp - EXPECTED_TSAMP) > 1e-12:
        LOG.warning(
            "  tsamp %.9e differs from the campaign value %.9e", tsamp, EXPECTED_TSAMP
        )

    npad = int(args.nsamps_pad)
    if n_real > npad:
        raise ValueError(f"series has {n_real} samples > pad target {npad}")
    ts_e = np.zeros(npad, dtype=np.float32)
    ts_v = np.zeros(npad, dtype=np.float32)
    ts_e[:n_real] = ts.ts_e
    ts_v[:n_real] = ts.ts_v
    LOG.info(
        "  padded to %d samples (%.2f%% real data, %.3f s of zero padding)",
        npad,
        100.0 * n_real / npad,
        (npad - n_real) * tsamp,
    )
    return ts_e, ts_v, tsamp, n_real


# ----------------------------------------------------------------------------
# Config / shard
# ----------------------------------------------------------------------------
def build_config(args: argparse.Namespace, tsamp: float):
    from loki.libloki.configs import PulsarSearchConfig
    from pyloki.config import ParamLimits

    f_lo = args.f_target - args.shard_halfwidth
    f_hi = args.f_target + args.shard_halfwidth
    if not (f_lo < args.f_target < f_hi):
        raise ValueError("requested shard does not bracket f_target")
    if f_lo <= 0:
        raise ValueError(f"shard lower bound must be positive, got {f_lo}")

    # from_circular sets the crackle..accel derivative bounds from the orbit
    # limits and additionally widens the freq bounds by the max Doppler shift.
    lims = ParamLimits.from_circular(
        freq=(f_lo, f_hi),
        p_orb_min=P_ORB_MIN,
        m_c_max=M_C_MAX,
        m_p_min=M_P_MIN,
        poly_order=PRUNE_POLY_ORDER,
    )
    cfg = PulsarSearchConfig(
        nsamps=int(args.nsamps_pad),
        tsamp=tsamp,
        nbins=NBINS,
        eta=ETA,
        param_limits=lims.limits,
        bseg_ffa=BSEG_FFA,
        prune_poly_order=PRUNE_POLY_ORDER,
        p_orb_min=P_ORB_MIN,
        m_c_max=M_C_MAX,
        m_p_min=M_P_MIN,
        ducy_max=args.ducy_max,
        wtsp=args.wtsp,
        nthreads=args.nthreads,
        # use_fourier only affects the standalone FFA-score helper; the EP path
        # is templated on its own fold type (lib/prune.cpp:179 FFAPlan<FoldType>).
        # Kept True for both backends, matching both validated reference runs.
        use_fourier=True,
    )

    LOG.info("--- search configuration ---")
    LOG.info("  basis            circular (ParamLimits.from_circular, poly_order=%d)",
             PRUNE_POLY_ORDER)
    LOG.info("  nbins=%d eta=%.3g bseg_ffa=%d (2^%d) bseg_brute=%d",
             NBINS, ETA, BSEG_FFA, int(math.log2(BSEG_FFA)), cfg.bseg_brute)
    LOG.info("  p_orb_min=%.1f s  m_c_max=%.2f  m_p_min=%.2f  min_pd=%.3f",
             P_ORB_MIN, M_C_MAX, M_P_MIN, MIN_PD)
    LOG.info("  ducy_max=%.3g wtsp=%.3g nthreads=%d nsamps=%d tsamp=%.9e",
             args.ducy_max, args.wtsp, args.nthreads, cfg.nsamps, cfg.tsamp)
    LOG.info("  param_names      %s", list(cfg.param_names))
    for name, (lo, hi) in zip(list(cfg.param_names), np.asarray(cfg.param_limits)):
        LOG.info("    %-9s [%+.6e, %+.6e]", name, lo, hi)
    # cfg.dparams / dparams_f are methods taking the current segment length
    LOG.info("  dparams(tseg_ffa)   %s", np.asarray(cfg.dparams(cfg.tseg_ffa)).tolist())
    LOG.info("  dparams(tseg_brute) %s", np.asarray(cfg.dparams(cfg.tseg_brute)).tolist())
    LOG.info("  tobs=%.3f s tseg_ffa=%.3f s nsegments=%d",
             cfg.tobs, cfg.tseg_ffa, int(cfg.nsamps // BSEG_FFA))

    # --- shard assertion (hard requirement) ---
    f_min, f_max = float(cfg.f_min), float(cfg.f_max)
    LOG.info("--- frequency shard ---")
    LOG.info("  requested          [%.9f, %.9f] Hz (halfwidth %.4f Hz)",
             f_lo, f_hi, args.shard_halfwidth)
    LOG.info("  effective (cfg)    [%.9f, %.9f] Hz  (Doppler-widened)", f_min, f_max)
    LOG.info("  f_target           %.9f Hz", args.f_target)
    LOG.info("  margin to edges    lo %+.6f Hz, hi %+.6f Hz",
             args.f_target - f_min, f_max - args.f_target)
    if not (f_min < args.f_target < f_max):
        raise AssertionError(
            f"shard [{f_min}, {f_max}] Hz does not contain f_target={args.f_target} Hz"
        )
    LOG.info("  ASSERT OK: shard contains f_target")
    shard = {
        "requested_f_min": f_lo,
        "requested_f_max": f_hi,
        "effective_f_min": f_min,
        "effective_f_max": f_max,
    }
    return cfg, shard


# ----------------------------------------------------------------------------
# Thresholds
# ----------------------------------------------------------------------------
def get_thresholds(args: argparse.Namespace, cfg) -> np.ndarray:
    from loki.libculoki.thresholds import DynamicThresholdSchemeCUDA
    from loki.libloki.plans import FFAPlanBase

    plan = FFAPlanBase(cfg)
    # "taylor" is correct for the circular-orbit search: circular candidates are
    # classified inside the taylor branching (lib/core/circular.hpp); "circular"
    # is NOT a valid poly_basis (only taylor|chebyshev, lib/plans.cpp:195-258).
    bp = np.asarray(plan.get_branching_pattern("taylor", ref_seg=0), dtype=np.float32)
    LOG.info("--- threshold scheme ---")
    LOG.info("  branching pattern: nstages=%d, first 8 = %s",
             bp.size, np.round(bp[:8], 4).tolist())

    cache = args.thresholds_cache
    if cache and os.path.exists(cache):
        z = np.load(cache)
        thr = np.asarray(z["thresholds"], dtype=np.float32)
        ok = int(z.get("seed", -1)) == args.seed and thr.size == bp.size
        LOG.info("  cache %s: seed=%s nstages=%d -> %s",
                 cache, z.get("seed", "?"), thr.size, "USING" if ok else "REJECTED")
        if ok:
            LOG.info("  thresholds (cached): %s", np.round(thr, 4).tolist())
            return thr

    LOG.info(
        "  building DynamicThresholdSchemeCUDA(ref_ducy=%.3g, nbins=%d, ntrials=%d, "
        "snr_final=%.3g, ducy_max=%.3g, wtsp=%.3g, seed=%d, device_id=%d)",
        args.ref_ducy, NBINS, args.thresh_ntrials, args.thresh_snr_final,
        args.ducy_max, args.wtsp, args.seed, args.device,
    )
    t0 = time.time()
    scheme = DynamicThresholdSchemeCUDA(
        branching_pattern=bp,
        ref_ducy=args.ref_ducy,
        nbins=NBINS,
        ntrials=args.thresh_ntrials,
        snr_final=args.thresh_snr_final,
        ducy_max=args.ducy_max,
        wtsp=args.wtsp,
        seed=args.seed,
        device_id=args.device,
    )
    scheme.run()
    thr = np.asarray(scheme.get_best_path_thresholds(min_pd=MIN_PD), dtype=np.float32)
    LOG.info("  scheme built in %.1f s; nthresholds=%d", time.time() - t0, thr.size)
    LOG.info("  thresholds: %s", np.round(thr, 4).tolist())
    try:
        saved = scheme.save(args.outdir)
        LOG.info("  scheme HDF5: %s", saved)
    except Exception as exc:  # noqa: BLE001  (non-fatal, but surfaced)
        LOG.warning("  scheme.save failed: %s", exc)
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)) or ".", exist_ok=True)
        np.savez(cache, thresholds=thr, seed=args.seed, branching_pattern=bp)
        LOG.info("  thresholds cached -> %s", cache)
    return thr


# ----------------------------------------------------------------------------
# EP run
# ----------------------------------------------------------------------------
def run_ep(args, cfg, thr, ts_e, ts_v, prefix) -> float:
    from loki.libculoki.prune import EPMultiPassFourierCUDA, EPMultiPassTimeCUDA

    ep_cls = EPMultiPassFourierCUDA if args.backend == "fourier" else EPMultiPassTimeCUDA
    LOG.info("--- EP multi-pass search ---")
    LOG.info("  class=%s n_runs=%d max_sugg=%d batch_size=%d poly_basis=taylor device_id=%d",
             ep_cls.__name__, args.n_runs, args.max_sugg, args.batch_size, args.device)
    ep = ep_cls(
        cfg,
        thr,
        n_runs=args.n_runs,
        max_sugg=args.max_sugg,
        batch_size=args.batch_size,
        poly_basis="taylor",
        device_id=args.device,
    )
    t0 = time.time()
    ep.execute(ts_e, ts_v, args.outdir, prefix)
    runtime = time.time() - t0
    LOG.info("  EP execute() returned after %.1f s", runtime)
    return runtime


# ----------------------------------------------------------------------------
# Read back
# ----------------------------------------------------------------------------
def _dec(x):
    return x.decode() if isinstance(x, bytes) else x


def find_result_file(outdir: str, prefix: str) -> str:
    """Result file is '<prefix>_pruning_nstages_<nsegments>_results.h5'.

    (lib/prune_cuda.cu:932-938).  Globbed rather than hardcoded; 'tmp_*' files
    are per-run intermediates of the CPU multi-threaded path and are skipped.
    """
    pat = os.path.join(outdir, f"{prefix}_pruning_nstages_*_results.h5")
    hits = [f for f in sorted(glob.glob(pat)) if not os.path.basename(f).startswith("tmp_")]
    if not hits:
        raise FileNotFoundError(f"no results HDF5 matching {pat}")
    if len(hits) > 1:
        LOG.warning("multiple result files matched, using the last: %s", hits)
    return hits[-1]


def read_results(path: str, n_top: int, f_target: float) -> dict:
    """Parse the pruning results HDF5.

    Layout verified in lib/cands.cpp:
      root attrs : pruning_version, param_names, nsegments, max_sugg, final_runtime
      root ds    : threshold_scheme
      runs/<ref_seg:03d>_<task_id:02d>/
          attrs    : total_pruning_gflops, termination_status
          datasets : param_sets (n_leaves, nparams+2, 2), scores (n_leaves,),
                     scores_ep (n_leaves,), snail_scheme, level_stats, timer_stats

    param_sets column layout (lib/taylor.cpp:1252-1269, and the mid-observation
    gauge transform in poly_taylor_report_batch, lib/taylor.cpp:1547-1581):
      cols 0..nparams-1 -> param_names = [crackle, snap, jerk, accel, freq],
          [:, i, 0] = value, [:, i, 1] = uncertainty/step.
          The freq column is f0 * (1 - v/c): the APPARENT frequency at the
          MID-OBSERVATION epoch, not the intrinsic spin frequency.
      col nparams     -> [d0 (phase/delay offset), 0]
      col nparams+1   -> [f0 (reference freq of the leaf), basis_flag(0=poly,1=physical)]
    """
    import h5py

    out = {"result_file": path, "runs": {}}
    with h5py.File(path, "r") as h:
        root_attrs = {k: _dec(v) for k, v in h.attrs.items()}
        param_names = [_dec(x) for x in np.atleast_1d(h.attrs["param_names"])]
        out["param_names"] = param_names
        out["nsegments"] = int(np.squeeze(h.attrs["nsegments"]))
        out["max_sugg"] = int(np.squeeze(h.attrs["max_sugg"]))
        out["final_runtime_s"] = float(np.squeeze(h.attrs.get("final_runtime", np.nan)))
        out["threshold_scheme"] = np.asarray(h["threshold_scheme"][()]).tolist()
        LOG.info("--- results readback: %s ---", path)
        LOG.info("  root attrs: %s", root_attrs)

        if "runs" not in h:
            raise RuntimeError(
                "results file has no 'runs' group. With this loki build every run "
                "writes a group (carrying termination_status) even when empty, so a "
                "missing group means the EP run did not reach report_survivors or an "
                "older binary produced this file."
            )

        rows = []  # top-N candidates only, kept small
        statuses = {}
        n_survivors = 0
        best_near = None  # (|df|, row) closest to f_target across all runs
        for run_name, grp in h["runs"].items():
            status = _dec(grp.attrs.get("termination_status", "<MISSING>"))
            n_leaves = int(grp["scores"].shape[0])
            gflops = float(np.squeeze(grp.attrs.get("total_pruning_gflops", np.nan)))
            statuses[run_name] = status
            out["runs"][run_name] = {
                "termination_status": status,
                "n_leaves": n_leaves,
                "total_pruning_gflops": gflops,
            }
            LOG.info("  run %-8s termination_status=%-24s n_leaves=%-8d gflops=%.3g",
                     run_name, status, n_leaves, gflops)
            n_survivors += n_leaves
            if n_leaves == 0:
                continue
            nparams = grp["param_sets"].shape[1] - 2
            if nparams != len(param_names):
                raise RuntimeError(
                    f"param_sets has {nparams} params, attrs list {len(param_names)}"
                )
            sc = np.asarray(grp["scores"][()], dtype=np.float64)
            sc_ep = np.asarray(grp["scores_ep"][()], dtype=np.float64)

            def make_rows(idx, grp=grp, sc=sc, sc_ep=sc_ep, run_name=run_name,
                          nparams=nparams):
                """Materialise dicts for a handful of leaf indices only."""
                idx = np.sort(np.asarray(idx, dtype=np.int64))
                sub = np.asarray(grp["param_sets"][idx, :, :], dtype=np.float64)
                built = []
                for k, i in enumerate(idx):
                    leaf = {
                        "run_id": run_name,
                        "score": float(sc[i]),
                        "score_ep": float(sc_ep[i]),
                    }
                    for j, nm in enumerate(param_names):
                        leaf[nm] = float(sub[k, j, 0])
                        leaf[f"d{nm}"] = float(sub[k, j, 1])
                    leaf["_d0_offset"] = float(sub[k, nparams, 0])
                    leaf["_f0_ref"] = float(sub[k, nparams + 1, 0])
                    leaf["_basis_flag"] = float(sub[k, nparams + 1, 1])
                    built.append(leaf)
                return built

            # top-n_top of this run by 'score' (argpartition: O(n), no full sort)
            k = min(n_top, n_leaves)
            top_idx = np.argpartition(-sc, k - 1)[:k]
            rows.extend(make_rows(top_idx))

            # nearest-in-frequency leaf of this run (read only the freq column)
            freqs = np.asarray(
                grp["param_sets"][:, len(param_names) - 1, 0], dtype=np.float64
            )
            j_near = int(np.argmin(np.abs(freqs - f_target)))
            cand = make_rows([j_near])[0]
            d = abs(cand["freq"] - f_target)
            if best_near is None or d < best_near[0]:
                best_near = (d, cand)

        out["termination_statuses"] = statuses
        out["n_survivors"] = n_survivors
        all_rows = rows

        # A single overall status: worst-case wins, so anomalies cannot hide.
        order = ["extinct_early_anomalous", "extinct_late", "completed"]
        seen = set(statuses.values())
        overall = next((s for s in order if s in seen), "<UNKNOWN>")
        out["termination_status"] = overall
        LOG.info("  n_survivors (all runs) = %d", out["n_survivors"])
        LOG.info("  overall termination_status = %s (per-run: %s)", overall, seen)
        if overall == "extinct_early_anomalous":
            LOG.error(
                "ANOMALOUS EARLY EXTINCTION reported by loki - empty/short result "
                "must be treated with suspicion (threshold/config problem)."
            )
        unknown = seen - set(order)
        if unknown:
            LOG.error("UNEXPECTED termination_status value(s): %s", unknown)

    if not all_rows:
        LOG.warning("no surviving leaves in any run - no candidates to report")
        out["top"] = []
        return out

    all_rows.sort(key=lambda r: r["score"], reverse=True)
    # Drop bit-exact duplicate leaves (they do occur) so the top list is informative.
    # n_survivors above is the raw count and is left untouched.
    seen_keys, deduped = set(), []
    for r in all_rows:
        key = (r["score"], *(r[nm] for nm in param_names))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(r)
    if len(deduped) < len(all_rows):
        LOG.info("  dropped %d bit-exact duplicate leaves from the top list",
                 len(all_rows) - len(deduped))
    all_rows = deduped
    out["top"] = all_rows[: n_top]

    LOG.info("--- TOP %d candidates (by 'score'; freq is APPARENT, at "
             "MID-OBSERVATION after loki's gauge transform) ---", n_top)
    for rank, r in enumerate(all_rows[:n_top], 1):
        pstr = "  ".join(f"{nm}={r[nm]:+.6e}" for nm in out["param_names"] if nm != "freq")
        LOG.info(
            "  #%-3d run=%s score=%.4f score_ep=%.4f freq_mid=%.9f +/- %.3e Hz "
            "(df_target=%+.6f Hz)  %s",
            rank, r["run_id"], r["score"], r["score_ep"], r["freq"], r["dfreq"],
            r["freq"] - f_target, pstr,
        )
        LOG.info("      leaf extras: d0_offset=%+.6e f0_ref=%.9f basis_flag=%g",
                 r["_d0_offset"], r["_f0_ref"], r["_basis_flag"])
        porb = derived_p_orb(r)
        if porb:
            LOG.info("      DERIVED (heuristic, NOT from the API): %s", porb)

    best = all_rows[0]
    if best_near is not None:
        near = best_near[1]
        LOG.info(
            "Closest-to-f_target survivor (any score): run=%s freq_mid=%.9f Hz "
            "(%+.6f Hz) score=%.4f",
            near["run_id"], near["freq"], near["freq"] - f_target, near["score"],
        )
        out["closest_to_target"] = near
    LOG.info("Top leaf freq offset from f_target: %+.9f Hz (%.3g ppm)",
             best["freq"] - f_target, 1e6 * (best["freq"] - f_target) / f_target)
    return out


def derived_p_orb(leaf: dict) -> dict:
    """Best-effort P_orb from the Taylor derivatives.

    NOT provided by the loki API: no orbital elements (P_orb, x, omega) are
    written to the results file - only mid-observation Taylor derivatives.
    For a circular orbit each derivative chain oscillates at omega_orb, so
    omega^2 ~ -snap/accel and ~ -crackle/jerk.  Purely diagnostic.
    """
    out = {}
    for hi, lo, key in (("snap", "accel", "p_orb_from_snap_accel"),
                        ("crackle", "jerk", "p_orb_from_crackle_jerk")):
        if hi in leaf and lo in leaf and leaf[lo] != 0.0:
            ratio = -leaf[hi] / leaf[lo]
            if ratio > 0:
                out[key] = 2.0 * math.pi / math.sqrt(ratio)
    return out


# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prefix = args.file_prefix or f"ep_{args.backend}_f{args.f_target:.6f}_s{args.seed}"
    logpath = setup_logging(args.outdir, f"{args.backend}_s{args.seed}")

    prov = log_provenance(args)
    LOG.info("outdir=%s  file_prefix=%s  logfile=%s", args.outdir, prefix, logpath)
    LOG.info("backend=%s  device=%d  seed=%d  f_target=%.9f Hz",
             args.backend, args.device, args.seed, args.f_target)
    LOG.info("full args: %s", json.dumps(vars(args), sort_keys=True))

    ts_e, ts_v, tsamp, n_real = load_and_pad(args)
    cfg, shard = build_config(args, tsamp)
    thr = get_thresholds(args, cfg)

    t_wall = time.time()
    runtime = run_ep(args, cfg, thr, ts_e, ts_v, prefix)

    res_path = find_result_file(args.outdir, prefix)
    res = read_results(res_path, args.n_top, args.f_target)

    # pyloki's own summary, purely informational. It builds one pandas row per
    # surviving leaf, so skip it when the survivor population is large.
    PGRAM_MAX_LEAVES = 200_000
    try:
        if res["n_survivors"] > PGRAM_MAX_LEAVES:
            raise RuntimeError(
                f"skipped: {res['n_survivors']} survivors > {PGRAM_MAX_LEAVES} "
                "(ScatteredPeriodogram would materialise them all in pandas)"
            )
        from pyloki.periodogram import ScatteredPeriodogram

        pgram = ScatteredPeriodogram.load(res_path)
        LOG.info("--- pyloki ScatteredPeriodogram summary ---\n%s",
                 pgram.get_summary_cands(args.n_top, score_type="score", run_id=None))
        LOG.info("--- best candidate per run ---\n%s", pgram.get_best_in_each_run())
    except Exception as exc:  # noqa: BLE001
        LOG.warning("ScatteredPeriodogram summary unavailable: %s", exc)

    top = res["top"][0] if res["top"] else None
    summary = {
        "backend": args.backend,
        "f_target": args.f_target,
        "shard": shard,
        "top_leaf_freq": (top["freq"] if top else None),
        "top_leaf_freq_note": "apparent freq at mid-observation (loki gauge transform)",
        "top_leaf_score": (top["score"] if top else None),
        "termination_status": res["termination_status"],
        "runtime_s": round(runtime, 3),
        "n_survivors": res["n_survivors"],
        "leaf_params": (
            {k: v for k, v in top.items() if k not in ("run_id",)} if top else None
        ),
        # extras
        "leaf_params_derived": derived_p_orb(top) if top else None,
        "top_leaf_run_id": (top["run_id"] if top else None),
        "top_leaf_score_ep": (top["score_ep"] if top else None),
        "param_names": res["param_names"],
        "per_run_termination_status": res["termination_statuses"],
        "result_file": res_path,
        "log_file": logpath,
        "loki_commit": prov["loki_commit"],
        "seed": args.seed,
        "device": args.device,
        "wall_s": round(time.time() - t_wall, 3),
        "nsamps_real": n_real,
        "nsamps_padded": int(args.nsamps_pad),
    }
    print("EP_SUMMARY_JSON " + json.dumps(summary, default=float), flush=True)
    LOG.info("done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - surface, never swallow
        traceback.print_exc()
        try:
            LOG.error("FAILED:\n%s", traceback.format_exc())
        except Exception:  # noqa: BLE001, S110
            pass
        print(
            "EP_SUMMARY_JSON "
            + json.dumps({"status": "failed", "error": traceback.format_exc(limit=3)}),
            flush=True,
        )
        sys.exit(1)
