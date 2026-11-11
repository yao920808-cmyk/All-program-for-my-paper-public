"""Descriptors extracted from the supplied to_juefei.py; legacy definitions preserved."""
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
FS_DEFAULT = 100.0

def safe_mean(x):
    return float(np.nanmean(x)) if len(x) else np.nan

def safe_std(x):
    return float(np.nanstd(x)) if len(x) else np.nan

def compute_td_basic(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {}
    feats = {}
    feats["mean"] = float(np.nanmean(x))
    feats["std"] = float(np.nanstd(x))
    feats["var"] = float(np.nanvar(x))
    feats["min"] = float(np.nanmin(x))
    feats["max"] = float(np.nanmax(x))
    feats["ptp"] = float(np.nanmax(x) - np.nanmin(x))
    feats["median"] = float(np.nanmedian(x))
    feats["iqr"] = float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))
    feats["rms"] = float(np.sqrt(np.nanmean(x**2)))
    feats["skew"] = float(skew(x)) if x.size>2 else 0.0
    feats["kurtosis"] = float(kurtosis(x)) if x.size>3 else 0.0
    feats["energy"] = float(np.nansum(x**2))
    feats["wfl"] = float(np.nansum(np.abs(np.diff(x))))
    return feats

def compute_prefix_freq_feats(x, fs=FS_DEFAULT):
    x = np.asarray(x, dtype=float)
    out = {}
    if len(x) < 4:
        out.update({"spec_total": np.nan, "spec_peak_freq": np.nan, "spec_centroid": np.nan, "spec_entropy": np.nan, "spec_median_freq": np.nan})
        return out
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    total_power = float(np.trapezoid(Pxx, f))
    out["spec_total"] = total_power
    out["spec_peak_freq"] = float(f[np.argmax(Pxx)]) if Pxx.size else np.nan
    out["spec_centroid"] = float(np.sum(f*Pxx)/np.sum(Pxx)) if total_power>0 else np.nan
    ps_norm = Pxx/np.sum(Pxx+1e-12)
    ps_norm = np.maximum(ps_norm, 1e-12)
    out["spec_entropy"] = -float(np.sum(ps_norm*np.log(ps_norm)))
    # median frequency
    cumsum = np.cumsum(Pxx)
    idx = np.searchsorted(cumsum, cumsum[-1]/2)
    out["spec_median_freq"] = float(f[idx]) if len(f)>0 else np.nan
    # band powers
    out["spec_low_power"] = float(np.sum(Pxx[f<3.5]))
    out["spec_high_power"] = float(np.sum(Pxx[f>=3.5]))
    out["spec_lfhf_ratio"] = (out["spec_low_power"]/(out["spec_high_power"]+1e-12))
    return out

def compute_impulse(x, t=None):
    x = np.asarray(x, dtype=float)
    if t is None: return float(np.trapezoid(x))
    t = np.asarray(t, dtype=float)
    return float(np.trapezoid(x, t))

def compute_loading_rate(x, t=None, pct=0.1):
    x = np.asarray(x, dtype=float)
    if t is None: t = np.arange(len(x))
    t = np.asarray(t, dtype=float)
    n = max(3,int(len(x)*pct))
    if len(x)<n+1: return np.nan
    try:
        coef = np.polyfit(t[:n], x[:n], 1)
        return float(coef[0])
    except: return np.nan

def sample_entropy(x, m=2, r=None):
    x = np.array(x)
    N = len(x)
    if N < (m+2):
        return np.nan
    if r is None: r=0.2*np.std(x)
    def _phi(m):
        count=0.0
        for i in range(N-m):
            template = x[i:i+m]
            # Chebyshev distance to next windows
            dist = np.max(np.abs(x[i+1:i+1+m]-template))
            count += np.sum(dist<=r)
        return count/(N-m)
    try: return -np.log(_phi(m+1)/_phi(m))
    except: return np.nan

def fractal_dimension_higuchi(x, kmax=10):
    x = np.asarray(x, dtype=float)
    N=len(x)
    if N < 5: return np.nan
    L=[]
    kmax = min(kmax, N//2)
    for k in range(1,kmax+1):
        Lk=0.0
        for m in range(k):
            idxs = np.arange(m,N,k)
            if len(idxs) < 2: continue
            lm = np.sum(np.abs(np.diff(x[idxs])))*(N-1)/(len(idxs)*k)
            Lk += lm
        L.append(Lk/k if k>0 else 0.0)
    lnL = np.log(np.maximum(L, 1e-12))
    lnk = np.log(1./np.arange(1,len(L)+1))
    try:
        fd = np.polyfit(lnk, lnL,1)[0]
    except:
        fd = np.nan
    return float(fd)

def compute_td_all_A(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {}
    feats = {}
    feats["mean"] = float(np.nanmean(x))
    feats["std"] = float(np.nanstd(x))
    feats["var"] = float(np.nanvar(x))
    feats["min"] = float(np.nanmin(x))
    feats["max"] = float(np.nanmax(x))
    feats["ptp"] = float(np.nanmax(x) - np.nanmin(x))
    feats["median"] = float(np.nanmedian(x))
    feats["iqr"] = float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))
    feats["rms"] = float(np.sqrt(np.nanmean(x**2)))
    feats["skew"] = float(skew(x)) if x.size>2 else 0.0
    feats["kurtosis"] = float(kurtosis(x)) if x.size>3 else 0.0
    feats["energy"] = float(np.nansum(x**2))
    feats["wfl"] = float(np.nansum(np.abs(np.diff(x))))
    feats["Frequency_domain"] = compute_prefix_freq_feats(x, fs=FS_DEFAULT)  ## Modified by Juefei Yuan
    feats["Sample_entropy"] = sample_entropy(x, m=2, r=None)     ## Modified by Juefei Yuan
    feats["Fractal_dimentaion_higuchi"] = fractal_dimension_higuchi(x, kmax=10)      ## Modified by Juefei Yuan
    feats["compute_impulse"] = compute_impulse(x)
    feats["compute_loading_rate"] = compute_loading_rate(x) 

    return feats