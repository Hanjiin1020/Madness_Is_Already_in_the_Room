#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import sqrt

from scipy.stats import chi2

def _expected_a(n1, n2, m1, psi, tol=1e-10, it=200):
    lo, hi = max(0.0, m1 - n2), min(float(n1), float(m1))
    lo, hi = lo + tol, hi - tol
    if hi <= lo:
        return None

    def g(A):
        return A * (n2 - m1 + A) - psi * (n1 - A) * (m1 - A)

    if g(lo) > 0 or g(hi) < 0:
        return None
    for _ in range(it):
        mid = (lo + hi) / 2
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2

def breslow_day(tabs, or_mh):

    stat, used = 0.0, 0
    for a, b, c, d in tabs:
        n1, n2, m1 = a + b, c + d, a + c
        Ae = _expected_a(n1, n2, m1, or_mh)
        if Ae is None:
            continue
        Be, Ce, De = n1 - Ae, m1 - Ae, n2 - m1 + Ae
        if min(Ae, Be, Ce, De) <= 1e-8:
            continue
        var = 1.0 / (1.0 / Ae + 1.0 / Be + 1.0 / Ce + 1.0 / De)
        stat += (a - Ae) ** 2 / var
        used += 1
    df = used - 1
    if df < 1:
        return float("nan"), 0, float("nan")
    return stat, df, float(chi2.sf(stat, df))

def standardize(rate_by_stratum, weight_by_stratum):
    w, r, tot = weight_by_stratum, rate_by_stratum, 0.0
    num = 0.0
    for s, wi in w.items():
        ri = r.get(s)
        if ri is None or ri != ri:
            continue
        num += wi * ri
        tot += wi
    return num / tot if tot > 0 else float("nan")

def mh_or(tabs):
    tabs = [t for t in tabs if sum(t) > 1]
    num = sum(a * d / sum((a, b, c, d)) for a, b, c, d in tabs)
    den = sum(b * c / sum((a, b, c, d)) for a, b, c, d in tabs)
    if den == 0 or num == 0:
        return float("nan"), float("nan"), float("nan")
    o = num / den
    P = [(a + d) / sum((a, b, c, d)) for a, b, c, d in tabs]
    Q = [(b + c) / sum((a, b, c, d)) for a, b, c, d in tabs]
    R = [a * d / sum((a, b, c, d)) for a, b, c, d in tabs]
    S = [b * c / sum((a, b, c, d)) for a, b, c, d in tabs]
    sR, sS = sum(R), sum(S)
    v = (sum(p * r for p, r in zip(P, R)) / (2 * sR ** 2)
         + sum(p * s + q * r for p, q, r, s in zip(P, Q, R, S)) / (2 * sR * sS)
         + sum(q * s for q, s in zip(Q, S)) / (2 * sS ** 2))
    se = sqrt(v)
    from math import exp, log
    return o, exp(log(o) - 1.96 * se), exp(log(o) + 1.96 * se)
