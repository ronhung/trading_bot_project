"""
Triple-Barrier Labeler for supervised learning in trading.

Given entry events and price paths, determines which barrier is hit first:
  - Upper barrier (take-profit)  → label = 1
  - Lower barrier (stop-loss)    → label = -1
  - Timeout (neither in horizon) → label = 0

Key rules:
  - First-touch wins: the barrier crossed at the earliest bar determines the label.
  - Same-bar tie → worst-case: if both barriers are touched within the same bar,
    label = -1 (stop-loss).  This is the conservative default.
  - Side-aware: long and short positions have inverted barrier semantics.

Vectorized implementation with chunked processing for memory efficiency.
Includes a pure-Python reference implementation for correctness validation.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def apply_triple_barrier(
    close: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    events: Union[np.ndarray, list],
    upper_barrier: float = 0.02,
    lower_barrier: float = -0.01,
    horizon: int = 288,
    side: Union[int, np.ndarray] = 1,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """
    Vectorized triple-barrier labeler.

    For each event at bar position `i`, scans forward up to `horizon` bars
    to determine which barrier is touched first.

    Parameters
    ----------
    close, high, low : 1-D arrays
        Price series in chronological order.
    events : array-like of int
        Bar positions (0-based indices) of entry events.
    upper_barrier : float, default 0.02
        Take-profit return.  Long: tp_price = entry * (1 + upper_barrier).
        Short: tp_price = entry / (1 + upper_barrier).
        Must be > 0.
    lower_barrier : float, default -0.01
        Stop-loss return.  Must be < 0.
    horizon : int, default 288
        Maximum holding period in bars.
    side : int or 1-D array, default 1
        Trade direction.  1 = long, -1 = short.
        If scalar, applied to all events.  If array, one value per event.
    chunk_size : int, default 5000
        Events processed per vectorized batch (bounds memory).

    Returns
    -------
    pd.DataFrame indexed by event position with columns:
      label         : 1 (TP), -1 (SL), 0 (timeout)
      barrier_hit   : 'upper' | 'lower' | 'timeout'
      exit_idx      : integer bar position of exit
      n_bars_held   : number of bars in the trade
      entry_price   : price at entry
      exit_price    : price at exit
      actual_return : trade-side log return
      truncated     : True if event is too close to data end for full horizon
    """
    # --- coerce inputs ---
    close_arr = np.asarray(close, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    ev = np.asarray(events, dtype=int)
    side_arr = np.full(len(ev), side, dtype=int) if np.ndim(side) == 0 else np.asarray(side, dtype=int)

    if len(ev) == 0:
        return pd.DataFrame(columns=[
            "label", "barrier_hit", "exit_idx", "n_bars_held",
            "entry_price", "exit_price", "actual_return", "truncated",
        ])

    if upper_barrier <= 0:
        raise ValueError(f"upper_barrier must be > 0, got {upper_barrier}")
    if lower_barrier >= 0:
        raise ValueError(f"lower_barrier must be < 0, got {lower_barrier}")

    n = len(close_arr)

    # --- pre-compute entry prices ---
    entry_prices = close_arr[ev]

    # --- compute barrier prices (side-aware) ---
    # Long:  tp = entry * (1 + ub),  sl = entry * (1 + lb)
    # Short: tp = entry / (1 + ub),  sl = entry / (1 + lb)
    is_long = side_arr == 1
    tp_price = np.where(
        is_long,
        entry_prices * (1.0 + upper_barrier),
        entry_prices / (1.0 + upper_barrier),
    )
    sl_price = np.where(
        is_long,
        entry_prices * (1.0 + lower_barrier),
        entry_prices / (1.0 + lower_barrier),
    )

    # --- pad arrays so out-of-range lookups never "hit" ---
    # high padded with -inf  → never >= tp_price
    # low  padded with +inf  → never <= sl_price
    high_pad = np.concatenate([high_arr, np.full(horizon, -np.inf)])
    low_pad = np.concatenate([low_arr, np.full(horizon, np.inf)])
    close_pad = np.concatenate([close_arr, np.full(horizon, np.nan)])

    # --- process events in chunks ---
    results = []
    offsets = np.arange(1, horizon + 1, dtype=int)  # shape (horizon,)

    for chunk_start in range(0, len(ev), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(ev))
        m = chunk_end - chunk_start

        ev_chunk = ev[chunk_start:chunk_end]
        tp_chunk = tp_price[chunk_start:chunk_end]
        sl_chunk = sl_price[chunk_start:chunk_end]
        side_chunk = side_arr[chunk_start:chunk_end]
        entry_chunk = entry_prices[chunk_start:chunk_end]

        # Build forward-window index matrix: shape (m, horizon)
        rows = ev_chunk[:, None] + offsets[None, :]  # (m, horizon)

        # Gather forward high/low
        win_high = high_pad[rows]  # (m, horizon)
        win_low = low_pad[rows]    # (m, horizon)

        # Which bars touch which barrier (SIDE-AWARE)
        # Long:  TP hit when price rises  → high >= tp_price
        #         SL hit when price falls  → low  <= sl_price
        # Short: TP hit when price falls  → low  <= tp_price
        #         SL hit when price rises → high >= sl_price
        is_long_chunk = side_chunk == 1
        is_short_chunk = ~is_long_chunk

        # Compute both directional masks
        up_hit_long  = win_high >= tp_chunk[:, None]   # long:  high touches TP above
        dn_hit_long  = win_low  <= sl_chunk[:, None]   # long:  low  touches SL below
        up_hit_short = win_low  <= tp_chunk[:, None]   # short: low  touches TP below
        dn_hit_short = win_high >= sl_chunk[:, None]   # short: high touches SL above

        # Select based on side
        up_hit = np.where(is_long_chunk[:, None], up_hit_long, up_hit_short)
        dn_hit = np.where(is_long_chunk[:, None], dn_hit_long, dn_hit_short)

        # First bar index (1-based offset) where each barrier is hit
        up_first_raw = np.where(up_hit.any(axis=1), up_hit.argmax(axis=1) + 1, 0)
        dn_first_raw = np.where(dn_hit.any(axis=1), dn_hit.argmax(axis=1) + 1, 0)

        up_never = up_first_raw == 0
        dn_never = dn_first_raw == 0

        # Decision logic:
        # - upper wins if: hit AND (dn never OR up_first < dn_first)
        # - lower wins if: hit AND (up never OR dn_first <= up_first)
        #   (≤ implements same-bar → worst-case -1)
        up_wins = ~up_never & (dn_never | (up_first_raw < dn_first_raw))
        dn_wins = ~dn_never & (up_never | (dn_first_raw <= up_first_raw))

        # Resolve conflicts: same-bar tie where both are hit
        # up_wins and dn_wins can both be True for same-bar hit → dn_wins takes priority
        up_wins = up_wins & ~dn_wins

        # Exit offset and label
        exit_offset = np.full(m, horizon, dtype=int)
        label = np.zeros(m, dtype=int)
        barrier_hit = np.full(m, "timeout", dtype=object)

        exit_offset[up_wins] = up_first_raw[up_wins]
        label[up_wins] = 1
        barrier_hit[up_wins] = "upper"

        exit_offset[dn_wins] = dn_first_raw[dn_wins]
        label[dn_wins] = -1
        barrier_hit[dn_wins] = "lower"

        # Handle events near end of data: cap at available bars
        max_avail = n - 1 - ev_chunk
        truncated = exit_offset > max_avail
        exit_offset = np.minimum(exit_offset, np.maximum(max_avail, 0))

        # If event is the last bar (max_avail == 0), exit at same bar
        exit_idx = ev_chunk + exit_offset
        exit_idx = np.minimum(exit_idx, n - 1)  # safety clamp

        exit_prices = close_arr[exit_idx]
        n_bars_held = exit_offset.copy()

        # For truncated events with no bars left, reset
        no_room = max_avail <= 0
        exit_idx[no_room] = ev_chunk[no_room]
        exit_prices[no_room] = entry_chunk[no_room]
        n_bars_held[no_room] = 0
        label[no_room] = 0
        barrier_hit[no_room] = "timeout"

        # Compute trade-side log returns
        # Long:  log(exit / entry)
        # Short: log(entry / exit)
        is_long_chunk = side_chunk == 1
        actual_return = np.where(
            is_long_chunk,
            np.log(exit_prices / entry_chunk),
            np.log(entry_chunk / exit_prices),
        )

        chunk_df = pd.DataFrame({
            "label": label,
            "barrier_hit": barrier_hit,
            "exit_idx": exit_idx,
            "n_bars_held": n_bars_held,
            "entry_price": entry_chunk,
            "exit_price": exit_prices,
            "actual_return": actual_return,
            "truncated": truncated,
        }, index=ev_chunk)
        chunk_df.index.name = "event_idx"

        results.append(chunk_df)

    out = pd.concat(results)
    return out


def apply_triple_barrier_loop(
    close: Union[np.ndarray, pd.Series],
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    events: Union[np.ndarray, list],
    upper_barrier: float = 0.02,
    lower_barrier: float = -0.01,
    horizon: int = 288,
    side: Union[int, np.ndarray] = 1,
) -> pd.DataFrame:
    """
    Reference implementation: per-event Python loop.
    Exists only for cross-validation of the vectorized version.
    Not optimized; use apply_triple_barrier for production.
    """
    close_arr = np.asarray(close, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    ev = np.asarray(events, dtype=int)
    side_arr = np.full(len(ev), side, dtype=int) if np.ndim(side) == 0 else np.asarray(side, dtype=int)

    n = len(close_arr)
    rows = []

    for k, (i, s) in enumerate(zip(ev, side_arr)):
        entry = close_arr[i]

        if s == 1:  # long
            tp = entry * (1.0 + upper_barrier)
            sl = entry * (1.0 + lower_barrier)
        else:  # short
            tp = entry / (1.0 + upper_barrier)
            sl = entry / (1.0 + lower_barrier)

        end = min(i + horizon + 1, n)
        label = 0
        barrier = "timeout"
        exit_i = min(i + horizon, n - 1)
        exit_px = close_arr[exit_i]
        truncated = (i + horizon) >= n

        for j in range(i + 1, end):
            h = high_arr[j]
            l = low_arr[j]
            # Side-aware barrier touches
            if s == 1:  # long
                up_touched = h >= tp
                dn_touched = l <= sl
            else:  # short
                up_touched = l <= tp  # price falls to TP
                dn_touched = h >= sl  # price rises to SL

            if up_touched and dn_touched:
                # Same bar → worst case
                label = -1
                barrier = "lower"
                exit_i = j
                exit_px = close_arr[j]
                break
            elif up_touched:
                label = 1
                barrier = "upper"
                exit_i = j
                exit_px = close_arr[j]
                break
            elif dn_touched:
                label = -1
                barrier = "lower"
                exit_i = j
                exit_px = close_arr[j]
                break

        if s == 1:
            ret = np.log(exit_px / entry)
        else:
            ret = np.log(entry / exit_px)

        rows.append({
            "label": label,
            "barrier_hit": barrier,
            "exit_idx": exit_i,
            "n_bars_held": exit_i - i,
            "entry_price": entry,
            "exit_price": exit_px,
            "actual_return": ret,
            "truncated": truncated and label == 0,
        })

    out = pd.DataFrame(rows, index=ev)
    out.index.name = "event_idx"
    return out


# ============================================================
# __main__ self-test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing research.labeling -- Triple Barrier Labeler")
    print("=" * 60)

    # Build a tiny hand-crafted price path
    # Bar:  0     1     2     3     4     5     6     7
    close = np.array([100, 101, 102, 103, 104, 103, 101, 99], dtype=float)
    high  = np.array([101, 102, 103, 104, 105, 104, 102, 100], dtype=float)
    low   = np.array([ 99, 100, 101, 100, 103, 102, 100,  98], dtype=float)

    def _run(desc, events, ub, lb, h, side=1):
        r_vec = apply_triple_barrier(close, high, low, events, ub, lb, h, side)
        r_loop = apply_triple_barrier_loop(close, high, low, events, ub, lb, h, side)
        # compare
        cols = ["label", "barrier_hit", "exit_idx", "n_bars_held"]
        ok = r_vec[cols].equals(r_loop[cols])
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {desc}")
        print(f"   events={list(events)}, ub={ub}, lb={lb}, horizon={h}, side={side}")
        print(f"   vectorized labels: {r_vec['label'].tolist()}")
        print(f"   loop labels:      {r_loop['label'].tolist()}")
        if not ok:
            print("   MISMATCH!")
            print("   vec:\n", r_vec[cols])
            print("   loop:\n", r_loop[cols])
        return ok

    all_ok = True

    # Test 1: Long — upper touched first at bar 3 (high=104 >= 102*1.02=104.04? no, 102*1.02=104.04)
    # Let's use entry at bar 1, price 101, tp=101*1.03=104.03, sl=101*0.98=98.98
    # high[3]=104 >= 104.03? barely no. Let me be more careful.
    # entry at bar 1: price=101. tp=101*1.04=105.04. high[4]=105 >= 105.04? no.
    # Let me just use bigger barriers.
    # entry at bar 1, price=101, ub=0.05 → tp=106.05, lb=-0.03 → sl=97.97
    # high[4]=105, high[5]=104... not enough.
    # entry at bar 0, price=100, ub=0.04 → tp=104, lb=-0.02 → sl=98
    # high[3]=104 >= 104 → upper hit at bar 3. low[1]=100 > 98.
    all_ok &= _run("Long -- upper hit first", [0], 0.04, -0.02, 6, side=1)

    # Test 2: Long — lower touched first
    # entry at bar 4, price=104, ub=0.05 → tp=109.2, lb=-0.02 → sl=101.92
    # bar 5: low=102 < 101.92? no. high=104 < 109.2.
    # bar 6: low=100 < 101.92 → lower hit!
    # bar 7: low=98...
    # So lower hit first at bar 6 → label -1
    all_ok &= _run("Long -- lower hit first", [4], 0.05, -0.02, 6, side=1)

    # Test 3: Same-bar both hit → worst case (-1)
    # entry at bar 3, price=103, ub=0.01 → tp=104.03, lb=-0.01 → sl=101.97
    # bar 4: high=105 >= 104.03 AND low=103 > 101.97 → upper only
    # Let me construct a case where both happen same bar.
    # Custom data for this test:
    c2 = np.array([100, 105, 99], dtype=float)   # entry at 0, price=100
    h2 = np.array([101, 110, 100], dtype=float)
    l2 = np.array([ 99,  90,  98], dtype=float)
    # entry=100, ub=0.05→tp=105, lb=-0.05→sl=95
    # bar 1: high=110>=105 (upper) AND low=90<=95 (lower) → same bar → label -1
    r = apply_triple_barrier(c2, h2, l2, [0], 0.05, -0.05, 5, side=1)
    rl = apply_triple_barrier_loop(c2, h2, l2, [0], 0.05, -0.05, 5, side=1)
    ok = r["label"].iloc[0] == -1 and rl["label"].iloc[0] == -1
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] Same-bar dual touch -> worst case (-1)")
    print(f"   label={r['label'].iloc[0]}, barrier={r['barrier_hit'].iloc[0]}")
    all_ok &= ok

    # Test 4: Timeout
    # entry at bar 1, price=101, ub=0.50→tp=151.5, lb=-0.50→sl=50.5
    # horizon=3, bars 2-4 won't touch → timeout
    all_ok &= _run("Timeout -- no barrier hit", [1], 0.50, -0.50, 3, side=1)

    # Test 5: Event at last bar (truncated)
    r = apply_triple_barrier(close, high, low, [7], 0.02, -0.01, 5, side=1)
    rl = apply_triple_barrier_loop(close, high, low, [7], 0.02, -0.01, 5, side=1)
    ok = (r["truncated"].iloc[0] == True) and (rl["truncated"].iloc[0] == True)
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] Event at last bar -> truncated")
    print(f"   label={r['label'].iloc[0]}, truncated={r['truncated'].iloc[0]}")
    all_ok &= ok

    # Test 6: Short side — price falls to TP
    # entry at bar 4, price=104, side=-1 (short)
    # ub=0.05 → tp=104/1.05=99.05, lb=-0.02 → sl=104/0.98=106.12
    # bar 5: low=102 > 99.05, high=104 < 106.12 → no hit
    # bar 6: low=100 > 99.05, high=102 < 106.12 → no hit
    # bar 7: low=98 < 99.05 → upper (TP for short) hit! → label 1
    all_ok &= _run("Short -- take-profit (price falls to TP)", [4], 0.05, -0.02, 6, side=-1)

    # Test 7: Random cross-validation (2000 events)
    print("\n--- Random cross-validation (2000 events) ---")
    np.random.seed(42)
    n_random = 5000
    c_rand = np.cumsum(np.random.randn(n_random) * 0.01) + 100.0
    c_rand = np.maximum(c_rand, 1.0)  # prices must stay positive
    h_rand = c_rand * (1.0 + np.random.uniform(0, 0.008, n_random))
    l_rand = c_rand * (1.0 - np.random.uniform(0, 0.008, n_random))
    ev_rand = np.random.choice(np.arange(100, n_random - 200), size=2000, replace=False)
    ev_rand.sort()
    side_rand = np.random.choice([1, -1], size=2000)

    rv = apply_triple_barrier(c_rand, h_rand, l_rand, ev_rand, 0.02, -0.01, 100, side_rand)
    rl = apply_triple_barrier_loop(c_rand, h_rand, l_rand, ev_rand, 0.02, -0.01, 100, side_rand)

    cols = ["label", "barrier_hit", "exit_idx", "n_bars_held"]
    ok_random = rv[cols].equals(rl[cols])
    status = "PASS" if ok_random else "FAIL"
    print(f"[{status}] Random 2000-event cross-validation")
    if not ok_random:
        mismatch = rv[cols] != rl[cols]
        n_mismatch = mismatch.any(axis=1).sum()
        print(f"   {n_mismatch} mismatches found!")
        bad = mismatch.any(axis=1)
        print("   First 5 mismatches:")
        for idx in rv.index[bad][:5]:
            print(f"   event={idx}: vec=({rv.loc[idx, 'label']},{rv.loc[idx, 'barrier_hit']}) "
                  f"loop=({rl.loc[idx, 'label']},{rl.loc[idx, 'barrier_hit']})")
    all_ok &= ok_random

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - review output above.")
    print("=" * 60)
