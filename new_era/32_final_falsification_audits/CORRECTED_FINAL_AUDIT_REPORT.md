# Corrected Final Falsification Audit Report

## 1. Correction Scope

Phase 033 corrects the portions of Phase 032 that were too easy to over-interpret. The exact matched random quote test is saturated and non-informative because `bid_only_spread_ge_1` selects every action in the exact candidate universe `side=bid and spread>=1`.

This report therefore separates four concepts: exact-universe random saturation, broader regime-vs-random baselines, rule ablations, and stricter timing/proxy-delay stress.

## 2. Matched Random Saturation

The exact matched random baseline must not be used as an alpha claim. It is now documented in `matched_random_saturation_report.md` as a saturation diagnostic only.

## 3. Broader Regime-vs-Random Baselines

Broader random baselines restore discriminative power by sampling from all valid bid/ask actions while matching counts by session or session/spread bucket. These are diagnostic controls for side/regime selection, not replacements for true queue replay.

- `unused_NEW3_TXFA6` vs `random_all_valid_match_session_spread_bucket`: baseline PnL `7,367,885`, random 95p `493,720`, gap `6,874,166`, p-value `0.0000`.
- `phase031_new4` vs `random_all_valid_match_session_spread_bucket`: baseline PnL `6,348,770`, random 95p `476,878`, gap `5,871,892`, p-value `0.0000`.
- `unused_NEW4_TXFA6` vs `random_all_valid_match_session_spread_bucket`: baseline PnL `6,348,770`, random 95p `476,878`, gap `5,871,892`, p-value `0.0000`.
- `unused_NEW1_TMFB6` vs `random_all_valid_match_session_spread_bucket`: baseline PnL `5,665,815`, random 95p `860,693`, gap `4,805,122`, p-value `0.0000`.
- `unused_NEW_TMFB6` vs `random_all_valid_match_session_spread_bucket`: baseline PnL `4,874,985`, random 95p `303,751`, gap `4,571,234`, p-value `0.0000`.
- `unused_NEW1_MXFB6` vs `random_all_valid_match_session_spread_bucket`: baseline PnL `4,174,225`, random 95p `359,987`, gap `3,814,238`, p-value `0.0000`.
- `unused_NEW_MXFB6` vs `random_all_valid_match_session_spread_bucket`: baseline PnL `3,992,335`, random 95p `387,002`, gap `3,605,334`, p-value `0.0000`.
- `unused_NEW1_TXFB6` vs `random_all_valid_match_session_spread_bucket`: baseline PnL `2,788,070`, random 95p `141,880`, gap `2,646,190`, p-value `0.0000`.
- `phase031_test` vs `random_all_valid_match_session_spread_bucket`: baseline PnL `2,312,590`, random 95p `-219,375`, gap `2,531,965`, p-value `0.0000`.
- `unused_NEW_TXFB6` vs `random_all_valid_match_session_spread_bucket`: baseline PnL `2,624,355`, random 95p `120,042`, gap `2,504,313`, p-value `0.0000`.

## 4. Rule Ablation

The ablation confirms the core rule is effectively `bid_all` in the current quote-eligible universe, because all candidate spreads are at least one tick. The edge should therefore be described as a bid-side regime effect under the proxy, not as proof that the spread threshold alone creates alpha.

- `core` aggregate across reported groups: `46,497,800` proxy PnL.
- `ask_all` aggregate across reported groups: `-50,540,870` proxy PnL.
- `all_actions` aggregate across reported groups: `-4,043,070` proxy PnL.

## 5. Timing and Delay Rebuild

The lagged-book test is now labeled as a timing-artifact stress: it changes decision features but does not recalculate delayed execution value. The new execution-delay proxy does recalculate value from the delayed entry row.

- Strict C `phase031_new4`: delayed-entry proxy PnL `61,505`, selected `713`, missing delayed entries `708`.
- Strict C `phase031_test`: delayed-entry proxy PnL `0`, selected `419`, missing delayed entries `419`.
- Strict C `unused_NEW1_MXFB6`: delayed-entry proxy PnL `0`, selected `680`, missing delayed entries `680`.
- Strict C `unused_NEW1_TMFB6`: delayed-entry proxy PnL `0`, selected `873`, missing delayed entries `873`.
- Strict C `unused_NEW1_TXFB6`: delayed-entry proxy PnL `0`, selected `422`, missing delayed entries `422`.
- Strict C `unused_NEW3_TXFA6`: delayed-entry proxy PnL `0`, selected `1397`, missing delayed entries `1397`.
- Strict C `unused_NEW4_TXFA6`: delayed-entry proxy PnL `61,505`, selected `713`, missing delayed entries `708`.
- Strict C `unused_NEW_MXFB6`: delayed-entry proxy PnL `0`, selected `680`, missing delayed entries `680`.
- Strict C `unused_NEW_TMFB6`: delayed-entry proxy PnL `0`, selected `873`, missing delayed entries `873`.
- Strict C `unused_NEW_TXFB6`: delayed-entry proxy PnL `0`, selected `422`, missing delayed entries `422`.

- Delay 10 `phase031_new4`: proxy PnL `71,570`, gap vs delay0 `-6,277,200`, effective delayed entries `6` of `714`.
- Delay 10 `phase031_test`: proxy PnL `0`, gap vs delay0 `-2,312,590`, effective delayed entries `2` of `419`.
- Delay 10 `unused_NEW1_MXFB6`: proxy PnL `58,940`, gap vs delay0 `-4,115,285`, effective delayed entries `5` of `682`.
- Delay 10 `unused_NEW1_TMFB6`: proxy PnL `14,120`, gap vs delay0 `-5,651,695`, effective delayed entries `5` of `873`.
- Delay 10 `unused_NEW1_TXFB6`: proxy PnL `14,285`, gap vs delay0 `-2,773,785`, effective delayed entries `2` of `422`.
- Delay 10 `unused_NEW3_TXFA6`: proxy PnL `0`, gap vs delay0 `-7,367,885`, effective delayed entries `0` of `1399`.
- Delay 10 `unused_NEW4_TXFA6`: proxy PnL `71,570`, gap vs delay0 `-6,277,200`, effective delayed entries `6` of `714`.
- Delay 10 `unused_NEW_MXFB6`: proxy PnL `29,320`, gap vs delay0 `-3,963,015`, effective delayed entries `5` of `682`.
- Delay 10 `unused_NEW_TMFB6`: proxy PnL `40,225`, gap vs delay0 `-4,834,760`, effective delayed entries `5` of `873`.
- Delay 10 `unused_NEW_TXFB6`: proxy PnL `14,285`, gap vs delay0 `-2,610,070`, effective delayed entries `2` of `422`.

Delay caveat: the available action universe is a sparse set of quote-candidate rows, not a full raw-tick EV table. When `local_row_id + delay` is not present in the action universe, the script records a missing delayed entry and contributes zero proxy PnL. This is a conservative proxy-execution stress, not a true queue replay.

## 6. NEW/NEW1/NEW3/NEW4 Dedup and Independence

- `NEW` vs `NEW1` classified `likely_duplicate` with `6` identical price-sequence hashes out of `6` common symbol/session pairs.

Do not sum likely-duplicate OOS PnL as independent evidence.

## 7. All Unused OOS Coverage

The corrected falsification scripts include every rebuilt unused-OOS group available from NEW, NEW1, NEW3, and NEW4, in addition to the Phase031 frozen test/new4 rows. NEW and NEW1 are reported separately and interpreted through the dedup audit.

## 8. Corrected Claim Boundary

Allowed claims: HLS-friendly comparator logic and proxy OOS robustness with duplicate-data caveats.

Forbidden claims: real-world profitability, FPGA PnL necessity, true queue replay, and beating exact-matched random selection.

Conservative conclusion: `bid_only_spread_ge_1` remains a simple, auditable, FPGA-friendly proxy controller candidate, but the corrected audit narrows the claim to proxy robustness and side-regime behavior. The exact matched-random result is saturated and cannot support a quote-selection skill claim.
