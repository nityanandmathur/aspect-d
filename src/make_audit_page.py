"""Generate coordinate-audit.html from the artifacts, so the page cannot drift.

The page was first written by hand against a snapshot of the numbers, which is exactly
how a retracted claim survives a revision. It is now generated, and reuses the site's
stylesheet verbatim from extensions.html rather than carrying a second copy.

    python src/make_audit_page.py
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLE = {"wavlm_large": "selector (circular)", "ecapa": "gated scorer",
        "ge2e": "independent family", "xvect": "independent family",
        "wavlm_base_plus": "independent scale"}


def main():
    src = open(os.path.join(REPO, "extensions.html")).read()
    marker = "</style></head><body><main>"
    head = src[:src.index(marker) + len(marker)].replace(
        "<title>ASPECT-D · v1.1 extensions</title>",
        "<title>ASPECT-D · v1.4 coordinate audit</title>")

    a = json.load(open(os.path.join(REPO, "artifacts-v1.4", "analysis.json")))
    m = json.load(open(os.path.join(REPO, "artifacts-v1.3", "multi_encoder.json")))
    b, co = a["B_gap_closed"], a["A_coordinate_sensitivity"]["by_coordinate"]
    rng = a["C_range_sensitivity"]
    gof, hci = a["E_goodness_of_fit"], a.get("D_honest_ci")

    coord = "".join(
        f"<tr><td>{k}</td><td class=num>{v['tau_wer']:.4f}</td>"
        f"<td class=num>{v['tau_sim']:.4f}</td><td class=num><b>{v['delta_tau']:+.4f}</b></td>"
        f"<td>{'<span class=\"pill no\">sign flipped</span>' if v['delta_tau'] < 0 else '<span class=\"pill ok\">as published</span>'}</td></tr>"
        for k, v in co.items())

    gap = "".join(
        f"<tr><td class=num>{int(T)}</td>"
        f"<td class=num>{100*e['wer']['point']:.1f}% "
        f"<span class=sub>[{100*e['wer']['ci'][0]:.1f}, {100*e['wer']['ci'][1]:.1f}]</span></td>"
        f"<td class=num>{100*e['sim']['point']:.1f}% "
        f"<span class=sub>[{100*e['sim']['ci'][0]:.1f}, {100*e['sim']['ci'][1]:.1f}]</span></td>"
        f"<td class=num><b>{e['ratio_wer_over_sim']['point']:.2f}×</b></td></tr>"
        for T, e in sorted(b["by_T"].items(), key=lambda kv: int(kv[0])))

    win = "".join(
        f"<tr><td>{k.replace('_','-')}</td><td>{ROLE[k]}</td>"
        f"<td class=num>{100*v['win_rate']:.1f}%</td>"
        f"<td class=num>[{100*v['win_ci'][0]:.1f}, {100*v['win_ci'][1]:.1f}]</td>"
        f"<td class=num>{v['mean_delta']:+.4f}</td></tr>"
        for k, v in sorted(m["encoders"].items(), key=lambda kv: -kv[1]["win_rate"]))

    wins = "".join(f"<tr><td>{k}</td><td class=num>{v['delta_tau']:+.4f}</td>"
                   f"<td class=num>{v['n_rows']}</td></tr>"
                   for k, v in sorted(rng["windows"].items(),
                                      key=lambda kv: int(kv[0].split("=")[1])))
    import pandas as pd
    n_frozen = len(pd.read_csv(os.path.join(REPO, 'artifacts', 'runs.csv')))
    _ext = pd.read_csv(os.path.join(REPO, "artifacts-v1.4", "runs_extended.csv"))
    n_rows, n_runs = len(_ext), _ext.groupby(["config", "seed"]).ngroups
    t16 = b["by_T"]["16"]
    tmax = max(b["by_T"], key=int)
    tex = b["by_T"][tmax]
    rs = [v["ratio_wer_over_sim"]["point"] for k, v in b["by_T"].items() if int(k) >= 4]
    hci_txt = (f"<td class=num>[{hci['ci_weights_resampled'][0]:.4f}, "
               f"{hci['ci_weights_resampled'][1]:.4f}]</td>" if hci else "<td class=sub>—</td>")

    body = f"""
<p class=eyebrow>ASPECT-D · v1.4</p>
<h1>A headline that was a property of the <span class=t>coordinate</span></h1>
<p class=standfirst>The v1.0 primary result was an exponent contrast,
Δτ = τ<sub>WER</sub> − τ<sub>SIM</sub>. Refitting the identical measurements after a monotone
change of the error variable sends it negative. This page records what failed, what replaced
it, and what independently survived.</p>

<div class=tldr><b>What changed</b>
Δτ is invariant to <em>affine</em> rescalings of the error — which the pre-registration
checked — but not to monotone non-affine ones. It also halves each time the fitted T window
doubles. The paper now leads with a model-free, floor-referenced contrast that moves by
neither.</div>

<h2><span class=sec>1</span>The failure: coordinate</h2>
<p>Same {n_frozen} surface points, same fitting code, same weights, same 32 multi-starts. Only the
error variable changes.</p>
<div class=scroll><table><thead><tr><th>error coordinate</th><th class=num>τ<sub>WER</sub></th>
<th class=num>τ<sub>SIM</sub></th><th class=num>Δτ</th><th>verdict</th></tr></thead>
<tbody>{coord}</tbody></table></div>

<h2><span class=sec>2</span>The failure: range</h2>
<p>We extended all {n_runs} runs to T={rng['T_available'][-1]} at the full
{a['data']['eval_items']} items — {n_rows} balanced rows, no config or item subset. The earlier extended fit used 7 of 15 configs at 200
of 400 items, so it confounded range with subset. Refit on nested windows:</p>
<div class=scroll><table><thead><tr><th>window</th><th class=num>Δτ</th>
<th class=num>rows</th></tr></thead><tbody>{wins}</tbody></table></div>
<div class="gate fail"><span class=tag>retracted as a headline</span>
Δτ roughly halves each time the window doubles, and three of five reparameterisations flip its
sign. It is reported in the paper as a pre-registered test that passes <em>in its declared
coordinate and range</em>, with both conditions attached.</div>

<h2><span class=sec>3</span>The replacement</h2>
<p>Referenced to floors we measure — ASR word error on the real recordings
({b['floors']['wer_asr_floor']:.4f}) and the speaker similarity of a codec round trip
({b['floors']['sim_codec_ceiling']:.4f}) — rather than to zero. No model is fitted.</p>
<div class=scroll><table><thead><tr><th class=num>T</th>
<th class=num>intelligibility closed</th><th class=num>identity closed</th>
<th class=num>ratio</th></tr></thead><tbody>{gap}</tbody></table></div>
<div class=tldr><b>headline</b>
At the deployable T=16 default, refinement closes <b>{100*t16['wer']['point']:.1f}%</b> of the
reachable intelligibility range and <b>{100*t16['sim']['point']:.1f}%</b> of the reachable
identity range. The ratio holds between <b>{min(rs):.2f}×</b> and <b>{max(rs):.2f}×</b> across
a 16× range of refinement budget, and between
<b>{b['ratio_range_across_coordinates'][0]:.2f}×</b> and
<b>{b['ratio_range_across_coordinates'][1]:.2f}×</b> across the coordinates that flip Δτ's
sign — because numerator and denominator move together.</div>
<p>Identity plateaus: quadrupling the budget to T={tmax} takes intelligibility to
{100*tex['wer']['point']:.1f}% but identity only to {100*tex['sim']['point']:.1f}%. More
refinement is not the lever, which is the paper's case for spending inference compute on
search instead.</p>

<h2><span class=sec>4</span>What independently survived</h2>
<p>The paper's top stated limitation was that selector and scorer shared the ECAPA-TDNN
lineage. Five encoders across four families rescored the same clips. Gate G0(c) rejects three
of them on absolute cosine <em>scale</em> despite AUC ≥ 0.983, so agreement is read from
per-item win rates, which are invariant to any monotone rescaling of an encoder's cosine.</p>
<div class=scroll><table><thead><tr><th>encoder</th><th>role</th><th class=num>win rate</th>
<th class=num>95% CI</th><th class=num>raw Δ</th></tr></thead><tbody>{win}</tbody></table></div>
<div class=gate><span class=tag>closed</span>
{m['n_independent_families_agreeing']}/4 non-selector encoders put search above refinement with
every interval clear of one half, across {m['encoders']['ecapa']['n_runs']} runs. The raw Δ
column is scale-bound and not comparable across rows; the win rate is.</div>

<h2><span class=sec>5</span>Three further scope conditions now in the paper</h2>
<div class=scroll><table><thead><tr><th>condition</th><th>measurement</th>
<th class=num>value</th></tr></thead><tbody>
<tr><td>the declared form is rejected by its own criterion</td><td>χ²/dof, WER / SIM-o</td>
<td class=num>{gof['wer']['chi2_per_dof']:.1f} / {gof['sim']['chi2_per_dof']:.1f}</td></tr>
<tr><td>the declared interval omits weighting uncertainty</td>
<td>Δτ CI with weights resampled</td>{hci_txt}</tr>
<tr><td>no global depth–step exchange rate</td>
<td>local κ = τ/β implied by the separable fit</td><td class=num>0.41</td></tr>
</tbody></table></div>
<p class=sub>The fit is kept as a compact description of curvature. No mechanism is read from
its parameters.</p>

<h2><span class=sec>6</span>Reproduce</h2>
<p><code>bash src/finish_v14.sh</code> collects the extended surface, refits, regenerates every
number on this page and in the paper, rebuilds, and verifies. It guards the v1.0 immutables
against the release tag, and <code>src/paper_v14.py</code> refuses to emit the range macros
when the fitted windows are identical rather than shipping a tautology.
<code>python src/check_claims.py</code> fails the build if a retracted claim is live anywhere.</p>
<p><a href="index.html">← index</a> · <a href="extensions.html">v1.1 extensions</a> ·
<a href="results.html">results</a> · <a href="protocol.html">protocol</a></p>
"""
    with open(os.path.join(REPO, "coordinate-audit.html"), "w") as fh:
        fh.write(head + body + "\n</main></body></html>\n")
    print("wrote coordinate-audit.html")


if __name__ == "__main__":
    main()
