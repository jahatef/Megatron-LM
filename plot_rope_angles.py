"""
Debug plots + correctness checks for 2D RoPE from the RotaryEmbeddingViT modules.

Part 1 (as before): plot ALL channels of the angle tensor exactly as the module
outputs them, with labels for what the module intends each channel to be:
    "dup"  -> should repeat another channel (rotate_half / interleaved layout)
    "pad"  -> beyond impl.rotary_dim, zero-padded by the module
    "pass" -> beyond freqs.shape[-1], so Megatron passes it through untouched

Part 2 (new): push the angles through Megatron's own
`_apply_rotary_pos_emb_bshd` -- the function that actually rotates q and k --
and check what it does with them:
    * channel pairing: which channel each channel is rotated against, and
      whether that pair gets a real rotation, a shear (one angle is 0),
      two different angles, or nothing
    * orthogonality: the per-position 2x2 blocks must be rotations
    * norm preservation: |RoPE(x)| == |x| at every position
    * relative-only scores: <R_m q, R_n k> == <q, R_(n-m) k>
    * untouched (NoPE) channel count vs. dim - rotary_dim
    * frequency schedule: compressed vs. p-RoPE (fastest freqs of the full
      schedule)
Then sweep several rope fractions and print a pass/fail table.
"""

import os

import matplotlib

SHOW = False  # True opens a window per figure (blocks until closed)
if not SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from megatron.core.models.common.embeddings import RotaryEmbeddingViT

try:
    from megatron.core.models.common.embeddings.rope_utils import (
        _apply_rotary_pos_emb_bshd as _UPSTREAM_APPLY,
    )
except ImportError:
    _UPSTREAM_APPLY = None


FIGURES_DIR = "../rope-figures"

H = 14
W = 14
DIM = 64
NUM_HEADS = 12
ROTARY_BASE = 100.0      # scalar, or a list of length NUM_HEADS; None = learned init
ROPE_FRACTION = 0.25     # fraction used for the plots, in (0, 1]
ROTATE_HALF = True       # passed to the module; upstream rotary_interleaved = not this
ROPE_IMPLS = ["axial", "mixed_axis", "polar", "mixed_polar", "hilbert"]

CHECK_FRACTIONS = [0.25, 0.5, 0.75, 1.0]  # fraction sweep for the summary table
USE_UPSTREAM_APPLY = True  # False -> use the reference copy below
TOL = 1e-4
N_SCORE_PAIRS = 512

PLOT_SINE = True         # plot sin(angle) (the rotary carrier) instead of raw angle
PER_CHANNEL_FILES = False  # True also saves one PNG per (head, channel)
SEED = 0                 # mixed_* variants draw a random direction per head


# ---------------------------------------------------------------------------
# Building the module and reading its angles
# ---------------------------------------------------------------------------

def build_rope(rope_impl, rope_fraction):
    torch.manual_seed(SEED)
    return RotaryEmbeddingViT(
        dim=DIM,
        num_heads=NUM_HEADS,
        rotary_base=ROTARY_BASE,
        rotate_half=ROTATE_HALF,
        rope_impl=rope_impl,
        rope_fraction=rope_fraction,
    )


def get_angles(rope, H, W, device="cpu"):
    """
    Raw module output [H*W, 1, heads, C] (C = whatever the module returns,
    dim if it pads, rotary_dim if it doesn't), plus the module's settings.
    """
    impl = rope.impl
    with torch.no_grad():
        raw = rope(H, W, device=device).detach().float()

    assert raw.dim() == 4 and raw.shape[0] == H * W and raw.shape[1] == 1, (
        f"expected [H*W, 1, heads, C], got {tuple(raw.shape)}"
    )
    assert raw.shape[-1] <= impl.dim and raw.shape[-1] % 2 == 0, (
        f"last dim {raw.shape[-1]} must be even and <= head dim {impl.dim}"
    )

    info = {
        "dim": impl.dim,
        "rotary_dim": impl.rotary_dim,
        # Megatron uses rot_dim = freqs.shape[-1]; this is what it will see.
        "upstream_rot_dim": raw.shape[-1],
        "rope_fraction": impl.rotary_dim / impl.dim,
        "rotate_half": impl.rotate_half,
        "num_heads": raw.shape[2],
    }
    return raw, info


def display_maps(raw, info, H, W):
    """Angle every head-dim channel actually receives, as [H, W, heads, dim]."""
    ang = raw[:, 0]
    missing = info["dim"] - ang.shape[-1]
    if missing:  # channels Megatron passes through untouched receive angle 0
        ang = torch.cat([ang, torch.zeros(*ang.shape[:-1], missing)], dim=-1)
    return ang.reshape(H, W, info["num_heads"], info["dim"])


def report_module_layout(raw, info):
    """Report-only checks of the layout the module itself claims to produce."""
    rd = info["rotary_dim"]
    rot = raw[:, 0, :, :rd]
    n = rd // 2
    if info["rotate_half"]:
        first, second = rot[..., :n], rot[..., n:]
    else:
        first, second = rot[..., 0::2], rot[..., 1::2]
    if not torch.allclose(first, second):
        print("  NOTE: module's duplicated rotary channels do not match")
    if raw.shape[-1] > rd:
        pad_max = raw[:, 0, :, rd:].abs().max().item()
        print(f"  NOTE: module zero-pads channels {rd}..{raw.shape[-1] - 1}"
              f" (max |angle| there = {pad_max:.3g})")


# ---------------------------------------------------------------------------
# Applying RoPE the way Megatron does
# ---------------------------------------------------------------------------

def _reference_apply(t, freqs, rotary_interleaved=False):
    """Copy of Megatron's unfused _apply_rotary_pos_emb_bshd (no MLA / mscale)."""
    rot_dim = freqs.shape[-1]
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    if rotary_interleaved:
        x1, x2 = t_rot[..., 0::2], t_rot[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).reshape(t_rot.shape)
    else:
        x1, x2 = torch.chunk(t_rot, 2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
    t_rot = t_rot * torch.cos(freqs) + rotated * torch.sin(freqs)
    return torch.cat((t_rot, t_pass), dim=-1)


def apply_rope(t, freqs, interleaved):
    """t: [s, b, heads, dim], freqs: [s, 1, heads, rot_dim]."""
    if USE_UPSTREAM_APPLY and _UPSTREAM_APPLY is not None:
        return _UPSTREAM_APPLY(t, freqs, rotary_interleaved=interleaved)
    return _reference_apply(t, freqs, rotary_interleaved=interleaved)


CATEGORIES = ["rotation", "identity", "pass-through", "shear", "mismatch"]
CAT_COLORS = ["#2a9d8f", "#9e9e9e", "#d9d9d9", "#e63946", "#f4a261"]


def analyze_pairs(raw, info):
    """
    Which channel Megatron rotates each channel against, and what the pair gets.

    rotate_half (not interleaved): channel i pairs with i + rot_dim/2,
    where rot_dim = freqs.shape[-1].  Interleaved: 2k pairs with 2k+1.
    """
    rot = info["upstream_rot_dim"]
    idx = torch.arange(rot)
    partner = (idx + rot // 2) % rot if info["rotate_half"] else idx ^ 1
    ang = raw[:, 0]  # [s, heads, rot]

    cats = []
    for c in range(info["dim"]):
        if c >= rot:
            cats.append("pass-through")
            continue
        a, b = ang[..., c], ang[..., partner[c]]
        a0, b0 = a.abs().max() < TOL, b.abs().max() < TOL
        if a0 and b0:
            cats.append("identity")
        elif torch.allclose(a, b, atol=TOL, rtol=TOL):
            cats.append("rotation")
        elif a0 or b0:
            cats.append("shear")
        else:
            cats.append("mismatch")

    partners = partner.tolist() + [None] * (info["dim"] - rot)
    pair_counts = {k: 0 for k in CATEGORIES}
    for c in range(rot):
        if c < partners[c]:
            pair_counts[cats[c]] += 1
    return cats, partners, pair_counts


def check_applied(raw, info, seed=SEED):
    """Run Megatron's apply function and test the properties RoPE must have."""
    g = torch.Generator().manual_seed(seed)
    interleaved = not info["rotate_half"]
    S, _, NH, _ = raw.shape
    dim = info["dim"]
    eye = torch.eye(dim)

    with torch.no_grad():
        # Exact linear map at every (position, head): rotate each basis vector.
        basis = eye[None, :, None, :].expand(S, dim, NH, dim).contiguous()
        out = apply_rope(basis, raw, interleaved)      # [S, basis, NH, dim]
        M = out.permute(0, 2, 3, 1).contiguous()      # [S, NH, out_ch, in_ch]

        orth = (M.transpose(-1, -2) @ M - eye).abs().amax(dim=(-1, -2))  # [S, NH]

        x = torch.randn(dim, generator=g)
        x = x / x.norm()
        norm_ratio = torch.einsum("shij,j->shi", M, x).norm(dim=-1)      # [S, NH]

        dev = (M - eye).abs().amax(dim=(0, 1))                           # [dim, dim]
        untouched = (dev.amax(dim=1) < TOL) & (dev.amax(dim=0) < TOL)

        # Scores: absolute positions vs. relative phase only.
        m = torch.randint(0, S, (N_SCORE_PAIRS,), generator=g)
        n = torch.randint(0, S, (N_SCORE_PAIRS,), generator=g)
        q = torch.randn(N_SCORE_PAIRS, NH, dim, generator=g)
        k = torch.randn(N_SCORE_PAIRS, NH, dim, generator=g)
        ang = raw[:, 0]

        def rot(v, a):
            return apply_rope(v[:, None], a[:, None], interleaved)[:, 0]

        s_abs = (rot(q, ang[m]) * rot(k, ang[n])).sum(-1)
        s_rel = (q * rot(k, ang[n] - ang[m])).sum(-1)
        rel_err = ((s_abs - s_rel).abs().max() / s_abs.abs().mean()).item()

    return {
        "orth": orth,
        "max_orth_err": orth.max().item(),
        "norm_ratio": norm_ratio,
        "max_norm_dev": (norm_ratio - 1).abs().max().item(),
        "n_untouched": int(untouched.sum()),
        "s_abs": s_abs.flatten(),
        "s_rel": s_rel.flatten(),
        "rel_err": rel_err,
    }


def schedule_check(rope_impl, raw, info):
    """
    Compare this module's frequencies with the full-fraction module's.

    Returns, for the first frequency block, which exponent index of the FULL
    schedule each rotary frequency equals.  p-RoPE semantics keep the fastest
    ones: [0, 1, 2, ...].  A compressed schedule gives e.g. [0, 4, 8, 12].
    """
    if info["rotary_dim"] == info["dim"]:
        return {"indices": None, "p_rope": True, "label": "full fraction"}

    full_raw, full_info = get_angles(build_rope(rope_impl, 1.0), H, W)

    def unique(r, inf):
        a = r[:, 0, :, : inf["rotary_dim"]]
        return a[..., : inf["rotary_dim"] // 2] if inf["rotate_half"] else a[..., 0::2]

    uf, uF = unique(raw, info), unique(full_raw, full_info)
    n_blocks = 1 if rope_impl == "hilbert" else 2
    bs_f, bs_F = uf.shape[-1] // n_blocks, uF.shape[-1] // n_blocks

    indices = []
    for u in range(bs_f):
        match = None
        for j in range(bs_F):
            if torch.allclose(uf[..., u], uF[..., j], atol=1e-5, rtol=1e-4):
                match = j
                break
        indices.append(match)

    p_rope = indices == list(range(bs_f))
    label = "p-RoPE (fastest of full schedule)" if p_rope else "compressed / re-spaced"
    return {"indices": indices, "n_full": bs_F, "p_rope": p_rope, "label": label}


def print_report(rope_impl, info, pair_counts, chk, sched):
    dim, rd, seen = info["dim"], info["rotary_dim"], info["upstream_rot_dim"]
    expected_nope = dim - rd

    def line(ok, text):
        print(f"    [{'PASS' if ok else 'FAIL'}] {text}")

    src = "upstream Megatron" if (USE_UPSTREAM_APPLY and _UPSTREAM_APPLY) else "reference copy"
    print(f"  applied with {src} _apply_rotary_pos_emb_bshd, "
          f"rotary_interleaved={not info['rotate_half']}")
    ok_dim = seen == rd
    print(f"    [{'PASS' if ok_dim else 'WARN'}] Megatron rot_dim = freqs.shape[-1] = {seen}, "
          f"module rotary_dim = {rd}")
    pc = pair_counts
    line(pc["shear"] == 0 and pc["mismatch"] == 0,
         f"pairs: {pc['rotation']} rotation, {pc['shear']} shear, "
         f"{pc['mismatch']} two-angle, {pc['identity']} identity")
    line(chk["max_orth_err"] < TOL, f"orthogonal: max |M^T M - I| = {chk['max_orth_err']:.3g}")
    line(chk["max_norm_dev"] < TOL, f"norm preserved: max ||RoPE(x)|/|x| - 1| = {chk['max_norm_dev']:.3g}")
    line(chk["rel_err"] < TOL, f"scores depend only on phase difference: rel err = {chk['rel_err']:.3g}")
    line(chk["n_untouched"] == expected_nope,
         f"untouched (NoPE) channels: {chk['n_untouched']} (expected {expected_nope})")
    if sched["indices"] is not None:
        print(f"    [INFO] schedule: {sched['label']}; frequency k uses full-schedule "
              f"exponent index {sched['indices']} of 0..{sched['n_full'] - 1}")


def verdict(info, pair_counts, chk):
    ok = (
        pair_counts["shear"] == 0
        and pair_counts["mismatch"] == 0
        and chk["max_orth_err"] < TOL
        and chk["max_norm_dev"] < TOL
        and chk["rel_err"] < TOL
        and chk["n_untouched"] == info["dim"] - info["rotary_dim"]
    )
    return "OK" if ok else "BROKEN"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _channel_tag(ch, info):
    """Short label: what the module intends this channel to be."""
    if ch >= info["upstream_rot_dim"]:
        return "pass"
    if ch >= info["rotary_dim"]:
        return "pad"
    n_freq = info["rotary_dim"] // 2
    if info["rotate_half"]:
        return "dup" if ch >= n_freq else ""
    return "dup" if ch % 2 == 1 else ""


def _prepare(maps, plot_sine):
    data = torch.sin(maps) if plot_sine else maps
    return data, data.min().item(), data.max().item()


def plot_head_grids(maps, info, cats, partners, out_dir, rope_impl, plot_sine=True):
    """One figure per head showing every channel, with its upstream pairing."""
    data, vmin, vmax = _prepare(maps, plot_sine)
    label = "sin(angle)" if plot_sine else "angle (rad)"
    num_heads, n_ch = info["num_heads"], info["dim"]
    cols = min(8, n_ch)
    rows = (n_ch + cols - 1) // cols

    for head in range(num_heads):
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 1.6 + 1.2, rows * 2.0 + 0.9),
            squeeze=False,
            layout="constrained",
        )
        head_angles = maps[:, :, head]
        im = None
        for ch, ax in enumerate(axes.flat):
            ax.axis("off")
            if ch < n_ch:
                im = ax.imshow(data[:, :, head, ch].numpy(), vmin=vmin, vmax=vmax)
                tag = _channel_tag(ch, info)
                a = head_angles[:, :, ch]
                pair = "" if partners[ch] is None else f"↔{partners[ch]} "
                bad = cats[ch] in ("shear", "mismatch")
                ax.set_title(
                    f"ch {ch} {tag}".strip()
                    + f"\n{pair}{cats[ch]}"
                    + f"\n[{a.min().item():.2f}, {a.max().item():.2f}] rad",
                    fontsize=6,
                    color="red" if bad else "black",
                )
        fig.colorbar(im, ax=axes, shrink=0.6, label=label)
        fig.suptitle(
            f"{rope_impl} head {head}  (rotary_dim {info['rotary_dim']}/{info['dim']}, "
            f"Megatron rot_dim {info['upstream_rot_dim']})\n"
            f"angle range this head: [{head_angles.min().item():.3g}, "
            f"{head_angles.max().item():.3g}] rad   |   "
            f"color scale: {label} in [{vmin:.3g}, {vmax:.3g}]",
            fontsize=9,
        )
        fig.savefig(f"{out_dir}/head_{head:02d}_grid.png", dpi=150)
        if SHOW:
            plt.show()
        plt.close(fig)
    print(f"  saved {num_heads} head grids to {out_dir}/")


def plot_channel_maps(maps, info, cats, partners, out_dir, rope_impl, plot_sine=True):
    """One PNG per (head, channel) for every channel, shared color scale."""
    data, vmin, vmax = _prepare(maps, plot_sine)
    label = "sin(angle)" if plot_sine else "angle (rad)"

    for head in range(info["num_heads"]):
        for ch in range(info["dim"]):
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(data[:, :, head, ch].numpy(), vmin=vmin, vmax=vmax)
            tag = _channel_tag(ch, info)
            a = maps[:, :, head, ch]
            pair = "" if partners[ch] is None else f"↔{partners[ch]} "
            ax.set_title(
                f"{rope_impl} head {head}, channel {ch} {tag}".strip()
                + f"\n{pair}{cats[ch]}, angle range "
                f"[{a.min().item():.3g}, {a.max().item():.3g}] rad",
                fontsize=9,
            )
            ax.set_xlabel("patch x")
            ax.set_ylabel("patch y")
            fig.colorbar(im, ax=ax, label=label)
            fig.tight_layout()
            fig.savefig(f"{out_dir}/head_{head:02d}_channel_{ch:02d}.png", dpi=200)
            if SHOW:
                plt.show()
            plt.close(fig)
    print(f"  saved {info['num_heads'] * info['dim']} channel maps to {out_dir}/")


def plot_frequency_growth(maps, info, out_dir, rope_impl):
    """Spatial std of the angle for every channel, one line per head."""
    magnitude = maps.std(dim=(0, 1))  # [heads, dim]

    fig, ax = plt.subplots(figsize=(7, 4))
    for head in range(info["num_heads"]):
        ax.plot(magnitude[head].numpy(), label=f"head {head}", linewidth=1)
    if magnitude.min() > 0:
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=1e-3)  # some channels get no rotation
    if info["rotate_half"]:
        ax.axvline(info["rotary_dim"] // 2 - 0.5, color="gray", ls=":", lw=1,
                   label="module: start of duplicate half")
        ax.axvline(info["upstream_rot_dim"] // 2 - 0.5, color="purple", ls="-.", lw=1,
                   label="Megatron: rotate_half split")
    if info["rotary_dim"] < info["dim"]:
        ax.axvline(info["rotary_dim"] - 0.5, color="red", ls="--", lw=1,
                   label="end of rotary channels")
    ax.set_title(f"{rope_impl}: angle spread per channel")
    ax.set_xlabel(f"channel (0..{info['dim'] - 1})")
    ax.set_ylabel("std of angle over positions (rad)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    filename = f"{out_dir}/frequency_growth_curve.png"
    fig.savefig(filename, dpi=200)
    if SHOW:
        plt.show()
    plt.close(fig)
    print(f"  saved {filename}")


def plot_checks(info, cats, chk, out_dir, rope_impl):
    """Four panels showing what Megatron actually does with these angles."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), layout="constrained")

    # (a) pairing strip
    ax = axes[0, 0]
    cat_idx = torch.tensor([CATEGORIES.index(c) for c in cats])[None, :]
    ax.imshow(cat_idx.numpy(), cmap=ListedColormap(CAT_COLORS), vmin=-0.5,
              vmax=len(CATEGORIES) - 0.5, aspect="auto")
    ax.set_yticks([])
    ax.set_xticks(range(0, info["dim"], 4))
    ax.set_xlabel("head-dim channel")
    split = "i ↔ i + rot_dim/2" if info["rotate_half"] else "2k ↔ 2k+1"
    ax.set_title(f"what each channel's pair gets ({split}, rot_dim={info['upstream_rot_dim']})",
                 fontsize=9)
    ax.axvline(info["rotary_dim"] - 0.5, color="black", lw=1, ls="--")
    present = [c for c in CATEGORIES if c in cats]
    ax.legend(handles=[Patch(color=CAT_COLORS[CATEGORIES.index(c)], label=c) for c in present],
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=len(present))

    # (b) norm ratio over positions, head 0
    ax = axes[0, 1]
    nr = chk["norm_ratio"][:, 0].reshape(H, W).numpy()
    im = ax.imshow(nr)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"|RoPE(x)| / |x| per position, head 0 (should be 1)\n"
                 f"range [{nr.min():.3f}, {nr.max():.3f}]", fontsize=9)

    # (c) orthogonality error over positions, max over heads
    ax = axes[1, 0]
    oe = chk["orth"].amax(dim=1).reshape(H, W).numpy()
    im = ax.imshow(oe)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"max |MᵀM − I| per position, all heads (should be 0)\n"
                 f"max {oe.max():.3g}", fontsize=9)

    # (d) absolute vs relative scores
    ax = axes[1, 1]
    s_abs, s_rel = chk["s_abs"].numpy(), chk["s_rel"].numpy()
    ax.scatter(s_rel, s_abs, s=3, alpha=0.4)
    lim = max(abs(s_abs).max(), abs(s_rel).max())
    ax.plot([-lim, lim], [-lim, lim], color="gray", lw=1, ls="--")
    ax.set_xlabel("<q, R(φn − φm) k>   (relative phase only)")
    ax.set_ylabel("<R(φm) q, R(φn) k>   (actual)")
    ax.set_title(f"q·k: on the diagonal = relative-only (rel err {chk['rel_err']:.2g})",
                 fontsize=9)

    fig.suptitle(f"{rope_impl}: RoPE as applied by Megatron "
                 f"(rotary_dim {info['rotary_dim']}/{info['dim']})")
    filename = f"{out_dir}/applied_rope_checks.png"
    fig.savefig(filename, dpi=150)
    if SHOW:
        plt.show()
    plt.close(fig)
    print(f"  saved {filename}")


# ---------------------------------------------------------------------------

def main():
    if USE_UPSTREAM_APPLY and _UPSTREAM_APPLY is None:
        print("WARNING: could not import Megatron's _apply_rotary_pos_emb_bshd; "
              "using the reference copy in this script")

    for rope_impl in ROPE_IMPLS:
        print(f"\n[{rope_impl}] rope_fraction={ROPE_FRACTION}")
        rope = build_rope(rope_impl, ROPE_FRACTION)
        raw, info = get_angles(rope, H, W)
        report_module_layout(raw, info)

        cats, partners, pair_counts = analyze_pairs(raw, info)
        chk = check_applied(raw, info)
        sched = schedule_check(rope_impl, raw, info)
        print_report(rope_impl, info, pair_counts, chk, sched)

        out_dir = os.path.join(FIGURES_DIR, rope_impl)
        os.makedirs(out_dir, exist_ok=True)
        maps = display_maps(raw, info, H, W)

        plot_head_grids(maps, info, cats, partners, out_dir, rope_impl, plot_sine=PLOT_SINE)
        if PER_CHANNEL_FILES:
            plot_channel_maps(maps, info, cats, partners, out_dir, rope_impl, plot_sine=PLOT_SINE)
        plot_frequency_growth(maps, info, out_dir, rope_impl)
        plot_checks(info, cats, chk, out_dir, rope_impl)

    print("\nFraction sweep (RoPE as applied by Megatron)")
    header = (f"{'impl':<12}{'frac':>5}{'rot_dim':>8}{'mega':>6}"
              f"{'rot':>5}{'shear':>6}{'2ang':>5}{'id':>4}"
              f"{'NoPE':>9}{'orth_err':>10}{'rel_err':>10}  schedule    verdict")
    print(header)
    print("-" * len(header))
    for rope_impl in ROPE_IMPLS:
        for frac in CHECK_FRACTIONS:
            raw, info = get_angles(build_rope(rope_impl, frac), H, W)
            _, _, pc = analyze_pairs(raw, info)
            chk = check_applied(raw, info)
            sched = schedule_check(rope_impl, raw, info)
            nope = f"{chk['n_untouched']}/{info['dim'] - info['rotary_dim']}"
            sch = "full" if sched["indices"] is None else ("p-RoPE" if sched["p_rope"] else "compressed")
            print(f"{rope_impl:<12}{frac:>5.2f}{info['rotary_dim']:>8}{info['upstream_rot_dim']:>6}"
                  f"{pc['rotation']:>5}{pc['shear']:>6}{pc['mismatch']:>5}{pc['identity']:>4}"
                  f"{nope:>9}{chk['max_orth_err']:>10.2g}{chk['rel_err']:>10.2g}  "
                  f"{sch:<11} {verdict(info, pc, chk)}")


if __name__ == "__main__":
    main()