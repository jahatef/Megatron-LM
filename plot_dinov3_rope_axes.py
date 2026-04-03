import os
import torch
import matplotlib.pyplot as plt

from megatron.core.models.common.embeddings import RotaryEmbeddingDinoV3, RotaryEmbeddingViT


FIGURES_DIR = "figures"


def ensure_output_dir():
    """
    Create figures directory if it does not exist.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)


def extract_channel_maps(rope, H, W, device="cpu"):
    """
    Extract RoPE angle maps per channel BEFORE rotate_half duplication.

    Output shape:
        [H, W, num_channels]
    """

    coords = rope.build_coords(H, W, device)
    coords = coords[:, :, None]

    periods = rope.periods[None, None, :].to(device)

    angles = 2 * torch.pi * coords * periods

    # flatten axis dimension into channel dimension
    angles = angles.flatten(1, 2)

    return angles.reshape(H, W, -1)


def compute_global_color_range(channel_maps, num_channels, plot_sine):
    """
    Compute shared vmin / vmax across selected channels.
    """

    subset = channel_maps[:, :, :num_channels]

    if plot_sine:
        subset = torch.sin(subset)

    vmin = subset.min().item()
    vmax = subset.max().item()

    print(f"Global color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

    return vmin, vmax


def plot_channel_maps(
    channel_maps,
    num_channels=12,
    show=True,
    plot_sine=False,
):
    """
    Plot spatial frequency maps for each channel
    using shared color axis scaling.
    """

    ensure_output_dir()

    vmin, vmax = compute_global_color_range(
        channel_maps,
        num_channels,
        plot_sine,
    )

    for ch in range(num_channels):

        plt.figure(figsize=(4, 4))

        if plot_sine:
            image = torch.sin(channel_maps[:, :, ch])
        else:
            image = channel_maps[:, :, ch]

        plt.imshow(
            image.cpu(),
            #vmin=vmin,
            #vmax=vmax,
        )

        plt.title(f"RoPE frequency channel {ch}")

        plt.colorbar()

        plt.tight_layout()

        filename = f"{FIGURES_DIR}/channel_{ch}.png"

        plt.savefig(filename, dpi=200)

        print(f"Saved: {filename}")

        if show:
            plt.show()

        plt.close()


def plot_frequency_growth(channel_maps, show=True):
    """
    Plot magnitude growth curve across channels.
    Helps verify exponential frequency scaling.
    """

    ensure_output_dir()

    magnitude = channel_maps.std(dim=(0, 1)).cpu()

    plt.figure()

    plt.plot(magnitude)

    plt.title("RoPE frequency magnitude per channel")

    plt.xlabel("channel")

    plt.ylabel("std(freq)")

    plt.tight_layout()

    filename = f"{FIGURES_DIR}/frequency_growth_curve.png"

    plt.savefig(filename, dpi=200)

    print(f"Saved: {filename}")

    if show:
        plt.show()

    plt.close()


def main():

    H = 8
    W = 8
    DIM = 64

    rope = RotaryEmbeddingViT(dim=DIM, rope_impl="hilbert",temperature=100)

    channel_maps = extract_channel_maps(
        rope,
        H,
        W,
        device="cpu",
    )

    print("Plotting per-channel spatial frequencies 📊")

    plot_channel_maps(
        channel_maps,
        num_channels=32,
        plot_sine=False,  # change to True to visualize actual rotary carrier
    )

    print("Plotting frequency growth curve 📈")

    plot_frequency_growth(channel_maps)


if __name__ == "__main__":
    main()