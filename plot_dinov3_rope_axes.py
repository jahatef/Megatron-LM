import os
import torch
import matplotlib.pyplot as plt

from megatron.core.models.common.embeddings import RotaryEmbeddingAxial, RotaryEmbeddingViT, RotaryEmbeddingMixedAxis, RotaryEmbeddingHilbert


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

    '''coords = rope.build_coords(H, W, device)
    print(coords.)
    coords = coords[:, :, None]

    periods = rope.periods[None, None, :].to(device)

    angles = 2 * torch.pi * coords * periods

    # flatten axis dimension into channel dimension
    angles = angles.flatten(1, 2)
    '''
    angles = rope(H,W, device=device)
    print("angles size. " ,angles.size())
    return angles.reshape(H, W, angles.size()[2],angles.size()[3])


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
    num_heads=12,
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

    for head in range(num_heads):
        for ch in range(num_channels):

            plt.figure(figsize=(4, 4))
            print(channel_maps.size())

            if plot_sine:
                image = torch.sin(channel_maps[:, :, head, ch])
            else:
                image = channel_maps[:, :, ch]

            plt.imshow(
                image.detach().cpu().numpy(),
                #vmin=vmin,
                #vmax=vmax,
            )

            plt.title(f"RoPE frequency channel {ch}")

            plt.colorbar()

            plt.tight_layout()

            filename = f"{FIGURES_DIR}/head_{head}_channel_{ch}.png"

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

    magnitude = channel_maps.std(dim=(0, 1)).detach().cpu()

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

    H = 14
    W = 14
    DIM = 64
    num_heads=12
    #rope = RotaryEmbeddingViT(dim=DIM, num_heads=12,rotary_base=torch.tensor([1,4,6,4,3,6,78,4,3,54,9,19])*100, rope_impl="axial")
    rope = RotaryEmbeddingViT(dim=DIM, num_heads=num_heads,rotary_base=1, rope_impl="mixed_polar")


    channel_maps = extract_channel_maps(
        rope,
        H,
        W,
        device="cpu",
    )
    print(channel_maps.size())

    print("Plotting per-channel spatial frequencies 📊")
    channel_maps = torch.load("angles.pt").resize(H,W,12,64)
    print(channel_maps.size())
    plot_channel_maps(
        channel_maps,
        num_channels=32,
        num_heads=num_heads,
        plot_sine=True,  # change to True to visualize actual rotary carrier
    )

    print("Plotting frequency growth curve 📈")

    plot_frequency_growth(channel_maps)


if __name__ == "__main__":
    main()