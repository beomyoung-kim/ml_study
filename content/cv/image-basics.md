# Image Basics: Pixels, Channels, and Tensors

> [!NOTE] Goal of this chapter
> If you are new to computer vision, start here. Through diagrams and short code, this chapter makes one idea concrete: **to a computer, an image is ultimately just a grid of numbers**. That intuition is the foundation for every vision chapter that follows (classification → CNNs → detection → segmentation).

## An image is made of numbers

To us, a photograph is a scene. To a computer, it is a grid of tiny points called **pixels**, each represented by a **number** that indicates brightness. In a grayscale image, one pixel is typically a single integer from `0` (black) to `255` (white).

<figure>
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="90" y="18" text-anchor="middle" fill="#98a3b2">What we see</text>
  <g>
    <rect x="30" y="30" width="120" height="120" fill="none" stroke="#98a3b2"/>
    <rect x="30" y="30" width="30" height="30" fill="#111"/><rect x="60" y="30" width="30" height="30" fill="#444"/><rect x="90" y="30" width="30" height="30" fill="#888"/><rect x="120" y="30" width="30" height="30" fill="#bbb"/>
    <rect x="30" y="60" width="30" height="30" fill="#333"/><rect x="60" y="60" width="30" height="30" fill="#666"/><rect x="90" y="60" width="30" height="30" fill="#aaa"/><rect x="120" y="60" width="30" height="30" fill="#ddd"/>
    <rect x="30" y="90" width="30" height="30" fill="#555"/><rect x="60" y="90" width="30" height="30" fill="#888"/><rect x="90" y="90" width="30" height="30" fill="#ccc"/><rect x="120" y="90" width="30" height="30" fill="#fff"/>
    <rect x="30" y="120" width="30" height="30" fill="#777"/><rect x="60" y="120" width="30" height="30" fill="#aaa"/><rect x="90" y="120" width="30" height="30" fill="#eee"/><rect x="120" y="120" width="30" height="30" fill="#fff"/>
  </g>
  <text x="230" y="95" font-size="22" fill="#e0533f">→</text>
  <text x="410" y="18" text-anchor="middle" fill="#98a3b2">What the computer sees (a number grid)</text>
  <g font-size="12" fill="currentColor" text-anchor="middle">
    <rect x="290" y="30" width="240" height="120" fill="none" stroke="#0ea5e9"/>
    <text x="320" y="50">16</text><text x="380" y="50">68</text><text x="440" y="50">136</text><text x="500" y="50">187</text>
    <text x="320" y="80">51</text><text x="380" y="80">102</text><text x="440" y="80">170</text><text x="500" y="80">221</text>
    <text x="320" y="110">85</text><text x="380" y="110">136</text><text x="440" y="110">204</text><text x="500" y="110">255</text>
    <text x="320" y="140">119</text><text x="380" y="140">170</text><text x="440" y="140">238</text><text x="500" y="140">255</text>
  </g>
</svg>
<figcaption>A 4×4 grayscale image. The left shows the shades a person sees; the right shows the actual values a computer handles (0 = black … 255 = white). A vision model learns patterns from the numbers on the right.</figcaption>
</figure>

- **Resolution**: the size of the grid. `1920×1080` means 1,920 pixels wide and 1,080 pixels tall.
- **Bit depth**: the number of levels one channel can represent. Common unsigned 8-bit data has $2^8=256$ levels (0–255), but camera RAW, medical imaging, and HDR data may use 10/12/16-bit integers or floating point.

## Color = three channels

A color image is **several grayscale grids stacked together**. The most common representation, RGB, consists of three grids—or channels—for **red (R), green (G), and blue (B)**. Each pixel's color is a combination of three values. For example, `(255,0,0)` is pure red, `(255,255,255)` is white, and `(0,0,0)` is black.

<figure>
<svg viewBox="0 0 600 180" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="40" y="40" width="90" height="90" rx="6" fill="#e0533f" opacity="0.85"/>
  <rect x="60" y="30" width="90" height="90" rx="6" fill="#12a150" opacity="0.85"/>
  <rect x="80" y="20" width="90" height="90" rx="6" fill="#0ea5e9" opacity="0.85"/>
  <text x="120" y="150" text-anchor="middle" fill="#98a3b2">3 channels overlap to make color</text>
  <text x="250" y="80" font-size="22" fill="currentColor">=</text>
  <g text-anchor="middle">
    <rect x="300" y="45" width="70" height="70" rx="6" fill="none" stroke="#e0533f" stroke-width="2"/><text x="335" y="130" fill="#e0533f">R</text>
    <rect x="390" y="45" width="70" height="70" rx="6" fill="none" stroke="#12a150" stroke-width="2"/><text x="425" y="130" fill="#12a150">G</text>
    <rect x="480" y="45" width="70" height="70" rx="6" fill="none" stroke="#0ea5e9" stroke-width="2"/><text x="515" y="130" fill="#0ea5e9">B</text>
  </g>
  <text x="425" y="30" text-anchor="middle" fill="#98a3b2">Each is an (H×W) brightness grid</text>
</svg>
<figcaption>An RGB image is a stack of three (H×W) grids. Split the channels apart and each channel is once again a single grayscale image.</figcaption>
</figure>

## As a tensor: (H, W, C) vs (C, H, W)

When represented in code, an image becomes a **tensor**, as introduced in the [NumPy Primer](#/ml-coding/numpy-primer). Beginners need to know one crucial **layout difference**:

<dl class="kv">
<dt>NumPy / image libraries</dt><dd>Usually <b>(H, W, C)</b> — height, width, then channels. Example: <code>(1080, 1920, 3)</code>.</dd>
<dt>PyTorch deep learning</dt><dd>Usually <b>(C, H, W)</b> — channels first. With a batch, this becomes <b>(N, C, H, W)</b>, because convolution operators expect a channel-first layout.</dd>
</dl>

> [!WARNING] The most common practical bug
> Confusing `(H,W,C)` with `(C,H,W)` and forgetting a `transpose`. This produces corrupted colors or a shape error. At every boundary, check the **shape, dtype, value range, and channel order**. In NumPy, HWC→CHW is `img.transpose(2,0,1)`, but OpenCV defaults to BGR while PIL usually uses RGB, so a transpose alone does not fix the color order.

## Normalization: 0–255 → 0–1

Before feeding pixels to a model, we map them to a smaller range. The simplest approach is to **divide by 255 to obtain values in 0–1**. Usually we go one step further and **subtract a per-channel mean, then divide by a per-channel standard deviation**. This makes training more stable; [Normalization & Training Stability](#/foundations/normalization-stability) explains why.

$$
x_\text{norm} = \frac{x/255 - \mu}{\sigma}
$$

Some ImageNet-pretrained models use $\mu=[0.485,0.456,0.406]$ and $\sigma=[0.229,0.224,0.225]$. Other models expect `[0,1]` or `[-1,1]`, and may also require a particular resize interpolation, crop policy, and color order. **Treat the preprocessing metadata or configuration distributed with the weights as the source of truth.** Different preprocessing changes the input distribution.

## Why fix the image size?

Images in a batch tensor usually need the same height and width, and some classification heads or positional embeddings assume a particular training resolution. Fully convolutional models and many modern backbones, however, can accept multiple sizes through padding, adaptive pooling, or position interpolation. In either case, check the resize/crop policy used during training and the backbone stride (for example, whether dimensions must be multiples of 32). See [Vision Data Augmentation](#/cv/augmentation) for the augmentation details.

## Try it yourself — RGB to grayscale

The following is a widely used BT.601-family luma approximation for gamma-encoded RGB: $Y' = 0.299R' + 0.587G' + 0.114B'$. The coefficients and their meaning change with the color space and linear/gamma convention. This lab defines its input contract as 8-bit RGB with shape `(H,W,3)`.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"rgb_to_gray","packages":["numpy"],"approx":true,"tol":0.01,"starter":"def rgb_to_gray(img):\n    # img: a nested (H, W, 3) list with values in 0..255. Return an (H, W) grayscale image.\n    # Y = 0.299*R + 0.587*G + 0.114*B  (channel order R,G,B)\n    # Hint: convert to a NumPy array and take a weighted sum over the last (channel) axis -> img @ [0.299,0.587,0.114]\n    pass","tests":[{"args":[[[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]]],"expect":[[76.245,149.685],[29.07,255.0]]},{"args":[[[[0,0,0]]]],"expect":[[0.0]]},{"args":[[[[100,100,100]]]],"expect":[[100.0]]}],"solution":"import numpy as np\n\ndef rgb_to_gray(img):\n    a = np.asarray(img, dtype=float)\n    if a.ndim != 3 or a.shape[-1] != 3:\n        raise ValueError(\"img must have shape (H, W, 3)\")\n    if not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 255)):\n        raise ValueError(\"pixel values must be finite and in [0, 255]\")\n    w = np.array([0.299, 0.587, 0.114])\n    return (a @ w).tolist()"}
</script>
</div>

Look at the third test: because the weights sum to 1, `(100,100,100)` remains `100`. This agrees with the fact that grayscale pixels have R=G=B.

## Q&A

<details class="qa"><summary>Why normalize pixels to 0–1? Can't we just feed in values from 0–255?</summary>
<div class="qa-body">

**Short answer:** Large input values can destabilize gradients and slow training.

**In depth:** Values as large as 0–255 are multiplied by the initial weights, which can produce large activations and make the learning rate harder to tune. Scaling to 0–1—or applying mean/std normalization—standardizes the input scale and improves [conditioning](#/foundations/linear-algebra-calculus). A pretrained model must reproduce the **same** normalization used during pretraining to perform as expected.
</div></details>

<details class="qa"><summary>What about an alpha channel (transparency), or a single-channel grayscale image?</summary>
<div class="qa-body">

**Short answer:** The same principles apply; only the number of channels changes.

**In depth:** RGBA has four channels, with A for transparency; grayscale has one. The input must match the number of channels expected by the model's first convolutional layer. You can replicate grayscale into three channels, or replace the first layer and retrain it. Do not automatically discard alpha: first determine whether it is straight or premultiplied, then composite it against a background. Transparent RGB pixels may contain arbitrary hidden colors.
</div></details>

## Cheat sheet

| Concept | In one line |
| --- | --- |
| Image | A grid of pixels (numbers); grayscale commonly uses 0–255 |
| Channel | Color = a stack of red, green, and blue grids |
| Layout | NumPy (H,W,C) · PyTorch (C,H,W) / batched (N,C,H,W) |
| Normalization | 0–255 → 0–1, usually followed by per-channel mean/std |
| Fixed size | Resize/crop to satisfy the model's input contract |

**Next:** [Image Classification](#/cv/classification) · [NumPy Primer](#/ml-coding/numpy-primer) · [Vision Data Augmentation](#/cv/augmentation)
