import numpy as np
import scipy.signal
import functools


@functools.lru_cache()
def gen_thirdoct_filter(fs, fft_size, cf0):
    """
    To generate 1/3 octave filter, where octave frequency defined here is based 2.

    params:
        fs: samplerate of signal.
        fft_size: fft size.
        num_bands: number of bands.
        cf0: center frequency of first band.

    returns:
        filter: thirdoct filter, whose shape is FxB.
                OCT[TxB] = STFT[TxF] @ Filter[FxB],
                where T is number of frames, F is fft size, B is number of bands.
        cfs: center frequencies of each bands.

    references:
        [1]: https://en.wikipedia.org/wiki/Octave_band
    """
    freq = np.linspace(0, fs / 2, fft_size // 2 + 1)

    if fs == 10000:
        # To keep consistency with 10kHz STOI
        num_bands = int(np.floor(np.log2(fs / 2 / (2 ** (1 / 6)) / cf0) * 3)) + 1
    else:
        # cover all frequency
        num_bands = int(np.ceil(np.log2(fs / 2 / (2 ** (1 / 6)) / cf0) * 3)) + 1

    band_idx = np.arange(0, num_bands)

    # center freqs
    cfs = cf0 * (2 ** (band_idx / 3.0))
    # upper bound
    ufs = cfs * (2 ** (1 / 6))
    # lower bound
    lfs = cfs / (2 ** (1 / 6))

    filter = np.zeros([fft_size // 2 + 1, num_bands])

    uf_idx = np.argmin(
        np.abs(ufs.repeat(fft_size // 2 + 1).reshape(-1, fft_size // 2 + 1) - freq[None]), axis=-1
    )
    lf_idx = np.argmin(
        np.abs(lfs.repeat(fft_size // 2 + 1).reshape(-1, fft_size // 2 + 1) - freq[None]), axis=-1
    )

    for k in range(num_bands):
        if fs == 10000:
            # To keep consistency with 10kHz STOI
            filter[lf_idx[k] : uf_idx[k], k] = 1
        else:
            filter[lf_idx[k] : uf_idx[k] + 1, k] = 1

    return filter


def framing(x, win_size, hop_size):
    num_frames = (len(x) - win_size) // hop_size + 1
    frames = [x[k * hop_size : k * hop_size + win_size] for k in range(num_frames)]
    frames = np.vstack(frames)
    return frames, num_frames


def ola(frames, win_size, hop_size):
    num_frames = len(frames)
    x = np.zeros((num_frames - 1) * hop_size + win_size)
    for k in range(num_frames):
        x[k * hop_size : k * hop_size + win_size] += frames[k]
    return x


def remove_silence(ref, deg, max_range, win_size, hop_size):
    ref_frames, num_frames = framing(ref, win_size, hop_size)
    deg_frames, num_frames = framing(deg, win_size, hop_size)

    mask = np.zeros(num_frames)
    win = np.hanning(win_size + 2)[1:-1]

    ref_frames = ref_frames * win[None]
    deg_frames = deg_frames * win[None]

    mask = 20 * np.log10(np.linalg.norm(ref_frames, axis=-1) / np.sqrt(win_size) + 1e-100)
    mask = (mask - np.max(mask)) > -max_range
    frame_idx = np.nonzero(mask)[0]

    # return ref_frames[frame_idx], deg_frames[frame_idx]

    # To keep consistency with 10kHz STOI
    ref_frames = framing(ola(ref_frames[frame_idx], win_size, hop_size), win_size, hop_size)[0] * win[None]
    deg_frames = framing(ola(deg_frames[frame_idx], win_size, hop_size), win_size, hop_size)[0] * win[None]
    return ref_frames, deg_frames



def corr(x, y):
    x_norm = x - np.mean(x, axis=-1, keepdims=True)
    x_norm = x_norm / (np.linalg.norm(x_norm, axis=-1, keepdims=True) + 1e-20)
    y_norm = y - np.mean(y, axis=-1, keepdims=True)
    y_norm = y_norm / (np.linalg.norm(y_norm, axis=-1, keepdims=True) + 1e-20)
    return np.sum(x_norm * y_norm, axis=-1)


def osstoi(ref, deg, fs=32000):
    """
    stoi for other samplerate.

    params:
        ref_sig:
        x: reference speech (clean speech).
        y: degraded speech (noisy speech or processed speech).
        fs: samplerate of speech signal.
    returns:
        os-stoi scores

    reference:
        [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
        Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
        IEEE Transactions on Audio, Speech, and Language Processing, 2011.
    """
    # sanity check
    if len(ref) != len(deg):
        raise ValueError("x and y should have the same length")

    win_size = 2 ** int(np.ceil(np.log2(25 * fs / 1000.0)))  # in samples, 25ms window
    hop_size = win_size // 2
    fft_size = win_size * 2  # in samples
    seg_size = int(384 / 1000 * fs / hop_size)  # in frames, 384 ms
    min_sdr = -15
    min_scale = 1 + 10 ** (-min_sdr / 20)
    max_range = 40

    cf0 = 150  # in Hz, center freqeucny of first band
    thirdoct_filter = gen_thirdoct_filter(fs, fft_size, cf0)

    # remove silent frames
    ref_frames, deg_frames = remove_silence(ref, deg, max_range, win_size, hop_size)

    ref_stft = np.fft.rfft(ref_frames, fft_size)
    deg_stft = np.fft.rfft(deg_frames, fft_size)

    ref_band = np.sqrt((np.abs(ref_stft) ** 2) @ thirdoct_filter)
    deg_band = np.sqrt((np.abs(deg_stft) ** 2) @ thirdoct_filter)

    ref_segs = np.concatenate(
        [ref_band[k : k + seg_size][None] for k in range(0, ref_band.shape[0] - seg_size)], axis=0
    )
    deg_segs = np.concatenate(
        [deg_band[k : k + seg_size][None] for k in range(0, deg_band.shape[0] - seg_size)], axis=0
    )

    # [num_segs, seg_size, num_bands]
    scale = np.linalg.norm(ref_segs, axis=1) / np.linalg.norm(deg_segs, axis=1)
    deg_segs_clipped = np.minimum(
        scale[:, None, :] * deg_segs,
        min_scale * ref_segs,
    )

    ref_segs = np.swapaxes(ref_segs, 1, 2)
    deg_segs_clipped = np.swapaxes(deg_segs_clipped, 1, 2)

    scores = corr(ref_segs, deg_segs_clipped)
    return np.mean(scores)

