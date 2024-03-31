import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt
import os
import asyncio

async def generate_signal(n, max_frequency, Fs):
    loop = asyncio.get_event_loop()
    random_signal = np.random.normal(0, 10, n)
    time_values = np.arange(n) / Fs
    w = max_frequency / (Fs / 2)
    filter_params = signal.butter(3, w, 'low', output='sos')
    filtered_signal = await loop.run_in_executor(None, signal.sosfiltfilt, filter_params, random_signal)
    return time_values, filtered_signal

if not os.path.exists("figures"):
    os.makedirs("figures")

variances = []
snr_ratios = []

async def main():
    n = 500
    Fs = 1000
    F_max = 21


    random_signal = np.random.normal(0, 10, n)


    time_values = np.arange(n) / Fs


    w = F_max / (Fs / 2)
    filter_params = signal.butter(3, w, 'low', output='sos')


    filtered_signal = signal.sosfiltfilt(filter_params, random_signal)


    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(time_values, filtered_signal, linewidth=1)
    ax.set_xlabel('Час, с', fontsize=14)
    ax.set_ylabel('Сигнал', fontsize=14)
    plt.title('Відфільтрований сигнал', fontsize=14)
    plt.savefig('./figures/signal.png', dpi=600)


    spectrum = fft.fft(filtered_signal)
    spectrum = np.abs(fft.fftshift(spectrum))
    freqs = fft.fftfreq(n, 1 / Fs)
    freqs = fft.fftshift(freqs)


    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(freqs, spectrum, linewidth=1)
    ax.set_xlabel('Частота, Гц', fontsize=14)
    ax.set_ylabel('Спектр', fontsize=14)
    plt.title('Спектр', fontsize=14)


    plt.savefig('./figures/spectrum.png', dpi=600)


    M_values = [4, 16, 64, 256]

    fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
    variances = []
    snr_ratios = []

    for s in range(4):
        M = M_values[s]
        delta = (np.max(filtered_signal) - np.min(filtered_signal)) / (M - 1)
        quantize_signal = delta * np.round(filtered_signal / delta)


        quantize_levels = np.arange(np.min(quantize_signal), np.max(quantize_signal) + 1, delta)
        quantize_bit = [format(bits, '0' + str(int(np.log2(M))) + 'b') for bits in np.arange(0, M)]
        quantize_table = np.c_[quantize_levels[:M], quantize_bit[:M]]

        fig_table, ax_table = plt.subplots(figsize=(14 / 2.54, M / 2.54))
        table = ax_table.table(cellText=quantize_table, colLabels=['Значення сигналу', 'Послідовність кодів'],
                               loc='center')
        table.set_fontsize(14)
        table.scale(1, 2)
        ax_table.axis('off')
        plt.savefig(f"./figures/Таблиця квантування для {M} рівнів.png", dpi=600)
        plt.close(fig_table)


        bits = []
        for signal_value in quantize_signal:
            for index, value in enumerate(quantize_levels[:M]):
                if np.round(np.abs(signal_value - value), 0) == 0:
                    bits.append(quantize_bit[index])
                    break

        bits = [int(item) for item in list(''.join(bits))]


        fig_bits, ax_bits = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
        ax_bits.step(np.arange(0, len(bits)), bits, linewidth=0.1)
        ax_bits.set_xlabel('Відліки')
        ax_bits.set_ylabel('Послідовність бітів')
        ax_bits.set_title(f'Послідовність бітів для {M} рівнів квантування')
        plt.savefig(f"./figures/Послідовність бітів для {M} рівнів.png", dpi=600)
        plt.close(fig_bits)


        variances.append(np.var(quantize_signal))
        snr_ratios.append(np.var(filtered_signal) / np.var(quantize_signal))


        i, j = divmod(s, 2)
        ax[i][j].step(time_values, quantize_signal, linewidth=1, where='post', label=f'M = {M}')

    fig.suptitle("Цифрові сигнали з різними рівнями квантування", fontsize=14)
    fig.supxlabel("Час", fontsize=14)
    fig.supylabel("Амплітуда цифрового сигналу", fontsize=14)


    plt.savefig("figures/Цифрові сигнали з різними рівнями квантування.png", dpi=600)


    plt.show()


    fig_variance, ax_variance = plt.subplots(figsize=(10, 6))
    ax_variance.plot(M_values, variances, marker='o', color='b', label='Дисперсія цифрового сигналу')
    ax_variance.set_xlabel('Кількість рівнів квантування')
    ax_variance.set_ylabel('Дисперсія')
    ax_variance.set_xscale('log', base=2)
    ax_variance.legend()
    plt.title("Залежність дисперсії цифрового сигналу від кількості рівнів квантування")


    plt.savefig("figures/Дисперсія цифрового сигналу.png", dpi=600)


    plt.show()


    fig_snr, ax_snr = plt.subplots(figsize=(10, 6))
    ax_snr.plot(M_values, snr_ratios, marker='o', color='r', label='Співвідношення сигнал-шум')
    ax_snr.set_xlabel('Кількість рівнів квантування')
    ax_snr.set_ylabel('Співвідношення сигнал-шум')
    ax_snr.set_xscale('log', base=2)
    ax_snr.legend()
    plt.title("Залежність співвідношення сигнал-шум від кількості рівнів квантування")


    plt.savefig("figures/Співвідношення сигнал-шум.png", dpi=600)


    plt.show()

asyncio.run(main())
