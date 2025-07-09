import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from rknnlite.api import RKNNLite
from tqdm import tqdm


# 参数设置
MAX_LENGTH = 200
DURATION = 3  # 秒
CHANNELS = 1
OUTPUT_FILE = "test.wav"
TTS_AUDIO_OUTPUT = "tts_output.wav"
ASR_ENCODER_MODEL = "onnx_or_rknn/whisper_encoder_base_20s.rknn"
ASR_DECODER_MODEL = "onnx_or_rknn/whisper_decoder_base_20s.rknn"
TTS_ENCODER_MODEL = "onnx_or_rknn/mms_tts_eng_encoder_200.rknn"
TTS_DECODER_MODEL = "onnx_or_rknn/mms_tts_eng_decoder_200.rknn"
VOCAB_FILE = "vocab.txt"
MEL_FILTER_TXT = "mel_80_filters.txt"   # 80×201 ASCII 表

USE_NPU = True



def record_audio(duration, samplerate, channels, filename):
    print(f"[INFO] 开始录音 {duration} 秒 ...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=channels, dtype='int16')
    sd.wait()
    
    # 检查音频质量
    audio_abs = np.abs(audio)
    max_amplitude = np.max(audio_abs)
    mean_amplitude = np.mean(audio_abs)
    # 改进音频检测 - 计算超过噪声阈值的样本比例
    noise_threshold = 500  # 噪声阈值
    voice_ratio = np.sum(audio_abs > noise_threshold) / len(audio)
    
    print(f"[INFO] 录音统计: 最大振幅={max_amplitude}, 平均振幅={mean_amplitude:.2f}, 有声比例={voice_ratio:.2%}")
    
    # 如果音量太小，提升音量
    if max_amplitude > 0 and max_amplitude < 2000:
        # 计算增益，使最大值达到10000左右，但不要太大
        gain = min(10.0, 10000 / max_amplitude)
        print(f"[INFO] 音量较小，应用增益: {gain:.2f}倍")
        audio = np.int16(np.clip(audio * gain, -32768, 32767))
    
    # 应用简单的降噪处理 - 小于噪声阈值的样本置零
    if mean_amplitude < 500:  # 如果整体音量很低
        noise_gate = noise_threshold / 2
        audio = np.where(np.abs(audio) < noise_gate, 0, audio)
        print(f"[INFO] 应用简单降噪处理，噪声阈值: {noise_gate}")
    
    sf.write(filename, audio, samplerate)
    print(f"[INFO] 录音已保存到 {filename}")
    
    # 判断录音质量
    if voice_ratio < 0.01:
        print("[警告] 几乎没有检测到语音，请确保麦克风正常工作")
    elif max_amplitude < 1000:
        print("[警告] 录音音量很小，识别效果可能不佳")


def play_audio(filename):
    print(f"[INFO] 正在播放 {filename} ...")
    data, fs = sf.read(filename, dtype='int16')
    sd.play(data, fs)
    sd.wait()
    print("[INFO] 播放结束")


def load_rknn_model(model_path, npu_core):
    rknn = RKNNLite()
    print(f"[INFO] 加载模型: {model_path}")
    if rknn.load_rknn(model_path) != 0:
        raise RuntimeError("模型加载失败")
    if rknn.init_runtime(core_mask=npu_core) != 0:
        raise RuntimeError("RKNN 初始化失败")
    return rknn


# 固定参数（与 RKNN 编码器训练时保持一致）
SAMPLE_RATE   = 16000
N_FFT         = 400
HOP_LENGTH    = 160
N_MELS        = 80
CHUNK_SECONDS = 20
N_SAMPLES     = SAMPLE_RATE * CHUNK_SECONDS      # 320 000

# 优化参数 - 并行处理开关
USE_PARALLEL_PROCESSING = True  # 启用并行处理以提高速度


def mel_filters(n_mels):
    if n_mels != 80:
        raise ValueError("只支持 80 Mel 频带")
    mels_path = "mel_80_filters.txt"
    return np.loadtxt(mels_path, dtype=np.float32).reshape(n_mels, N_FFT // 2 + 1)
def pad_or_trim_asr(array, length=N_SAMPLES, axis=-1):
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array

def preprocess_audio(file_path: str) -> np.ndarray:
    """返回形状 (1, 80, 3000) 的 log-mel 特征（float32）"""
    wav, sr = sf.read(file_path)
    if sr != SAMPLE_RATE:
        raise ValueError("采样率必须为 16 kHz")
    if wav.ndim > 1:                       # 取单声道
        wav = wav[:, 0]
    wav_array = np.array(wav, dtype=np.float32)
    audio = pad_or_trim_asr(wav_array.flatten())

    # Hann 窗
    window = np.hanning(N_FFT).astype(np.float32)

    # 帧切
    num_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, N_FFT),
        strides=(audio.strides[0] * HOP_LENGTH, audio.strides[0])
    )

    # STFT 幅度平方 -> 功率谱  (F=201, T)
    stft = np.fft.rfft(frames * window, n=N_FFT, axis=1)
    power_spec = np.abs(stft) ** 2  # (T, 201)
    power_spec = power_spec.T  # (201, T)

    # Mel 滤波  (80, 201) @ (201, T) -> (80, T)
    mel_bank = mel_filters(N_MELS)  # (80, 201)
    mel_spec = mel_bank @ power_spec

    # 对数与归一化
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0  # 区间 [0, 1]
    # 右侧 0 填充 / 截断到 2000 帧  ← 这是模型的固定需求
    if log_spec.shape[1] < 2000:
        log_spec = np.pad(log_spec, ((0, 0), (0, 2000 - log_spec.shape[1])))
    else:
        log_spec = log_spec[:, :2000]
    return log_spec[np.newaxis, :, :].astype(np.float32)  # (1, 80, 2000)

def load_vocab(path):
    vocab = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            key = parts[0]
            value = parts[1] if len(parts) > 1 else ""
            vocab[key] = value
    return vocab

def decode_autoregressive(encoder_output: np.ndarray,
                          decoder,
                          vocab: dict) -> str:
    """
    encoder_output : 编码器输出 (1, *, D)  float32
    decoder        : RKNN 对象（已 init）
    vocab          : {token_id(str): token(str)}
    """
    # Whisper 固定 token
    end_token = 50257
    next_token = 50258
    timestamp_begin = 50364
    tokens = [50258, 50259, 50359, 50363]
    max_tokens = 12
    pop_id = max_tokens
    tokens_str = ''
    tokens = tokens * int(max_tokens/4)

    while next_token != end_token:
        dec_out = decoder.inference(
            inputs=[np.asarray([tokens], dtype=np.int64), encoder_output]
        )[0]  # (1, len+1, V)

        next_token = dec_out[0, -1].argmax()
        next_token_str = vocab[str(next_token)]
        tokens.append(next_token)

        if next_token == end_token:
            tokens.pop(-1)
            next_token = tokens[-1]
            break
        if next_token > timestamp_begin:
            continue
        if pop_id > 4:
            pop_id -= 1

        tokens.pop(pop_id)
        tokens_str += next_token_str

    result = tokens_str.replace('\u0120', ' ').replace('<|endoftext|>', '')
    return result



def run_asr(audio_file):
    # 检查文件是否存在且有效
    if not os.path.exists(audio_file):
        print(f"[ERROR] 音频文件不存在: {audio_file}")
        return "[BLANK_AUDIO]"
    
    try:
        # 检查音频文件内容
        audio_data, sr = sf.read(audio_file, dtype='int16')
        
        # 检查音频是否太安静或者太短
        mean_amp = np.mean(np.abs(audio_data))
        max_amp = np.max(np.abs(audio_data))
        voice_ratio = np.sum(np.abs(audio_data) > 500) / len(audio_data)
        
        print(f"[INFO] 音频检查: 最大幅度={max_amp}, 平均幅度={mean_amp:.2f}, 有声比例={voice_ratio:.2%}")
        
        # 如果音频太安静或者几乎没有声音，返回空白标记
        if max_amp < 500 or voice_ratio < 0.005:
            print("[警告] 检测到空白或极低音量的录音")
            return "[BLANK_AUDIO]"
        
        # 预处理可以在主线程进行
        print("[INFO] 预处理音频...")
        mel = preprocess_audio(audio_file)
        
        if USE_PARALLEL_PROCESSING:
            # 并行加载模型，使用多线程提高速度
            print("[INFO] 并行加载Whisper模型...")
            import threading
            
            encoder_ready = False
            decoder_ready = False
            encoder = None
            decoder = None
            vocab = None
            
            # 加载编码器线程
            def load_encoder():
                nonlocal encoder, encoder_ready
                encoder = load_rknn_model(ASR_ENCODER_MODEL, RKNNLite.NPU_CORE_0)
                encoder_ready = True
                
            # 加载解码器线程
            def load_decoder():
                nonlocal decoder, decoder_ready
                decoder = load_rknn_model(ASR_DECODER_MODEL, RKNNLite.NPU_CORE_1)
                decoder_ready = True
                
            # 加载词汇表线程
            def load_vocab_data():
                nonlocal vocab
                vocab = load_vocab(VOCAB_FILE)
            
            # 并行启动三个线程
            t1 = threading.Thread(target=load_encoder)
            t2 = threading.Thread(target=load_decoder)
            t3 = threading.Thread(target=load_vocab_data)
            
            t1.start()
            t2.start()
            t3.start()
            
            # 等待所有线程完成
            t3.join()  # 加载词汇表通常很快
            
            # 等待编码器就绪
            t1.join()
            print("[INFO] 编码器加载完成")
            
            # 运行编码器
            print("[INFO] 运行 Whisper 编码器...")
            encoder_output = encoder.inference(inputs=[mel])[0]
            
            # 等待解码器就绪
            t2.join()
            print("[INFO] 解码器加载完成")
            
            # 运行解码器
            print("[INFO] 运行 Whisper 解码器...")
            text = decode_autoregressive(encoder_output, decoder, vocab)
            
            # 释放资源
            encoder.release()
            decoder.release()
        else:
            # 原有的串行执行方式
            encoder = load_rknn_model(ASR_ENCODER_MODEL, RKNNLite.NPU_CORE_0)
            decoder = load_rknn_model(ASR_DECODER_MODEL, RKNNLite.NPU_CORE_1)
            vocab = load_vocab(VOCAB_FILE)

            print("[INFO] 运行 Whisper 编码器...")
            encoder_output = encoder.inference(inputs=[mel])[0]

            print("[INFO] 运行 Whisper 解码器...")
            text = decode_autoregressive(encoder_output, decoder, vocab)
            
            encoder.release()
            decoder.release()

        # 检查结果是否为空或太短
        if not text or len(text.strip()) < 2:
            print("[警告] ASR结果太短或为空")
            return "[BLANK_AUDIO]"

        print(f"[ASR 结果] {text}")
        return text
    except Exception as e:
        print(f"[ERROR] ASR处理失败: {e}")
        import traceback
        traceback.print_exc()
        return "[ERROR]"

def pad_or_trim_tts(token_id, attention_mask, max_length):
    pad_len = max_length - len(token_id)
    if pad_len <= 0:
        token_id = token_id[:max_length]
        attention_mask = attention_mask[:max_length]

    if pad_len > 0:
        token_id = token_id + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return token_id, attention_mask

def preprocess_input(text, vocab, max_length=MAX_LENGTH):
    text = list(text.lower())
    input_id = []
    for token in text:
        if token not in vocab:
            continue
        input_id.append(0)
        input_id.append(int(vocab[token]))
    input_id.append(0)
    attention_mask = [1] * len(input_id)

    input_id, attention_mask = pad_or_trim_tts(input_id, attention_mask, max_length)

    input_ids_array = np.array(input_id)[None, ...]
    attention_mask_array = np.array(attention_mask)[None, ...]

    return input_ids_array, attention_mask_array

def middle_process(log_duration, input_padding_mask, max_length):
    log_duration = np.array(log_duration)
    input_padding_mask = np.array(input_padding_mask)

    length_scale = 1.0
    duration = np.ceil(np.exp(log_duration) * input_padding_mask * length_scale)
    predicted_lengths = np.clip(duration.sum(axis=(1, 2)), a_min=1, a_max=None).astype(np.int32)

    predicted_lengths_max_real = predicted_lengths.max()
    predicted_lengths_max = max_length * 2

    batch_size, _, in_len = duration.shape
    out_len = predicted_lengths_max

    indices = np.arange(out_len)
    output_padding_mask = (indices[None, :] < predicted_lengths[:, None]).astype(np.float32)
    output_padding_mask = output_padding_mask[:, None, :]  # (B, 1, T_out)

    attn = np.zeros((batch_size, 1, out_len, in_len), dtype=np.float32)

    for b in range(batch_size):
        t = 0
        for i in range(in_len):
            d = int(duration[b, 0, i])
            attn[b, 0, t:t + d, i] = 1.0
            t += d
            if t >= out_len:
                break
        attn[b, 0, :, :] *= output_padding_mask[b].transpose(1, 0)

    return attn, output_padding_mask, predicted_lengths_max_real



def run_tts(text, max_duration=None):
    # 如果文本过长，截断以加快处理速度
    if len(text) > 300:  # 超过300个字符截断
        text = text[:300]
        print(f"[INFO] 文本过长，截断至300字符以提高速度")

    if USE_PARALLEL_PROCESSING:
        # 并行加载模型
        print("[INFO] 并行加载TTS模型...")
        import threading
        
        encoder_ready = False
        decoder_ready = False
        encoder = None
        decoder = None
        
        # 加载编码器线程
        def load_encoder():
            nonlocal encoder, encoder_ready
            encoder = load_rknn_model(TTS_ENCODER_MODEL, RKNNLite.NPU_CORE_0)
            encoder_ready = True
            
        # 加载解码器线程
        def load_decoder():
            nonlocal decoder, decoder_ready
            decoder = load_rknn_model(TTS_DECODER_MODEL, RKNNLite.NPU_CORE_1)
            decoder_ready = True
        
        # 并行启动两个线程
        t1 = threading.Thread(target=load_encoder)
        t2 = threading.Thread(target=load_decoder)
        
        t1.start()
        t2.start()
        
        # 准备TTS输入数据
        print("[INFO] 准备TTS输入数据...")
        vocab = {' ': 19, "'": 1, '-': 14, '0': 23, '1': 15, '2': 28, '3': 11, '4': 27, '5': 35, '6': 36, '_': 30,
                 'a': 26, 'b': 24, 'c': 12, 'd': 5, 'e': 7, 'f': 20, 'g': 37, 'h': 6, 'i': 18, 'j': 16, 'k': 0, 'l': 21,
                 'm': 17,
                 'n': 29, 'o': 22, 'p': 13, 'q': 34, 'r': 25, 's': 8, 't': 33, 'u': 4, 'v': 32, 'w': 9, 'x': 31, 'y': 3,
                 'z': 2, '–': 10}
        input_ids_array, attention_mask_array = preprocess_input(text, vocab, max_length=MAX_LENGTH)
        
        # 等待编码器就绪
        t1.join()
        print("[INFO] TTS编码器加载完成")
        
        # 运行TTS编码器
        print("[INFO] 运行 TTS 编码器...")
        log_duration, input_padding_mask, prior_means, prior_log_variances = encoder.inference(inputs=[input_ids_array, attention_mask_array])
        attn, output_padding_mask, predicted_lengths_max_real = middle_process(log_duration, input_padding_mask, MAX_LENGTH)
        
        # 等待解码器就绪
        t2.join()
        print("[INFO] TTS解码器加载完成")
        
        # 运行TTS解码器
        print("[INFO] 运行 TTS 解码器...")
        waveform = decoder.inference(inputs=[attn, output_padding_mask, prior_means, prior_log_variances])[0]
        
        # 释放资源
        encoder.release()
        decoder.release()
    else:
        # 原有的串行执行方式
        encoder = load_rknn_model(TTS_ENCODER_MODEL, RKNNLite.NPU_CORE_0)
        decoder = load_rknn_model(TTS_DECODER_MODEL, RKNNLite.NPU_CORE_1)

        vocab = {' ': 19, "'": 1, '-': 14, '0': 23, '1': 15, '2': 28, '3': 11, '4': 27, '5': 35, '6': 36, '_': 30,
                'a': 26, 'b': 24, 'c': 12, 'd': 5, 'e': 7, 'f': 20, 'g': 37, 'h': 6, 'i': 18, 'j': 16, 'k': 0, 'l': 21,
                'm': 17,
                'n': 29, 'o': 22, 'p': 13, 'q': 34, 'r': 25, 's': 8, 't': 33, 'u': 4, 'v': 32, 'w': 9, 'x': 31, 'y': 3,
                'z': 2, '–': 10}

        input_ids_array, attention_mask_array = preprocess_input(text, vocab, max_length=MAX_LENGTH)

        print("[INFO] 运行 TTS 编码器...")
        log_duration, input_padding_mask, prior_means, prior_log_variances = encoder.inference(inputs=[input_ids_array, attention_mask_array])
        attn, output_padding_mask, predicted_lengths_max_real = middle_process(log_duration, input_padding_mask, MAX_LENGTH)

        print("[INFO] 运行 TTS 解码器...")
        waveform = decoder.inference(inputs=[attn, output_padding_mask, prior_means, prior_log_variances])[0]
        
        # 释放资源
        encoder.release()
        decoder.release()

    # Post process
    sample_rate = 16000  # 16kHz
    
    # 计算最大时间限制
    if max_duration:
        max_samples = int(max_duration * sample_rate)
        # 样本数量超过限制时才截断
        if predicted_lengths_max_real * 256 > max_samples:
            print(f"[INFO] 输出音频截断到 {max_duration} 秒")
            sf.write(file=TTS_AUDIO_OUTPUT, data=np.array(waveform[0][:max_samples]), samplerate=sample_rate)
        else:
            sf.write(file=TTS_AUDIO_OUTPUT, data=np.array(waveform[0][:predicted_lengths_max_real * 256]), samplerate=sample_rate)
    else:
        sf.write(file=TTS_AUDIO_OUTPUT, data=np.array(waveform[0][:predicted_lengths_max_real * 256]), samplerate=sample_rate)
    
    print(f"[INFO] TTS输出文件已保存到: {TTS_AUDIO_OUTPUT}")


if __name__ == "__main__":
    print("[DEBUG] 已进入主函数")
    try:
        # record_audio(DURATION, SAMPLE_RATE, CHANNELS, OUTPUT_FILE)
        # play_audio(OUTPUT_FILE)
        text = run_asr(OUTPUT_FILE)
        run_tts(text)
        play_audio(TTS_AUDIO_OUTPUT)
        print("[SUCCESS] 语音->文字->语音 全流程完成。")
    except Exception as e:
        print(f"[ERROR] 操作失败: {e}")
