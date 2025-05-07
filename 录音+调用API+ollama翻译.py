import sounddevice as sd
import numpy as np
import requests
import wave
import io
import time
import threading
import queue
import json # For Ollama communication

# --- 配置参数 ---
FUNASR_API_URL = "http://localhost:5000/transcribe"  # FunASR API 服务地址
OLLAMA_API_URL = "http://localhost:11434/api"      # Ollama API 基础地址

# TARGET_SAMPLE_RATE 仍然用于尝试以该速率打开麦克风
# 但如果失败，将使用设备的默认采样率，并且该采样率将用于发送给FunASR
TARGET_SAMPLE_RATE = 16000  # 尝试使用的目标采样率 (Hz)
CHANNELS = 1  # 声道数 (单声道)
BLOCK_DURATION_S = 0.05  # 每次回调处理的音频块时长（秒）
VOLUME_THRESHOLD = 0.02  # 音量阈值 (RMS, 范围 0.0 到 1.0)

SPEECH_CONFIRM_S = 0.3  # 声音持续超过阈值多少秒后开始录制
SILENCE_CONFIRM_S = 0.8  # 声音持续低于阈值多少秒后停止录制

# --- 全局变量 ---
recording_frames = []
is_currently_recording = False
consecutive_silent_blocks = 0
consecutive_speech_blocks = 0
audio_processing_queue = queue.Queue()
selected_device_id = None
actual_sample_rate = TARGET_SAMPLE_RATE # 麦克风实际使用的采样率
block_size_frames = 0
speech_trigger_blocks_count = 0
silence_trigger_blocks_count = 0

# Ollama 相关全局变量
selected_ollama_model = None
selected_target_language_code = None
target_language_name = ""

# 支持的翻译目标语言
SUPPORTED_LANGUAGES = {
    "en": "English (英文)",
    "zh": "Chinese (中文)",
    "ja": "Japanese (日文)",
    "ko": "Korean (韩文)"
}


def list_microphones():
    """列出所有可用的输入设备 (麦克风) 并让用户选择。"""
    print("可用的麦克风设备:")
    devices = sd.query_devices()
    input_devices_info = []
    
    idx_counter = 0
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default_sr = device.get('default_samplerate', 'N/A')
            device_name = device['name']
            host_api_name = sd.query_hostapis(device['hostapi'])['name'] if 'hostapi' in device and device['hostapi'] is not None else 'N/A'
            print(f"  {idx_counter}) {device_name} (ID: {i}, 默认采样率: {default_sr} Hz, HostAPI: {host_api_name})")
            input_devices_info.append({'id': i, 'name': device_name, 'default_samplerate': default_sr})
            idx_counter += 1

    if not input_devices_info:
        print("错误：未找到可用的麦克风设备。")
        return None

    while True:
        try:
            choice = int(input("请选择麦克风设备的序号: "))
            if 0 <= choice < len(input_devices_info):
                return input_devices_info[choice]['id']
            else:
                print("无效的选择，请输入列表中的序号。")
        except ValueError:
            print("无效的输入，请输入数字。")

def get_ollama_models():
    """从Ollama API获取可用的模型列表。"""
    print("\n正在从 Ollama 获取可用模型列表...")
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        response.raise_for_status()
        models_data = response.json()
        if "models" in models_data and models_data["models"]:
            return [model["name"] for model in models_data["models"]]
        else:
            print("Ollama 中没有找到可用的模型。")
            return []
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到 Ollama 服务 ({OLLAMA_API_URL}/tags)。请确保 Ollama 正在运行。")
        return []
    except requests.exceptions.RequestException as e:
        print(f"获取 Ollama 模型列表时出错: {e}")
        return []
    except json.JSONDecodeError:
        print(f"错误: 解析 Ollama 模型列表响应失败。响应内容: {response.text}")
        return []


def select_ollama_model(models):
    """让用户从列表中选择一个Ollama模型。"""
    if not models:
        return None
    
    print("\n请选择用于翻译的 Ollama 模型:")
    for i, model_name in enumerate(models):
        print(f"  {i}) {model_name}")
    
    while True:
        try:
            choice = int(input("请输入模型的序号: "))
            if 0 <= choice < len(models):
                return models[choice]
            else:
                print("无效的选择，请输入列表中的序号。")
        except ValueError:
            print("无效的输入，请输入数字。")

def select_target_language():
    """让用户选择目标翻译语言。"""
    print("\n请选择目标翻译语言:")
    lang_codes = list(SUPPORTED_LANGUAGES.keys())
    for i, code in enumerate(lang_codes):
        print(f"  {i}) {SUPPORTED_LANGUAGES[code]}")
    
    while True:
        try:
            choice = int(input("请输入目标语言的序号: "))
            if 0 <= choice < len(lang_codes):
                selected_code = lang_codes[choice]
                return selected_code, SUPPORTED_LANGUAGES[selected_code]
            else:
                print("无效的选择，请输入列表中的序号。")
        except ValueError:
            print("无效的输入，请输入数字。")


def translate_text_with_ollama(text_to_translate, model_name, target_lang_name_for_prompt):
    """使用Ollama翻译文本。"""
    print(f"\n正在使用 Ollama 模型 '{model_name}' 将文本翻译为 '{target_lang_name_for_prompt}'...")
    
    # 构建一个更明确的提示，告知模型输入语言可能是中文（来自FunASR）
    # 或者可以尝试让模型自动检测源语言，但这取决于模型能力
    prompt = f"Please translate the following text into {target_lang_name_for_prompt}. The original text might be in Chinese or another language recognized by an ASR system. Translate accurately:\n\n{text_to_translate} ; \n\nOnly reply to the translated content, do not say anything unnecessary!"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # 获取单个完整响应
    }
    
    try:
        response = requests.post(f"{OLLAMA_API_URL}/generate", json=payload, timeout=60) # 增加超时
        response.raise_for_status()
        ollama_response_data = response.json()
        
        if "response" in ollama_response_data:
            return ollama_response_data["response"].strip()
        elif "error" in ollama_response_data:
            print(f"Ollama API 返回错误: {ollama_response_data['error']}")
            return None
        else:
            print(f"Ollama API 返回了未知格式的响应: {ollama_response_data}")
            return None
            
    except requests.exceptions.Timeout:
        print("错误: Ollama API 请求超时。")
        return None
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到 Ollama 服务进行翻译。")
        return None
    except requests.exceptions.RequestException as e:
        print(f"调用 Ollama API 进行翻译时出错: {e}")
        return None
    except json.JSONDecodeError:
        print(f"错误: 解析 Ollama 翻译响应失败。响应内容: {response.text}")
        return None

def calculate_rms(data):
    """计算音频数据的均方根 (RMS)。"""
    return np.sqrt(np.mean(data**2))

def process_transcription_and_translate(transcribed_text):
    """处理识别结果并进行翻译"""
    global selected_ollama_model, target_language_name
    
    print("\n----------------------------------------")
    print("原文 (FunASR):")
    print(transcribed_text)
    print("----------------------------------------")

    if selected_ollama_model and target_language_name:
        # 从 target_language_name (e.g., "English (英文)") 中提取纯英文名称给Ollama
        ollama_target_lang_prompt_name = target_language_name.split(" (")[0]
        
        translated_text = translate_text_with_ollama(transcribed_text, selected_ollama_model, ollama_target_lang_prompt_name)
        if translated_text:
            print(f"\n译文 ({target_language_name}):")
            print(translated_text)
            print("----------------------------------------")
        else:
            print("\n翻译失败或未返回有效译文。")
            print("----------------------------------------")
    else:
        print("\n未选择 Ollama 模型或目标语言，跳过翻译。")
        print("----------------------------------------")


def save_and_send_audio_to_funasr(audio_data_np, recorded_sample_rate, funasr_api_url):
    """
    将音频数据保存为WAV格式，以其录制采样率发送到FunASR API，并处理结果。
    """
    print("\n正在处理录音数据并发送到 FunASR...")
    if audio_data_np.size == 0:
        print("没有录到音频数据。")
        return

    final_sample_rate_for_api = recorded_sample_rate

    try:
        audio_data_int16 = (audio_data_np * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(int(final_sample_rate_for_api))
            wf.writeframes(audio_data_int16.tobytes())
        wav_buffer.seek(0)

        files = {'audio_file': ('recording.wav', wav_buffer, 'audio/wav')}
        duration_sec = len(audio_data_int16) / final_sample_rate_for_api
        print(f"发送 {duration_sec:.2f} 秒的音频 (采样率 {final_sample_rate_for_api}Hz) 到 FunASR API: {funasr_api_url}")
        
        response = requests.post(funasr_api_url, files=files, timeout=30)
        response.raise_for_status()
        result = response.json()

        if "transcription" in result and result["transcription"]:
            process_transcription_and_translate(result["transcription"])
        elif "error" in result:
            print(f"FunASR API 返回错误: {result['error']}")
        else:
            print(f"FunASR API 未返回有效识别结果或错误信息: {result}")

    except requests.exceptions.Timeout:
        print("错误: FunASR API 请求超时。")
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到 FunASR API ({funasr_api_url})。请确保服务正在运行。")
    except requests.exceptions.RequestException as e:
        print(f"发送到 FunASR API 时出错: {e}")
    except Exception as e:
        print(f"处理录音或 FunASR API 响应时发生意外错误: {e}")
    finally:
        if 'wav_buffer' in locals() and wav_buffer:
            wav_buffer.close()


def audio_processing_worker():
    """工作线程，从队列中获取录音数据并处理。"""
    while True:
        try:
            audio_data_np, rate = audio_processing_queue.get(timeout=1)
            save_and_send_audio_to_funasr(audio_data_np, rate, FUNASR_API_URL)
            audio_processing_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"音频处理工作线程发生错误: {e}")


def audio_callback(indata, frames, callback_time, status):
    """音频输入流的回调函数。"""
    global is_currently_recording, recording_frames
    global consecutive_silent_blocks, consecutive_speech_blocks
    global speech_trigger_blocks_count, silence_trigger_blocks_count, actual_sample_rate

    if status:
        print(f"音频流状态信息: {status}", flush=True)

    volume_norm = calculate_rms(indata[:, 0])

    if is_currently_recording:
        recording_frames.append(indata.copy())
        if volume_norm < VOLUME_THRESHOLD:
            consecutive_silent_blocks += 1
            if consecutive_silent_blocks >= silence_trigger_blocks_count:
                print(f"\n检测到静音 ({SILENCE_CONFIRM_S}s)，停止录音。", flush=True)
                is_currently_recording = False
                if recording_frames:
                    full_recording = np.concatenate(recording_frames)
                    audio_processing_queue.put((full_recording, actual_sample_rate))
                recording_frames = []
                consecutive_speech_blocks = 0
        else:
            consecutive_silent_blocks = 0
            print(".", end="", flush=True)
    else: 
        if volume_norm >= VOLUME_THRESHOLD:
            consecutive_speech_blocks += 1
            if consecutive_speech_blocks >= speech_trigger_blocks_count:
                print(f"\n检测到语音 ({SPEECH_CONFIRM_S}s)，开始录音...", flush=True)
                is_currently_recording = True
                recording_frames = [indata.copy()]
                consecutive_silent_blocks = 0
        else:
            consecutive_speech_blocks = 0

def main():
    global selected_device_id, actual_sample_rate, block_size_frames
    global speech_trigger_blocks_count, silence_trigger_blocks_count
    global selected_ollama_model, selected_target_language_code, target_language_name

    # 1. 选择Ollama模型
    ollama_models = get_ollama_models()
    if not ollama_models:
        print("未能获取Ollama模型列表，或列表为空。请检查Ollama服务和模型。程序将退出。")
        return
    selected_ollama_model = select_ollama_model(ollama_models)
    if not selected_ollama_model:
        print("未选择Ollama模型。程序将退出。")
        return
    print(f"已选择Ollama模型: {selected_ollama_model}")

    # 2. 选择目标翻译语言
    selected_target_language_code, target_language_name = select_target_language()
    print(f"已选择目标翻译语言: {target_language_name} (代码: {selected_target_language_code})")


    # 3. 选择麦克风设备
    selected_device_id = list_microphones()
    if selected_device_id is None:
        return
    print(f"\n您选择了麦克风设备 ID: {selected_device_id}")

    try:
        sd.check_input_settings(device=selected_device_id, samplerate=TARGET_SAMPLE_RATE, channels=CHANNELS)
        actual_sample_rate = TARGET_SAMPLE_RATE
        print(f"设备 ID {selected_device_id} 支持目标采样率 {TARGET_SAMPLE_RATE} Hz 和 {CHANNELS} 声道。")
        print(f"将以 {actual_sample_rate} Hz 进行录制和发送给 FunASR。")
    except sd.PortAudioError:
        print(f"设备 ID {selected_device_id} 不直接支持目标采样率 {TARGET_SAMPLE_RATE} Hz。")
        try:
            device_info = sd.query_devices(selected_device_id)
            default_sr_str = device_info.get('default_samplerate', '')
            if not default_sr_str: raise ValueError("设备未提供有效的默认采样率。")
            default_sr = int(float(default_sr_str))
            if default_sr <= 0: raise ValueError(f"设备的默认采样率 ({default_sr}) 无效。")

            print(f"尝试使用设备的默认采样率: {default_sr} Hz。")
            sd.check_input_settings(device=selected_device_id, samplerate=default_sr, channels=CHANNELS)
            actual_sample_rate = default_sr
            print(f"将以设备的默认采样率 {actual_sample_rate} Hz 进行录制和发送给 FunASR。")
            if actual_sample_rate != TARGET_SAMPLE_RATE:
                 print(f"警告: FunASR API 通常期望 {TARGET_SAMPLE_RATE} Hz。以 {actual_sample_rate} Hz 发送可能影响识别效果。")
        except (sd.PortAudioError, ValueError, TypeError) as e_default:
            print(f"错误: 设备 ID {selected_device_id} 也不支持其默认采样率，或默认采样率无效/无法使用。错误信息: {e_default}")
            return
        except Exception as e_check:
            print(f"检查设备默认采样率时发生未知错误: {e_check}")
            return
    except Exception as e:
        print(f"检查设备能力时发生未知错误: {e}")
        return

    block_size_frames = int(actual_sample_rate * BLOCK_DURATION_S)
    speech_trigger_blocks_count = int(SPEECH_CONFIRM_S / BLOCK_DURATION_S)
    silence_trigger_blocks_count = int(SILENCE_CONFIRM_S / BLOCK_DURATION_S)

    print(f"实际使用麦克风采样率: {actual_sample_rate} Hz")
    print(f"音频块大小: {block_size_frames} 帧 ({BLOCK_DURATION_S*1000:.0f} ms)")
    print(f"音量阈值 (RMS): {VOLUME_THRESHOLD}")
    print(f"语音确认时长: {SPEECH_CONFIRM_S} 秒 ({speech_trigger_blocks_count} 个连续音频块)")
    print(f"静音确认时长: {SILENCE_CONFIRM_S} 秒 ({silence_trigger_blocks_count} 个连续音频块)")
    print("\n正在监听麦克风... (按 Ctrl+C 退出)")

    worker_thread = threading.Thread(target=audio_processing_worker, daemon=True)
    worker_thread.start()

    try:
        with sd.InputStream(device=selected_device_id,
                             channels=CHANNELS,
                             samplerate=actual_sample_rate,
                             blocksize=block_size_frames,
                             callback=audio_callback,
                             dtype='float32'):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n程序被用户中断。正在退出...")
    except Exception as e:
        print(f"打开音频流或主循环时发生错误: {e}")
    finally:
        print("程序结束。")
        if audio_processing_queue and not audio_processing_queue.empty():
            print("等待剩余音频处理完成...")
            audio_processing_queue.join() 
        print("所有任务已处理完毕。")

if __name__ == "__main__":
    main()
