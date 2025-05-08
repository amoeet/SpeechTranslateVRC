import sounddevice as sd
import numpy as np
import requests
import wave
import io
import time
import threading
import queue
import json
import argparse
import gradio as gr
from pythonosc import udp_client
 


 


# --- 全局变量 ---
recording_frames = []
is_currently_recording = False
consecutive_silent_blocks = 0
consecutive_speech_blocks = 0
audio_processing_queue = queue.Queue()
selected_device_id = None
actual_sample_rate = 48000  # 默认采样率
block_size_frames = 0
speech_trigger_blocks_count = 0
silence_trigger_blocks_count = 0

# Ollama 相关全局变量
selected_ollama_model = None
selected_target_language_code = None
target_language_name = ""

# OSC 相关全局变量
osc_client = None

# 设备和流

worker_thread = None
is_running = False

# 程序状态
log_messages = []

# 支持的翻译目标语言
SUPPORTED_LANGUAGES = {
    "en": "English (英文)",
    "zh": "Chinese (中文)",
    "ja": "Japanese (日文)",
    "ko": "Korean (韩文)"
}

# --- 配置参数 (默认值) ---
TARGET_SAMPLE_RATE = 16000  # 尝试使用的目标采样率 (Hz)
CHANNELS = 1  # 声道数 (单声道)
BLOCK_DURATION_S = 0.05  # 每次回调处理的音频块时长（秒）
VOLUME_THRESHOLD = 0.02  # 音量阈值 (RMS, 范围 0.0 到 1.0)
SPEECH_CONFIRM_S = 0.3  # 声音持续超过阈值多少秒后开始录制
SILENCE_CONFIRM_S = 0.8  # 声音持续低于阈值多少秒后停止录制
PRE_RECORD_SECONDS = 0.5 # 定义前置录音的秒数

# 我们需要一个地方来存储前置录音的音频帧
pre_record_buffer = []


def log_message(message):
    """记录消息并更新UI日志"""
    global log_messages
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    log_messages.append(formatted_message)
    # 保留最近的100条消息，防止日志过长
    if len(log_messages) > 100:
        log_messages = log_messages[-100:]
    return "\n".join(log_messages)


def get_microphones():
    """获取所有可用麦克风设备"""
    devices = sd.query_devices()
    input_devices_info = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default_sr = device.get('default_samplerate', 'N/A')
            device_name = device['name']
            host_api_name = sd.query_hostapis(device['hostapi'])['name'] if 'hostapi' in device and device['hostapi'] is not None else 'N/A'
            info = f"{device_name} (ID: {i}, 默认采样率: {default_sr} Hz, HostAPI: {host_api_name})"
            input_devices_info.append((info, i))
    
    return input_devices_info


def get_ollama_models(ollama_api_url):
    """从Ollama API获取可用的模型列表"""
    log_message("正在从 Ollama 获取可用模型列表...")
    try:
        response = requests.get(f"{ollama_api_url}/api/tags")
        response.raise_for_status()
        models_data = response.json()
        if "models" in models_data and models_data["models"]:
            models = [(model["name"], model["name"]) for model in models_data["models"]]
            log_message(f"获取到 {len(models)} 个 Ollama 模型")
            return models
        else:
            log_message("Ollama 中没有找到可用的模型")
            return []
    except requests.exceptions.ConnectionError:
        log_message(f"错误: 无法连接到 Ollama 服务 ({ollama_api_url}/api/tags)。请确保 Ollama 正在运行。")
        return []
    except requests.exceptions.RequestException as e:
        log_message(f"获取 Ollama 模型列表时出错: {e}")
        return []
    except json.JSONDecodeError:
        log_message(f"错误: 解析 Ollama 模型列表响应失败。")
        return []


def calculate_rms(data):
    """计算音频数据的均方根 (RMS)"""
    return np.sqrt(np.mean(data**2))


def translate_text_with_ollama(text_to_translate, model_name, target_lang_name_for_prompt, ollama_api_url):
    """使用Ollama翻译文本"""
    log_message(f"正在使用 Ollama 模型 '{model_name}' 将文本翻译为 '{target_lang_name_for_prompt}'...")
    
    prompt = f"Please translate the following text into {target_lang_name_for_prompt}. The original text might be in Chinese or another language recognized by an ASR system. Translate accurately and provide only the translated text:\n\n{text_to_translate}"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(f"{ollama_api_url}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        ollama_response_data = response.json()
        
        if "response" in ollama_response_data:
            return ollama_response_data["response"].strip()
        elif "error" in ollama_response_data:
            log_message(f"Ollama API 返回错误: {ollama_response_data['error']}")
            return None
        else:
            log_message(f"Ollama API 返回了未知格式的响应")
            return None
            
    except requests.exceptions.Timeout:
        log_message("错误: Ollama API 请求超时")
        return None
    except requests.exceptions.ConnectionError:
        log_message(f"错误: 无法连接到 Ollama 服务进行翻译")
        return None
    except requests.exceptions.RequestException as e:
        log_message(f"调用 Ollama API 进行翻译时出错: {e}")
        return None
    except json.JSONDecodeError:
        log_message(f"错误: 解析 Ollama 翻译响应失败")
        return None


def send_text_to_vrchat(original_text, translated_text, osc_client):
    """将原文和译文通过OSC发送到VRChat"""
    if not osc_client:
        log_message("OSC 客户端未初始化，跳过发送到 VRChat")
        return

    if translated_text:  # 确保有译文才发送
        message_to_send = f"{original_text}\n{translated_text}"
        # VRChat 聊天框对消息长度有限制
        max_bytes = 140 
        message_bytes = message_to_send.encode('utf-8')
        if len(message_bytes) > max_bytes:
            temp_message = message_to_send
            while len(temp_message.encode('utf-8')) > max_bytes:
                temp_message = temp_message[:-1]
            message_to_send = temp_message + "..." if temp_message != message_to_send else temp_message
            log_message(f"警告: 消息过长，已截断")

        try:
            osc_client.send_message("/chatbox/input", [message_to_send, True, False])
            log_message(f"已通过 OSC 发送消息到 VRChat")
        except Exception as e:
            log_message(f"通过 OSC 发送消息到 VRChat 时出错: {e}")
    else:
        # 如果只有原文，也可以选择只发送原文
        try:
            osc_client.send_message("/chatbox/input", [original_text, True, False])
            log_message(f"已通过 OSC (仅原文) 发送消息到 VRChat")
        except Exception as e:
            log_message(f"通过 OSC (仅原文) 发送消息到 VRChat 时出错: {e}")


def process_transcription_and_translate(transcribed_text, ollama_model, target_lang_code, ollama_api_url, osc_client):
    """处理识别结果，进行翻译，并发送到VRChat"""
    
    log_message("\n----------------------------------------")
    log_message("原文 (FunASR):")
    log_message(transcribed_text)
    log_message("----------------------------------------")

    translated_text_content = None

    if ollama_model and target_lang_code:
        target_lang_name = SUPPORTED_LANGUAGES[target_lang_code]
        ollama_target_lang_prompt_name = target_lang_name.split(" (")[0]
        translated_text_content = translate_text_with_ollama(
            transcribed_text, 
            ollama_model, 
            ollama_target_lang_prompt_name,
            ollama_api_url
        )
        
        if translated_text_content:
            log_message(f"\n译文 ({target_lang_name}):")
            log_message(translated_text_content)
            log_message("----------------------------------------")
        else:
            log_message("\n翻译失败或未返回有效译文")
            log_message("----------------------------------------")
    else:
        log_message("\n未选择 Ollama 模型或目标语言，跳过翻译")
        log_message("----------------------------------------")
    
    # 无论翻译是否成功，都尝试发送
    send_text_to_vrchat(transcribed_text, translated_text_content, osc_client)


def save_and_send_audio_to_funasr(audio_data_np, recorded_sample_rate, funasr_api_url, ollama_model, target_lang_code, ollama_api_url, osc_client):
    """将音频数据保存为WAV格式，发送到FunASR API，并处理结果"""
    log_message("正在处理录音数据并发送到 FunASR...")
    if audio_data_np.size == 0:
        log_message("没有录到音频数据")
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
        log_message(f"发送 {duration_sec:.2f} 秒的音频 (采样率 {final_sample_rate_for_api}Hz) 到 FunASR API")
        
        response = requests.post(funasr_api_url, files=files, timeout=30)
        response.raise_for_status()
        result = response.json()

        if "transcription" in result and result["transcription"]:
            process_transcription_and_translate(
                result["transcription"], 
                ollama_model, 
                target_lang_code,
                ollama_api_url,
                osc_client
            )
        elif "error" in result:
            log_message(f"FunASR API 返回错误: {result['error']}")
        else:
            log_message(f"FunASR API 未返回有效识别结果或错误信息")

    except requests.exceptions.Timeout:
        log_message("错误: FunASR API 请求超时")
    except requests.exceptions.ConnectionError:
        log_message(f"错误: 无法连接到 FunASR API。请确保服务正在运行")
    except requests.exceptions.RequestException as e:
        log_message(f"发送到 FunASR API 时出错: {e}")
    except Exception as e:
        log_message(f"处理录音或 FunASR API 响应时发生意外错误: {e}")
    finally:
        if 'wav_buffer' in locals() and wav_buffer:
            wav_buffer.close()


def audio_processing_worker():
    """工作线程，从队列中获取录音数据并处理"""
    global is_running, audio_processing_queue
    while is_running:
        try:
            audio_data_np, rate, funasr_api_url, ollama_model, target_lang_code, ollama_api_url, osc_client_local = audio_processing_queue.get(timeout=1)
            save_and_send_audio_to_funasr(
                audio_data_np, 
                rate, 
                funasr_api_url, 
                ollama_model,
                target_lang_code,
                ollama_api_url,
                osc_client_local
            )
            audio_processing_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            log_message(f"音频处理工作线程发生错误: {e}")


def audio_callback2(indata, frames, callback_time, status):
    """音频输入流的回调函数"""
    global is_currently_recording, recording_frames
    global consecutive_silent_blocks, consecutive_speech_blocks
    global speech_trigger_blocks_count, silence_trigger_blocks_count, actual_sample_rate
    global audio_processing_queue, osc_client, is_running
    global current_funasr_url, current_ollama_url, current_ollama_model, current_target_lang

    if status:
        log_message(f"音频流状态信息: {status}")

    volume_norm = calculate_rms(indata[:, 0])

    if is_currently_recording:
        recording_frames.append(indata.copy())
        if volume_norm < VOLUME_THRESHOLD:
            consecutive_silent_blocks += 1
            if consecutive_silent_blocks >= silence_trigger_blocks_count:
                log_message(f"检测到静音 ({SILENCE_CONFIRM_S}s)，停止录音")
                is_currently_recording = False
                if recording_frames:
                    full_recording = np.concatenate(recording_frames)
                    audio_processing_queue.put((
                        full_recording, 
                        actual_sample_rate, 
                        current_funasr_url, 
                        current_ollama_model, 
                        current_target_lang,
                        current_ollama_url,
                        osc_client
                    ))
                recording_frames = []
                consecutive_speech_blocks = 0
        else:
            consecutive_silent_blocks = 0
    else: 
        if volume_norm >= VOLUME_THRESHOLD:
            consecutive_speech_blocks += 1
            if consecutive_speech_blocks >= speech_trigger_blocks_count:
                log_message(f"检测到语音 ({SPEECH_CONFIRM_S}s)，开始录音...")
                is_currently_recording = True
                recording_frames = [indata.copy()]
                consecutive_silent_blocks = 0
        else:
            consecutive_speech_blocks = 0


def audio_callback(indata, frames, callback_time, status):
    """音频输入流的回调函数"""
    global is_currently_recording, recording_frames
    global consecutive_silent_blocks, consecutive_speech_blocks
    global speech_trigger_blocks_count, silence_trigger_blocks_count, actual_sample_rate
    global audio_processing_queue, osc_client, is_running
    global current_funasr_url, current_ollama_url, current_ollama_model, current_target_lang
    global pre_record_buffer # 引入前置录音缓存

    # 动态计算 pre_record_buffer_max_blocks，如果尚未设置
    # 这假设 actual_sample_rate 和 frames 是稳定的
    # 注意：更健壮的做法是在启动音频流之前就计算好这个值
    # 或者确保 actual_sample_rate 和 frames 在流的生命周期内不变
    # global pre_record_buffer_max_blocks
    # if pre_record_buffer_max_blocks == 0 and actual_sample_rate > 0 and frames > 0:
    # pre_record_buffer_max_blocks = int(PRE_RECORD_SECONDS * actual_sample_rate / frames)
    # 如果 actual_sample_rate 或 frames 可能变化，这里的逻辑需要更复杂
    # 为简单起见，我们假设它能被正确计算并设置。
    # 为了这个示例，我们直接使用一个计算好的值，或者你需要在你的代码中正确设置它。
    # 假设 pre_record_buffer_max_blocks 已经根据 PRE_RECORD_SECONDS, actual_sample_rate, 和 frames (块大小) 计算好了。
    # 例如: pre_record_buffer_max_blocks = int(PRE_RECORD_SECONDS / (frames / actual_sample_rate))
    # 我们将在下面直接使用 PRE_RECORD_SECONDS 和采样率、帧数来管理缓存大小

    if status:
        log_message(f"音频流状态信息: {status}")

    current_audio_block = indata.copy() # 复制当前音频块以便使用

    # --- 管理前置录音缓存 ---
    # 无论是否正在录音，都将当前块添加到前置缓存
    pre_record_buffer.append(current_audio_block)
    # 保持前置缓存的大小不超过 PRE_RECORD_SECONDS 定义的长度
    # 计算当前 pre_record_buffer 中所有帧的总数
    current_buffered_frames = sum(len(block) for block in pre_record_buffer)
    max_buffered_frames = int(PRE_RECORD_SECONDS * actual_sample_rate)

    while current_buffered_frames > max_buffered_frames and pre_record_buffer:
        removed_block = pre_record_buffer.pop(0) # 从头部移除旧的块
        current_buffered_frames -= len(removed_block)
    # --- 结束管理前置录音缓存 ---

    volume_norm = calculate_rms(current_audio_block[:, 0])

    if is_currently_recording:
        recording_frames.append(current_audio_block) # 添加当前块到正式录音
        if volume_norm < VOLUME_THRESHOLD:
            consecutive_silent_blocks += 1
            # 你需要确保 silence_trigger_blocks_count 是基于块的数量，而不是秒
            # 例如: silence_trigger_blocks_count = int(SILENCE_CONFIRM_S / (frames / actual_sample_rate))
            if consecutive_silent_blocks >= silence_trigger_blocks_count:
                log_message(f"检测到静音 ({SILENCE_CONFIRM_S}s)，停止录音")
                is_currently_recording = False
                if recording_frames: # 确保 recording_frames 不是空的
                    full_recording = np.concatenate(recording_frames)
                    audio_processing_queue.put((
                        full_recording,
                        actual_sample_rate,
                        current_funasr_url,
                        current_ollama_model,
                        current_target_lang,
                        current_ollama_url,
                        osc_client
                    ))
                recording_frames = [] # 清空录音帧
                # pre_record_buffer 仍然在后台持续填充，不需要在这里清空
                consecutive_speech_blocks = 0 # 重置语音块计数
        else:
            consecutive_silent_blocks = 0 # 音量高于阈值，重置静音块计数
    else: # 当前未在录音
        if volume_norm >= VOLUME_THRESHOLD:
            consecutive_speech_blocks += 1
            # 你需要确保 speech_trigger_blocks_count 是基于块的数量
            # 例如: speech_trigger_blocks_count = int(SPEECH_CONFIRM_S / (frames / actual_sample_rate))
            if consecutive_speech_blocks >= speech_trigger_blocks_count:
                log_message(f"检测到语音 ({SPEECH_CONFIRM_S}s)，开始录音...")
                is_currently_recording = True
                # --- 开始录音时，将前置缓存的内容加入到 recording_frames ---
                # 我们需要获取 pre_record_buffer 中最后 PRE_RECORD_SECONDS 的数据。
                # 由于 pre_record_buffer 已经维护了大约 PRE_RECORD_SECONDS 的数据，
                # 并且最近的数据在尾部，我们可以直接使用它。
                # 我们需要确保不会重复添加已经因为 speech_trigger_blocks_count 而累积的块。

                # `consecutive_speech_blocks` 已经累积了一些块，这些块也存在于 `pre_record_buffer` 的尾部。
                # 我们需要小心，不要重复添加这些块。

                # 一个简单的方法是：
                # 1. 清空 recording_frames
                # 2. 将 pre_record_buffer 的内容（它包含了触发前的音频和触发时的音频）全部放入 recording_frames
                # 3. `current_audio_block` 已经是 `pre_record_buffer` 的最后一个元素，
                #    并且在 `is_currently_recording` 为 True 后的下一次迭代中，它会被 `recording_frames.append(current_audio_block)` 加入。

                # 改进的逻辑：当检测到语音并开始录音时，
                # `pre_record_buffer` 包含了触发点之前0.5秒的音频，
                # 以及导致触发的 `speech_trigger_blocks_count` 数量的音频块。
                # 我们希望录音从触发点前0.5秒开始。

                temp_recording_frames = []
                # 将 pre_record_buffer 中的所有帧连接起来
                if pre_record_buffer:
                    # 我们需要确保只取大约 PRE_RECORD_SECONDS 的数据
                    # 并且这些数据是在当前触发块之前的。
                    # `pre_record_buffer` 的管理逻辑确保它大致包含 PRE_RECORD_SECONDS 的数据。
                    # 当我们决定开始录音时，`current_audio_block` 是使 `consecutive_speech_blocks` 达到阈值的那个块。
                    # `pre_record_buffer` 此时的最后一个元素就是 `current_audio_block`。

                    # 让我们从 `pre_record_buffer` 构建初始的 `recording_frames`。
                    # `pre_record_buffer` 已经通过其维护逻辑限制了长度。
                    temp_recording_frames.extend(pre_record_buffer) # 将整个缓存作为前置音频

                # `speech_trigger_blocks_count` 块是导致我们开始录音的块。
                # 这些块也都在 `pre_record_buffer` 的末尾。
                # `current_audio_block` 是最新的块，并且已经被添加到 `pre_record_buffer` 中了。

                # 当 `is_currently_recording` 变为 True 时，
                # `recording_frames` 将从空的开始，然后 `pre_record_buffer` 的内容会被加进去。
                # 紧接着，在 `if is_currently_recording:` 分支中，`current_audio_block` 会被加入。

                # 清理并设置初始录音帧
                recording_frames = list(pre_record_buffer) # 将当前整个前置缓存作为录音的开始
                                                           # 注意：pre_record_buffer 的最后一个块是 current_audio_block
                                                           # 所以下一行不需要再添加 current_audio_block
                                                           # 因为在下一次回调的 if is_currently_recording: 分支中，新的 indata 会被添加

                # log_message(f"前置缓存包含 {len(pre_record_buffer)} 个块，总计约 {sum(len(b) for b in pre_record_buffer)/actual_sample_rate:.2f} 秒")
                # log_message(f"使用前置缓存开始录音，包含 {len(recording_frames)} 个块。")

                consecutive_silent_blocks = 0 # 重置静音块计数
        else:
            consecutive_speech_blocks = 0 # 音量低于阈值，重置语音块计数
            # 如果没有达到语音触发条件，我们不需要做特别处理，pre_record_buffer 会继续更新

    # 如果程序即将退出，确保清理
    if not is_running and is_currently_recording:
        # 这是一个简化的处理，实际应用中可能需要更复杂的逻辑来确保数据被处理
        log_message("程序正在停止，处理剩余录音...")
        is_currently_recording = False
        if recording_frames:
            full_recording = np.concatenate(recording_frames)
            audio_processing_queue.put((
                full_recording,
                actual_sample_rate,
                current_funasr_url,
                current_ollama_model,
                current_target_lang,
                current_ollama_url,
                osc_client
            ))
        recording_frames = []



def check_device_compatibility(device_id, target_sample_rate):
    """检查设备兼容性并确定实际采样率"""
    try:
        sd.check_input_settings(device=device_id, samplerate=target_sample_rate, channels=CHANNELS)
        log_message(f"设备 ID {device_id} 支持目标采样率 {target_sample_rate} Hz 和 {CHANNELS} 声道")
        return target_sample_rate
    except sd.PortAudioError:
        log_message(f"设备 ID {device_id} 不直接支持目标采样率 {target_sample_rate} Hz")
        try:
            device_info = sd.query_devices(device_id)
            default_sr_str = device_info.get('default_samplerate', '')
            if not default_sr_str:
                raise ValueError("设备未提供有效的默认采样率")
            default_sr = int(float(default_sr_str))
            if default_sr <= 0:
                raise ValueError(f"设备的默认采样率 ({default_sr}) 无效")

            log_message(f"尝试使用设备的默认采样率: {default_sr} Hz")
            sd.check_input_settings(device=device_id, samplerate=default_sr, channels=CHANNELS)
            log_message(f"将以设备的默认采样率 {default_sr} Hz 进行录制")
            if default_sr != target_sample_rate:
                log_message(f"警告: FunASR API 通常期望 {target_sample_rate} Hz。以 {default_sr} Hz 发送可能影响识别效果")
            return default_sr
        except Exception as e:
            log_message(f"检查设备默认采样率时发生错误: {e}")
            return None
    except Exception as e:
        log_message(f"检查设备能力时发生未知错误: {e}")
        return None


def start_listening(
    device_id, 
    volume_threshold, 
    speech_confirm, 
    silence_confirm,
    funasr_url, 
    ollama_url, 
    ollama_model, 
    target_lang,
    osc_ip, 
    osc_port
):
    """开始监听麦克风"""
    global is_running, worker_thread, osc_client
    global VOLUME_THRESHOLD, SPEECH_CONFIRM_S, SILENCE_CONFIRM_S
    global actual_sample_rate, block_size_frames, speech_trigger_blocks_count, silence_trigger_blocks_count
    global current_funasr_url, current_ollama_url, current_ollama_model, current_target_lang
    
    if is_running:
        log_message("已经在运行中，请先停止当前监听")
        return log_message(""), False
    
    try:
        # 更新配置参数
        VOLUME_THRESHOLD = float(volume_threshold)
        SPEECH_CONFIRM_S = float(speech_confirm)
        SILENCE_CONFIRM_S = float(silence_confirm)
        
        # 保存当前配置
        current_funasr_url = funasr_url
        current_ollama_url = ollama_url
        current_ollama_model = ollama_model
        current_target_lang = target_lang
        
        # 设置OSC客户端
        try:
            osc_client = udp_client.SimpleUDPClient(osc_ip, int(osc_port))
            log_message(f"OSC 客户端已初始化，将发送到 {osc_ip}:{osc_port}")
        except Exception as e:
            log_message(f"初始化 OSC 客户端失败: {e}")
            log_message("程序将继续运行，但不会发送 OSC 消息到 VRChat")
            osc_client = None
        
        # 检查设备兼容性
        actual_sample_rate = check_device_compatibility(int(device_id), TARGET_SAMPLE_RATE)
        if actual_sample_rate is None:
            log_message("设备兼容性检查失败，无法继续")
            return log_message(""), False
            
        # 计算音频块参数
        block_size_frames = int(actual_sample_rate * BLOCK_DURATION_S)
        speech_trigger_blocks_count = int(SPEECH_CONFIRM_S / BLOCK_DURATION_S)
        silence_trigger_blocks_count = int(SILENCE_CONFIRM_S / BLOCK_DURATION_S)
        
        log_message(f"实际使用麦克风采样率: {actual_sample_rate} Hz")
        log_message(f"音频块大小: {block_size_frames} 帧 ({BLOCK_DURATION_S*1000:.0f} ms)")
        log_message(f"音量阈值 (RMS): {VOLUME_THRESHOLD}")
        log_message(f"语音确认时长: {SPEECH_CONFIRM_S} 秒 ({speech_trigger_blocks_count} 个连续音频块)")
        log_message(f"静音确认时长: {SILENCE_CONFIRM_S} 秒 ({silence_trigger_blocks_count} 个连续音频块)")
        
        # 启动处理线程
        is_running = True
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
            log_message("\n程序被用户中断。正在退出...")
            print("\n程序被用户中断。正在退出...")
        except Exception as e:
            log_message("打开音频流或主循环时发生错误...")
            print(f"打开音频流或主循环时发生错误: {e}")
        finally:
            log_message("程序结束。")
            print("程序结束。")
            if audio_processing_queue and not audio_processing_queue.empty():
                print("等待剩余音频处理完成...")
                audio_processing_queue.join() 
            print("所有任务已处理完毕。")
 
        log_message("成功启动监听，正在监听麦克风...")
        return log_message(""), True
        
    except Exception as e:
        log_message(f"启动监听时发生错误: {e}")
        stop_listening()
        return log_message(""), False


def stop_listening():
    """停止监听麦克风"""
    global is_running, osc_client
    
    is_running = False
    
    if not audio_processing_queue.empty():
        log_message("等待剩余音频处理完成...")
        try:
            audio_processing_queue.join(timeout=5)
            log_message("所有音频处理任务已完成")
        except Exception:
            log_message("等待音频处理超时，可能有未完成的任务")
    
    osc_client = None
    log_message("已完全停止监听")
    return log_message("")


def refresh_ollama_models(ollama_url):
    """刷新Ollama模型列表"""
    models = get_ollama_models(ollama_url) 
    return gr.Dropdown(  choices=models , label="Ollama 模型") 


def create_ui():
    """创建Gradio UI界面"""
    with gr.Blocks(title="语音转文字+翻译+VRChat OSC") as app:
        gr.Markdown("# 语音转文字 + 翻译 + VRChat OSC集成")
        
        with gr.Tabs():
            with gr.TabItem("主界面"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 设备设置")
                        input_devices = get_microphones()
                        mic_dropdown = gr.Dropdown(
                            choices=input_devices,
                            label="麦克风设备",
                            value=input_devices[0][1] if input_devices else None
                        )
                        
                        with gr.Row():
                            volume_threshold = gr.Slider(
                                minimum=0.001, maximum=0.1, value=0.02, step=0.001,
                                label="音量阈值"
                            )
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                speech_confirm = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.3, step=0.1,
                                    label="语音确认时长(秒)"
                                )
                            with gr.Column(scale=1):
                                silence_confirm = gr.Slider(
                                    minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                                    label="静音确认时长(秒)"
                                )
                                
                        gr.Markdown("### 服务设置")
                        with gr.Row():
                            with gr.Column(scale=1):
                                funasr_url = gr.Textbox(
                                    label="FunASR API URL",
                                    value="http://localhost:5000/transcribe"
                                )
                            
                        with gr.Accordion("Ollama设置", open=True):
                            with gr.Row():
                                ollama_url = gr.Textbox(
                                    label="Ollama API URL",
                                    value="http://localhost:11434"
                                )
                                refresh_btn = gr.Button("刷新模型", variant="secondary")
                            
                            with gr.Row():
                                with gr.Column(scale=3):
                                    ollama_model_dropdown = gr.Dropdown(
                                        choices=[],
                                        label="Ollama 模型"
                                    )
                                with gr.Column(scale=1):
                                    target_lang_dropdown = gr.Dropdown(
                                        choices=[(v, k) for k, v in SUPPORTED_LANGUAGES.items()],
                                        label="目标语言",
                                        value="Chinese (中文)"
                                    )
                        
                        with gr.Accordion("OSC设置 (VRChat)", open=True):
                            with gr.Row():
                                with gr.Column(scale=3):
                                    osc_ip = gr.Textbox(
                                        label="VRChat IP",
                                        value="127.0.0.1"
                                    )
                                with gr.Column(scale=1):
                                    osc_port = gr.Number(
                                        label="VRChat 端口",
                                        value=9000,
                                        precision=0
                                    )
                                
                        with gr.Row():
                            start_btn = gr.Button("开始监听", variant="primary")
                            stop_btn = gr.Button("停止监听", variant="stop")
                            
                        status_indicator = gr.Checkbox(label="正在运行", value=False, interactive=False)
                    
                    with gr.Column(scale=1):
                        log_display = gr.Textbox(
                            label="日志",
                            value="",
                            lines=25,
                            max_lines=25,
                            autoscroll=True,
                            interactive=False
                        )
            
            with gr.TabItem("帮助"):
                gr.Markdown("""
                ## 使用说明
                
                此工具可以实现以下功能:
                1. 监听麦克风，检测到语音时开始录音，检测到静音时停止录音
                2. 将录制的音频发送到 FunASR 语音识别 API 进行转写
                3. 使用 Ollama 将转写的文本翻译成选定的目标语言
                4. 通过 OSC 将原文和翻译后的文本发送到 VRChat
                
                ### 必要准备
                - 需要已运行 FunASR 服务
                - 需要已运行 Ollama 服务且加载了相应模型
                - VRChat 需要开启 OSC 支持
                
                ### 参数说明
                - **音量阈值**: 判断是否有声音的阈值，值越低越敏感
                - **语音确认时长**: 声音持续超过阈值多少秒后开始录制
                - **静音确认时长**: 声音持续低于阈值多少秒后停止录制
                
                ### 常见问题
                1. 如果无法检测到麦克风输入，请检查麦克风设备选择是否正确
                2. 如果 FunASR 转写不准确，可能需要调整麦克风设置或重新配置 FunASR 服务
                3. 如果 Ollama 翻译效果不佳，可以尝试使用不同的模型
                """)
            
        # 事件处理
        refresh_btn.click(
            fn=refresh_ollama_models,
            inputs=[ollama_url],
            outputs=[ollama_model_dropdown]
        )
        
        start_btn.click(
            fn=start_listening,
            inputs=[
                mic_dropdown, volume_threshold, speech_confirm, silence_confirm,
                funasr_url, ollama_url, ollama_model_dropdown, target_lang_dropdown,
                osc_ip, osc_port
            ],
            outputs=[log_display, status_indicator]
        )
        
        stop_btn.click(
            fn=stop_listening,
            outputs=[log_display]
        )
        
        # 页面加载时自动刷新Ollama模型列表
        app.load(
            fn=lambda: gr.Dropdown(choices=get_ollama_models("http://localhost:11434"), label="Ollama 模型"),
            outputs=[ollama_model_dropdown]
        )
        
    return app


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", share=False)