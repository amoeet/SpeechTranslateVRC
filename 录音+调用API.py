import sounddevice as sd
import numpy as np
import requests
import wave
import io
import time
import threading
import queue

# --- 配置参数 ---
API_URL = "http://localhost:5000/transcribe"  # FunASR API 服务地址
# TARGET_SAMPLE_RATE 仍然用于尝试以该速率打开麦克风
# 但如果失败，将使用设备的默认采样率，并且该采样率将用于发送
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

def calculate_rms(data):
    """计算音频数据的均方根 (RMS)。"""
    return np.sqrt(np.mean(data**2))

def save_and_send_audio(audio_data_np, recorded_sample_rate, api_url):
    """
    将音频数据保存为WAV格式，以其录制采样率发送到API，并打印结果。
    注意：API可能期望特定的采样率（如16000Hz）。以不同采样率发送可能导致识别问题。
    """
    print("\n正在处理录音数据...")
    if audio_data_np.size == 0:
        print("没有录到音频数据。")
        return

    final_sample_rate_for_api = recorded_sample_rate # 直接使用录制采样率

    try:
        # 将 float32 (-1.0 到 1.0) 转换为 int16 (PCM)
        audio_data_int16 = (audio_data_np * 32767).astype(np.int16)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16 需要 2 字节
            wf.setframerate(int(final_sample_rate_for_api)) # 使用录制采样率
            wf.writeframes(audio_data_int16.tobytes())
        wav_buffer.seek(0)

        files = {'audio_file': ('recording.wav', wav_buffer, 'audio/wav')}
        duration_sec = len(audio_data_int16) / final_sample_rate_for_api
        print(f"发送 {duration_sec:.2f} 秒的音频 (采样率 {final_sample_rate_for_api}Hz) 到 API: {api_url}")
        
        response = requests.post(api_url, files=files, timeout=30)
        response.raise_for_status()

        result = response.json()
        if "transcription" in result and result["transcription"]:
            print(f"识别结果: {result['transcription']}")
        elif "error" in result:
            print(f"API 返回错误: {result['error']}")
        else:
            print(f"API 未返回有效识别结果或错误信息: {result}")

    except requests.exceptions.Timeout:
        print("错误: API 请求超时。")
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到 API ({api_url})。请确保服务正在运行。")
    except requests.exceptions.RequestException as e:
        print(f"发送到 API 时出错: {e}")
    except Exception as e:
        print(f"处理录音或 API 响应时发生意外错误: {e}")
    finally:
        if 'wav_buffer' in locals() and wav_buffer:
            wav_buffer.close()


def audio_processing_worker():
    """工作线程，从队列中获取录音数据并处理。"""
    while True:
        try:
            audio_data_np, rate = audio_processing_queue.get(timeout=1)
            save_and_send_audio(audio_data_np, rate, API_URL)
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
                    # 将录制的音频（以 actual_sample_rate 录制）和其采样率放入队列
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

    selected_device_id = list_microphones()
    if selected_device_id is None:
        return

    print(f"\n您选择了麦克风设备 ID: {selected_device_id}")

    try:
        # 1. 尝试使用目标采样率 (e.g., 16000 Hz)
        sd.check_input_settings(device=selected_device_id, samplerate=TARGET_SAMPLE_RATE, channels=CHANNELS)
        actual_sample_rate = TARGET_SAMPLE_RATE
        print(f"设备 ID {selected_device_id} 支持目标采样率 {TARGET_SAMPLE_RATE} Hz 和 {CHANNELS} 声道。")
        print(f"将以 {actual_sample_rate} Hz 进行录制和发送。")
    except sd.PortAudioError:
        print(f"设备 ID {selected_device_id} 不直接支持目标采样率 {TARGET_SAMPLE_RATE} Hz。")
        try:
            # 2. 尝试使用设备的默认采样率
            device_info = sd.query_devices(selected_device_id)
            default_sr_str = device_info.get('default_samplerate', '')
            if not default_sr_str:
                 raise ValueError("设备未提供有效的默认采样率。")
            default_sr = int(float(default_sr_str))
            
            if default_sr <= 0:
                raise ValueError(f"设备的默认采样率 ({default_sr}) 无效。")

            print(f"尝试使用设备的默认采样率: {default_sr} Hz。")
            sd.check_input_settings(device=selected_device_id, samplerate=default_sr, channels=CHANNELS)
            actual_sample_rate = default_sr
            print(f"将以设备的默认采样率 {actual_sample_rate} Hz 进行录制和发送。")
            if actual_sample_rate != TARGET_SAMPLE_RATE:
                 print(f"警告: FunASR API 通常期望 {TARGET_SAMPLE_RATE} Hz。以 {actual_sample_rate} Hz 发送可能导致识别效果不佳。")


        except (sd.PortAudioError, ValueError, TypeError) as e_default:
            print(f"错误: 设备 ID {selected_device_id} 也不支持其默认采样率，或默认采样率无效/无法使用。")
            print(f"错误信息: {e_default}")
            print("请尝试选择其他麦克风设备，或检查您的麦克风驱动程序和操作系统音频设置。")
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
