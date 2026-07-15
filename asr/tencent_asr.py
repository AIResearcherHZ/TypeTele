import base64
import json
import os
import threading
import time
from queue import Empty, Queue
from collections import deque

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from tencentcloud.asr.v20190614 import asr_client, models
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from ui import console, kv_panel


class AsrServer:
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = self._get_default_config()

        self.cfg = cfg
        self.verbose = cfg["verbose"]

        self.secret_id = cfg["credentials"]["secret_id"]
        self.secret_key = cfg["credentials"]["secret_key"]

        self.CHANNELS = cfg["audio"]["channels"]
        self.RATE = cfg["audio"]["sample_rate"]
        self.CHUNK_DURATION = cfg["audio"]["chunk_duration"]

        self.SILENCE_THRESHOLD = cfg["vad"]["silence_threshold"]
        self.MIN_AUDIO_LENGTH = cfg["vad"]["min_audio_length"]
        self.MAX_SILENCE_DURATION = cfg["vad"]["max_silence_duration"]

        self.audio_queue = Queue()
        self._stop = threading.Event()
        self._stop.set()
        self.is_recording = False
        self.is_running = False
        self.current_recording = []
        self.voice_detected = False
        self.silence_start_time = 0
        self.voice_start_time = 0

        self.result_queue = Queue(maxsize=1)

        self.have_new_result = False
        self._result_lock = threading.Lock()

        self.record_thread = None
        self.process_thread = None

        self._init_client()

        if cfg.get("test_microphone", True):
            self._test_microphone_volume()

    def _get_default_config(self):
        return {
            "credentials": {
                "secret_id": "YOUR_SECRET_ID",
                "secret_key": "YOUR_SECRET_KEY",
            },
            "audio": {"channels": 1, "sample_rate": 16000, "chunk_duration": 0.1},
            "vad": {
                "silence_threshold": 500,
                "min_audio_length": 0.5,
                "max_silence_duration": 2.0,
            },
            "tencent": {
                "endpoint": "asr.tencentcloudapi.com",
                "region": "ap-guangzhou",
                "engine_service_type": "16k_zh",
            },
            "test_microphone": True,
        }

    def _test_microphone_volume(self) -> int:
        CHUNK = int(self.CHUNK_DURATION * self.RATE)
        CHANNELS = self.CHANNELS
        RATE = self.RATE
        TEST_DURATION = 5

        console.print("[dim]正在打开麦克风... 测试 5 秒[/]")

        volumes = deque(maxlen=50)

        console.print("[bold]麦克风音量测试[/]")
        console.print("[dim]请说话或保持安静，观察音量变化[/]")
        console.print("[dim]这有助于你设置合适的静音阈值[/]")

        def audio_callback(indata, frames, time_info, status):
            if status:
                console.print(f"音频流状态: {status}")

            rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
            volumes.append(rms)

            if volumes:
                avg_volume = np.mean(volumes)
                max_volume = np.max(volumes) if volumes else 0
                min_volume = np.min(volumes) if volumes else 0

                bar_length = int(rms / 0.01)
                bar = "█" * min(bar_length, 50)

                console.print(
                    f"当前音量: {rms:6.3f} |{bar:<50}| 平均: {avg_volume:6.3f} 最大: {max_volume:6.3f} 最小: {min_volume:6.3f}",
                    end="\r",
                    highlight=False,
                )

        try:
            with sd.InputStream(
                callback=audio_callback,
                channels=CHANNELS,
                samplerate=RATE,
                blocksize=CHUNK,
                dtype=np.float32,
            ):
                console.print("[green]麦克风就绪，开始采集音频...[/]")
                threading.Event().wait(TEST_DURATION)

        except sd.PortAudioError as e:
            console.print(f"[red]无法打开麦克风: {e}[/]")
            console.print("[red]请检查麦克风设备是否可用[/]")
            return 0

        console.print()
        console.print("[green]测试结束[/]")

        if volumes:
            all_volumes = list(volumes)
            avg_volume = np.mean(all_volumes)
            max_volume = np.max(all_volumes)
            min_volume = np.min(all_volumes)

            console.print(
                kv_panel(
                    "统计结果",
                    [
                        ("平均音量", f"{avg_volume:.3f}"),
                        ("最大音量", f"{max_volume:.3f}"),
                        ("最小音量", f"{min_volume:.3f}"),
                    ],
                )
            )

            suggested_threshold = avg_volume + (max_volume - avg_volume) * 0.3

            console.print(
                kv_panel(
                    "建议的静音阈值",
                    [
                        ("1. 安静环境", f"{min_volume * 1.5:.3f}"),
                        ("2. 普通环境", f"{suggested_threshold:.3f}"),
                        ("3. 嘈杂环境", f"{max_volume * 0.8:.3f}"),
                    ],
                )
            )

            choice = input("请选择一个阈值选项 (1, 2, 3): ").strip()

            if choice == "1":
                selected_threshold = min_volume * 1.5
            elif choice == "2":
                selected_threshold = suggested_threshold
            elif choice == "3":
                selected_threshold = max_volume * 0.8
            else:
                console.print("[yellow]选择无效，使用默认的普通环境阈值[/]")
                selected_threshold = suggested_threshold

            self.SILENCE_THRESHOLD = int(selected_threshold * 120000)
            console.print(f"[green]已选择阈值: {self.SILENCE_THRESHOLD:.3f}[/]")

        else:
            console.print("[red]没有采集到音频数据，请检查麦克风连接[/]")
            self.SILENCE_THRESHOLD = 500

    def _init_client(self):
        try:
            cred = credential.Credential(self.secret_id, self.secret_key)

            httpProfile = HttpProfile()
            httpProfile.endpoint = self.cfg["tencent"]["endpoint"]

            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile

            self.client = asr_client.AsrClient(
                cred, self.cfg["tencent"]["region"], clientProfile
            )

            console.print("[green]腾讯云语音识别客户端初始化成功[/]")
        except Exception as e:
            console.print(f"[red]腾讯云客户端初始化失败: {e}[/]")
            raise e

    def _calculate_volume(self, audio_data):
        try:
            if isinstance(audio_data, np.ndarray):
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            else:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            return rms
        except Exception as e:
            console.print(f"[red]计算音量出错: {e}[/]")
            return 0

    def _is_silence(self, audio_data):
        volume = self._calculate_volume(audio_data)
        return volume < self.SILENCE_THRESHOLD

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            console.print(f"[yellow]音频状态警告: {status}[/]")

        current_time = time.time()
        is_silent = self._is_silence(indata)

        if not is_silent:
            if not hasattr(self, "voice_detected") or not self.voice_detected:
                self.voice_detected = True
                self.voice_start_time = current_time
                self.silence_start_time = 0
                if self.verbose:
                    console.print("[green]检测到语音，开始录音...[/]")

            self.current_recording.append(indata.copy())
            self.silence_start_time = 0

        else:
            if hasattr(self, "voice_detected") and self.voice_detected:
                if self.silence_start_time == 0:
                    self.silence_start_time = current_time
                    if self.verbose:
                        console.print("[dim]进入静音...[/]")

                silence_duration = current_time - self.silence_start_time

                if silence_duration < self.MAX_SILENCE_DURATION:
                    self.current_recording.append(indata.copy())
                else:
                    if self.verbose:
                        console.print(
                            f"[dim]静音超过 {self.MAX_SILENCE_DURATION} 秒，停止录音[/]"
                        )

                    if self.current_recording:
                        audio_data = np.concatenate(self.current_recording, axis=0)
                        total_duration = len(audio_data) / self.RATE

                        if total_duration >= self.MIN_AUDIO_LENGTH:
                            if self.verbose:
                                console.print(
                                    f"[green]录音结束[/]，时长 {total_duration:.2f} 秒，正在发送识别..."
                                )
                            self.audio_queue.put(audio_data)
                        else:
                            if self.verbose:
                                console.print(
                                    f"[yellow]录音太短 ({total_duration:.2f} 秒)，已丢弃[/]"
                                )

                    self.current_recording = []
                    self.voice_detected = False
                    self.silence_start_time = 0
                    if self.verbose:
                        console.print("[dim]等待语音输入...[/]")

    def _record_audio(self):
        if self.verbose:
            console.print("[dim]开始智能录音检测...[/]")
            console.print("[dim]等待语音输入...[/]")

        self.voice_detected = False
        self.silence_start_time = 0
        self.current_recording = []

        chunk_frames = int(self.RATE * self.CHUNK_DURATION)

        try:
            with sd.InputStream(
                samplerate=self.RATE,
                channels=self.CHANNELS,
                dtype="int16",
                blocksize=chunk_frames,
                callback=self.audio_callback,
            ):
                while self.is_recording:
                    if self._stop.wait(0.1):
                        break
        except Exception as e:
            console.print(f"[red]录音出错: {e}[/]")

        console.print("[dim]录音线程已结束[/]")

    def _save_wav_file(self, audio_data, filename):
        try:
            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype != np.int16:
                    audio_data = (audio_data * 32767).astype(np.int16)
                if audio_data.ndim > 1:
                    audio_data = audio_data.flatten()
            else:
                audio_data = np.frombuffer(audio_data, dtype=np.int16)

            wavfile.write(filename, self.RATE, audio_data)
        except Exception as e:
            console.print(f"[red]保存 WAV 文件失败: {e}[/]")

    def _recognize_audio_data(self, audio_data):
        try:
            temp_filename = "temp_audio.wav"
            self._save_wav_file(audio_data, temp_filename)

            with open(temp_filename, "rb") as f:
                wav_data = f.read()
            audio_base64 = base64.b64encode(wav_data).decode("utf-8")

            req = models.SentenceRecognitionRequest()
            params = {
                "EngSerViceType": self.cfg["tencent"]["engine_service_type"],
                "SourceType": 1,
                "VoiceFormat": "wav",
                "UsrAudioKey": f"realtime-{int(time.time())}",
                "Data": audio_base64,
                "DataLen": len(wav_data),
            }
            req.from_json_string(json.dumps(params))

            resp = self.client.SentenceRecognition(req)

            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            return resp.Result if resp.Result else ""

        except TencentCloudSDKException as err:
            console.print(f"[red]识别请求失败: {err}[/]")
            return ""
        except Exception as e:
            console.print(f"[red]识别过程出错: {e}[/]")
            return ""

    def _process_audio(self):
        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                if self.verbose:
                    console.print("[dim]正在调用腾讯云接口识别...[/]")
                result = self._recognize_audio_data(audio_data)
                if result.strip():
                    if self.verbose:
                        console.print(f"[bold green]识别结果:[/] {result}")

                    with self._result_lock:
                        if not self.result_queue.empty():
                            try:
                                self.result_queue.get_nowait()
                            except Empty:
                                pass

                        self.result_queue.put(result)
                        self.have_new_result = True
                else:
                    if self.verbose:
                        console.print("[yellow]接口返回结果为空[/]")
            except Exception as e:
                if self.is_running:
                    console.print(f"[red]处理音频出错: {e}[/]")

        console.print("[dim]音频处理线程已结束[/]")

    def set_silence_threshold(self, threshold):
        self.SILENCE_THRESHOLD = threshold
        console.print(f"[green]静音阈值已设为: {threshold}[/]")

    def set_silence_duration(self, duration):
        self.MAX_SILENCE_DURATION = duration
        console.print(f"[green]最大静音时长已设为: {duration} 秒[/]")

    def start(self):
        if self.is_running:
            console.print("[yellow]语音识别已经在运行了！[/]")
            return False

        console.print("[bold]智能连续语音识别系统[/]")
        console.print("[dim]已启用智能语音分段检测[/]")
        console.print(f"静音阈值: {self.SILENCE_THRESHOLD}")
        console.print(f"最大静音间隔: {self.MAX_SILENCE_DURATION} 秒")
        console.print(f"最短音频长度: {self.MIN_AUDIO_LENGTH} 秒")
        console.print("[dim]连续录音模式: 有声开始，静音结束[/]")
        console.print("[dim]避免句子被切碎，保持语句完整[/]")

        self._stop.clear()
        self.is_recording = True
        self.is_running = True

        self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.record_thread.start()

        self.process_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.process_thread.start()

        console.print("[green]实时语音识别已启动！[/]")
        return True

    def stop(self):
        console.print("[dim]正在停止录音和处理线程...[/]")
        self._stop.set()
        self.is_recording = False
        self.is_running = False

        if self.record_thread and self.record_thread.is_alive():
            console.print("[dim]等待录音线程结束...[/]")
            self.record_thread.join(timeout=1.0)

        if self.process_thread and self.process_thread.is_alive():
            console.print("[dim]等待处理线程结束...[/]")
            self.process_thread.join(timeout=1.0)

        console.print("[green]实时语音识别已停止[/]")

    def get(self):
        try:
            with self._result_lock:
                if self.have_new_result:
                    result = self.result_queue.get_nowait()
                    self.have_new_result = False
                    return result
                return None
        except Empty:
            return None

    def has_new_result(self):
        with self._result_lock:
            return self.have_new_result

    def is_running_status(self):
        return self.is_running

    def __del__(self):
        self.stop()


def main():
    cfg = {
        "verbose": True,
        "credentials": {"secret_id": "YOUR_SECRET_ID", "secret_key": "YOUR_SECRET_KEY"},
        "audio": {"channels": 1, "sample_rate": 16000, "chunk_duration": 0.1},
        "vad": {
            "silence_threshold": 500,
            "min_audio_length": 0.5,
            "max_silence_duration": 2.0,
        },
        "tencent": {
            "endpoint": "asr.tencentcloudapi.com",
            "region": "ap-guangzhou",
            "engine_service_type": "16k_zh",
        },
    }

    asr = None

    try:
        asr = AsrServer(cfg)

        asr.start()

        console.print()
        console.print("[bold]实时语音识别运行中[/]")
        console.print("[dim]按 Ctrl+C 退出[/]")

        idle = threading.Event()
        while not idle.wait(0.1):
            if asr.has_new_result():
                result = asr.get()
                if result:
                    console.print(f"[bold green]语音识别:[/] {result}")

    except KeyboardInterrupt:
        console.print("\n[yellow]程序被用户中断[/]")
    except Exception as e:
        console.print(f"[red]程序运行出错: {e}[/]")
    finally:
        console.print("[dim]正在清理资源...[/]")
        if asr is not None:
            try:
                asr.stop()
            except Exception as e:
                console.print(f"[red]清理语音识别资源出错: {e}[/]")

        console.print("[green]程序已完全退出[/]")


if __name__ == "__main__":
    main()
