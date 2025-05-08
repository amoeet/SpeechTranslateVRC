
# SpeechTranslateVRC

#### 介绍
VRChat 语音翻译并显示在头顶 
![72fcf9d23b914fc1f5f89dc8db944b86](https://github.com/user-attachments/assets/9ef5e060-d453-469b-b2f0-593094957c80)




#### 软件架构
使用到Funasr、ollama、osc、实现简单的麦克风输入经过翻译并显示在VRChat里


#### 安装教程

1.  python环境基本等同于[FunASR ](https://github.com/modelscope/FunASR)
2.  下载[SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/main)并将SenseVoiceSmall文件夹放置在 开启asr的API0.py 同一层级
（如需使用其他模型可以修改 开启asr的API0.py 中的模型路径）

#### 使用说明

0.  打开VRChat
1.  运行：开启asr的API0.py 
2.  运行：录音+调用API+ollama翻译+osc显示到vrchat+webui3.py
3.  在webui里选择麦克风、ollama模型和翻译语言，点击 开始监听

![98238cb2b5312c27602b21d691bddfb4](https://github.com/user-attachments/assets/6c63c6e0-55d2-40cb-88f4-6f23b43ccc8d)


参考及引用
https://github.com/modelscope/FunASR （用到其中的语音识别相关内容）
https://github.com/newkincode/VRChatTTS (了解了osc通讯相关内容)


