from flask import Flask, render_template, request, jsonify
import openai
import os
from diffusers import CogVideoXPipeline
import torch
from datetime import datetime

app = Flask(__name__)

# 初始化 CogVideo model
pipe = CogVideoXPipeline.from_pretrained(
    "models/CogVideoX-5b",
    torch_dtype=torch.float16,
)
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()

# 確保輸出目錄存在
output_dir = 'static/videos'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def improve_prompt(prompt):
    """使用 ChatGPT 改善提示詞"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一個專業的 AI 影片生成提示詞專家。請改善用戶的提示詞，使其更加詳細和具體，以便生成更好的影片。保持原始意圖但加入更多細節。"},
                {"role": "user", "content": f"請改善這個提示詞，使其更適合 AI 影片生成：{prompt}"}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return str(e)

def generate_video(prompt):
    """生成影片"""
    try:
        # 生成唯一的檔案名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/video_{timestamp}.mp4"
        
        # 生成影片
        video_frames = pipe(
            prompt=prompt,
            height=384,
            width=384,
            num_frames=24,
            num_inference_steps=50,
            guidance_scale=6.5
        ).frames[0]
        
        # 儲存影片
        video_path = export_to_video(video_frames, output_path)
        return os.path.join('videos', os.path.basename(output_path))
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('service.html')

@app.route('/improve_prompt', methods=['POST'])
def improve_prompt_api():
    data = request.json
    prompt = data.get('prompt', '')
    improved_prompt = improve_prompt(prompt)
    return jsonify({'improved_prompt': improved_prompt})

@app.route('/generate', methods=['POST'])
def generate_api():
    data = request.json
    prompt = data.get('prompt', '')
    video_path = generate_video(prompt)
    return jsonify({'video_path': video_path})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
