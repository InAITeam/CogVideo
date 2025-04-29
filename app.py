from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import io
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import os
import datetime

app = Flask(__name__)

# 確保輸出目錄存在
output_dir = 'resources/videos'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 初始化模型
print("載入模型中...")
pipe = CogVideoXPipeline.from_pretrained(
    "models/CogVideoX-5b",
    torch_dtype=torch.float16,
)

# 啟用記憶體優化
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()

# 儲存參數歷史
history_params = []

# 當前參數
last_params = {
    'prompt': '',
    'num_inference_steps': 50,
    'guidance_scale': 6.5,
    'height': 384,
    'width': 384,
    'num_frames': 24,
    'negative_prompt': ''
}

@app.route('/')
def index():
    # 獲取所有生成的視頻並按修改時間排序
    videos = []
    if os.path.exists(output_dir):
        # 獲取所有 mp4 文件並添加修改時間
        videos = [(f, os.path.getmtime(os.path.join(output_dir, f))) 
                 for f in os.listdir(output_dir) if f.endswith('.mp4')]
        # 按修改時間排序，最新的在前面
        videos.sort(key=lambda x: x[1], reverse=True)
        # 只保留文件名
        videos = [v[0] for v in videos]
        # 只取前5個視頻
        videos = videos[:5]
    return render_template('index.html', videos=videos, last_params=last_params, history_params=history_params)

@app.route('/generate', methods=['POST'])
def generate():
    # 獲取表單參數
    prompt = request.form.get('prompt', '')
    num_inference_steps = int(request.form.get('num_inference_steps', 50))
    guidance_scale = float(request.form.get('guidance_scale', 6.5))
    height = int(request.form.get('height', 384))
    width = int(request.form.get('width', 384))
    num_frames = int(request.form.get('num_frames', 24))
    
    # 更新參數
    global last_params, history_params
    current_params = {
        'prompt': prompt,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'height': height,
        'width': width,
        'num_frames': num_frames,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    last_params = current_params.copy()
    history_params.insert(0, current_params)
    
    # 只保留最近10個歷史記錄
    if len(history_params) > 10:
        history_params.pop()

    # 生成視頻
    print("開始生成影片...")
    video = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_frames=num_frames  # 使用表單中選擇的幀數
    ).frames[0]

    # 產生時間戳記作為檔名
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{timestamp}.mp4')

    # 儲存視頻
    print("儲存影片...")
    export_to_video(video, output_file, fps=4)  # 固定為 4fps 來產生 6 秒的視頻
    print(f"完成！影片存到 {output_file}")

    # 獲取更新後的視頻列表並按修改時間排序
    videos = []
    if os.path.exists(output_dir):
        videos = [(f, os.path.getmtime(os.path.join(output_dir, f))) 
                 for f in os.listdir(output_dir) if f.endswith('.mp4')]
        videos.sort(key=lambda x: x[1], reverse=True)
        videos = [v[0] for v in videos]
        # 只取前5個視頻
        videos = videos[:5]

    return render_template('index.html', videos=videos, last_video=f'{timestamp}.mp4', last_params=last_params)

@app.route('/generate_from_image', methods=['POST'])
def generate_from_image():
    # 獲取上傳的圖片
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return 'No selected image', 400

    # 讀取圖片
    image = Image.open(io.BytesIO(image_file.read()))
    
    # 獲取其他參數
    image_prompt = request.form.get('image_prompt', '')
    num_frames = int(request.form.get('num_frames', 24))
    
    # 轉換圖片為 RGB 模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 調整圖片大小為 384x384
    image = image.resize((384, 384))
    
    # 生成時間戳
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 生成視頻
    print("開始生成影片...")
    # 獲取所有參數
    num_inference_steps = int(request.form.get('num_inference_steps', 50))
    guidance_scale = float(request.form.get('guidance_scale', 7.5))
    negative_prompt = request.form.get('negative_prompt', '')
    
    # 更新參數歷史
    global last_params, history_params
    current_params = {
        'prompt': image_prompt,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'height': 384,
        'width': 384,
        'num_frames': num_frames,
        'negative_prompt': negative_prompt,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': 'image-to-video'
    }
    
    last_params = current_params.copy()
    history_params.insert(0, current_params)
    
    # 只保留最近10個歷史記錄
    if len(history_params) > 10:
        history_params.pop()
    
    video = pipe(
        prompt=image_prompt,  # 可以是空字符串
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=384,
        width=384,
        num_frames=num_frames
    ).frames[0]
    
    # 儲存視頻
    output_file = os.path.join(output_dir, f'{timestamp}.mp4')
    print("儲存影片...")
    export_to_video(video, output_file, fps=4)
    print(f"完成！影片存到 {output_file}")
    
    # 獲取更新後的視頻列表並按修改時間排序
    videos = []
    if os.path.exists(output_dir):
        videos = [(f, os.path.getmtime(os.path.join(output_dir, f))) 
                 for f in os.listdir(output_dir) if f.endswith('.mp4')]
        videos.sort(key=lambda x: x[1], reverse=True)
        videos = [v[0] for v in videos]
        videos = videos[:5]
    
    return render_template('index.html', videos=videos, last_video=f'{timestamp}.mp4', last_params=last_params)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(output_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
