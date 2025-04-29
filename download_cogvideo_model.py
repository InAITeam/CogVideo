from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="THUDM/CogVideoX-5b",
    local_dir="models/CogVideoX-5b",
    local_dir_use_symlinks=False
)
