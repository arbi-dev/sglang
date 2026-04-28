# GroupNorm SiLU PR Evidence

Artifacts for `codex/optimize-group-norm-silu`.

- `main/`: evidence from `origin/main` at `144038fba`
- `pr/`: evidence from PR commit `e844f696d`

The current 5s cat/dog video generations use HunyuanVideo with:

```text
prompt="Wide shot of two clearly different pets on a sunlit living room rug: a gray tabby kitten on the left and a small black-and-white puppy on the right. Both animals are fully visible at the same time, playing tug-of-war with a red rope toy, cute realistic home video, smooth natural motion, warm daylight, distinct cat and dog."
num_frames=121, width=960, height=544, num_inference_steps=12, seed=2026042901
SGLANG_USE_CUDA_HUNYUANVIDEO_GROUP_NORM_SILU=1
```

`ffprobe` reports H264, 960x544, 121 frames, duration 5.041667s for both
`main/hunyuan_cat_dog_5s_v2_main.mp4` and `pr/hunyuan_cat_dog_5s_v2_pr.mp4`.
