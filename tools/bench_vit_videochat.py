"""VIT performance + memory benchmark for VideoChatFlash."""
import argparse
import gc
import statistics
import time

import numpy as np
import openvino as ov
import openvino_genai as ov_genai


def read_rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def make_image(h=448, w=448):
    return ov.Tensor(np.linspace(0, 255, h * w * 3, dtype=np.uint8).reshape(h, w, 3))


def make_video(frames=10, h=224, w=224):
    return ov.Tensor(np.linspace(0, 255, frames * h * w * 3, dtype=np.uint8).reshape(frames, h, w, 3))


def bench(pipe, *, prompt, images=None, videos=None, warmup=2, iters=10):
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 1
    kwargs = {}
    if images:
        kwargs["images"] = images
    if videos:
        kwargs["videos"] = videos
    for _ in range(warmup):
        pipe.generate(prompt, generation_config=cfg, **kwargs)
    samples = []
    for _ in range(iters):
        r = pipe.generate(prompt, generation_config=cfg, **kwargs)
        m = r.perf_metrics.get_prepare_embeddings_duration()
        samples.append(m.mean)
    return samples


def fmt(samples):
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return (f"mean={mean:8.2f}ms  stdev={stdev:6.2f}ms  "
            f"min={min(samples):8.2f}ms  max={max(samples):8.2f}ms  n={len(samples)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/ywang2/models/intel/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tag", default="plan-b")
    ap.add_argument("--skip-image", action="store_true")
    args = ap.parse_args()

    print(f"[bench] tag={args.tag} device={args.device} iters={args.iters} warmup={args.warmup}")
    print(f"[bench] model={args.model}")
    gc.collect()
    mem_start = read_rss_mb()
    print(f"[mem] start RSS: {mem_start:.1f} MB")

    t0 = time.time()
    pipe = ov_genai.VLMPipeline(args.model, args.device)
    load_time = time.time() - t0
    gc.collect()
    mem_after_load = read_rss_mb()
    print(f"[bench] pipeline load: {load_time:.2f}s")
    print(f"[mem] after pipeline load: {mem_after_load:.1f} MB  (delta {mem_after_load - mem_start:+.1f} MB)")

    image = make_image()
    video = make_video()

    mem_after_image = mem_after_load
    if not args.skip_image:
        print("\n=== image path ===")
        try:
            img_samples = bench(pipe, prompt="describe this image", images=[image],
                                warmup=args.warmup, iters=args.iters)
            print(f"image: {fmt(img_samples)}")
            gc.collect()
            mem_after_image = read_rss_mb()
            print(f"[mem] after image bench: {mem_after_image:.1f} MB  (delta {mem_after_image - mem_after_load:+.1f} MB)")
        except Exception as e:
            print(f"image: SKIPPED ({type(e).__name__}: {e})")
    else:
        print("\n=== image path: SKIPPED by flag ===")

    print("\n=== video path ===")
    vid_samples = bench(pipe, prompt="describe this video", videos=[video],
                        warmup=args.warmup, iters=args.iters)
    print(f"video: {fmt(vid_samples)}")
    gc.collect()
    mem_after_video = read_rss_mb()
    print(f"[mem] after video bench: {mem_after_video:.1f} MB  (delta from prev {mem_after_video - mem_after_image:+.1f} MB)")

    print("\n=== SUMMARY ===")
    print(f"tag                : {args.tag}")
    print(f"mem after load     : {mem_after_load:.1f} MB")
    print(f"mem after image    : {mem_after_image:.1f} MB")
    print(f"mem after video    : {mem_after_video:.1f} MB")
    print(f"peak vs start delta: {mem_after_video - mem_start:+.1f} MB")


if __name__ == "__main__":
    main()
