"""Reproducible CPU feasibility experiment; no obstacle ground truth is assumed."""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'edge/teed'))
from run_teed import load_model, prepare_tensor, infer, clear_fire360_osd
from contours import select_contours, thin, overlay, edge_disagreement

DEVICE = torch.device('cpu')
SIZE = (320, 240)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames or fps <= 0:
        raise RuntimeError(f'Empty or invalid video: {path}')
    return frames, fps


def tensor_for(frame, structural=False, size=SIZE):
    if structural:
        frame = cv2.bilateralFilter(frame, 5, 25, 3)
    return prepare_tensor(frame, *size, DEVICE, 0. if structural else 2.)[1]


def predict(model, frame, structural=False, size=SIZE, timed=False):
    tensor = tensor_for(frame, structural, size)
    started = time.perf_counter()
    with torch.no_grad():
        p = model(tensor)[-1].sigmoid()[0, 0].numpy()
    ms = (time.perf_counter()-started)*1000
    return (p, ms) if timed else p


def adapt(teacher, frames, out, train_name='sample_3.MP4', osd=False, start_index=0):
    if not 0 <= start_index < len(frames):
        raise ValueError('Training clip must contain frames after start_index')
    student = copy.deepcopy(teacher)
    for name, param in student.named_parameters():
        param.requires_grad_(name.startswith('block_cat.'))
    indices = np.linspace(start_index, len(frames) - 1, 32, dtype=int).tolist()
    features, targets, weights = [], [], []
    start = time.perf_counter()
    for i in indices:
        with torch.no_grad():
            outputs = teacher(tensor_for(frames[i], True))
            p = outputs[-1].sigmoid()
            coarse = torch.from_numpy(cv2.resize(
                predict(teacher, frames[i], True, (160, 120)), SIZE))[None, None]
            # TEED's final smish has a nonzero sigmoid floor (~.438).
            # Suppress fine-scale-only responses without inventing class labels.
            support = ((coarse - .44) / .4).clamp(0, 1)
            scale = .35 + .65 * support
            scale = torch.where(p > .93, scale.clamp_min(.75), scale)
            target = .44 + (p - .44).clamp_min(0) * scale
            features.append(torch.cat(outputs[:3], 1).detach())
            targets.append(target.detach())
            weight = torch.where(p > .6, 5., 1.)
            if osd:
                valid = clear_fire360_osd(np.ones((240,320),np.float32))
                weight *= torch.from_numpy(valid)[None,None]
            weights.append(weight)
    params = [p for p in student.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=1e-4)

    def loss_at(i):
        pred = student.block_cat(features[i]).sigmoid()
        return (F.binary_cross_entropy(pred, targets[i], reduction='none') * weights[i]).mean()

    with torch.no_grad():
        initial = float(torch.stack([loss_at(i) for i in range(32)]).mean())
    epochs = []
    for epoch in range(8):
        losses = []
        for i in torch.randperm(32).tolist():
            optim.zero_grad()
            loss = loss_at(i)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            optim.step()
            losses.append(float(loss.detach()))
        epochs.append(float(np.mean(losses)))
        print(f'adapt epoch {epoch + 1}/8: {epochs[-1]:.6f}', flush=True)
    with torch.no_grad():
        final = float(torch.stack([loss_at(i) for i in range(32)]).mean())
    checkpoint = out / 'teed_fusion_adapted.pth'
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save(student.state_dict(), checkpoint)
    changed = [k for k,v in student.state_dict().items() if not torch.equal(v, teacher.state_dict()[k])]
    record = dict(seed=20260906, train_video=train_name, train_indices=indices,
                  steps=256, epochs=epochs, lr=1e-4, initial_loss=initial, final_loss=final,
                  loss='weighted soft binary cross entropy to multiscale pseudo targets',
                  changed_tensors=changed, trainable_parameters=sum(p.numel() for p in params),
                  checkpoint_sha256=digest(checkpoint), seconds=time.perf_counter()-start,
                  caveat='No human obstacle labels; loss reduction is not evidence of obstacle recall.')
    (out / 'training.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return student.eval(), record


def load_pidinet():
    upstream = HERE / 'third_party/pidinet'
    sys.path.insert(0, str(upstream))
    from models import pidinet
    model = pidinet(SimpleNamespace(config='carv4', sa=True, dil=True))
    path = upstream / 'trained_models/table5_pidinet.pth'
    state = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()}, strict=True)
    return model.eval(), path


def pidi_predict(model, frame, timed=False):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    rgb = (rgb - np.array([.485,.456,.406], np.float32)) / np.array([.229,.224,.225], np.float32)
    x = torch.from_numpy(rgb.transpose(2,0,1).copy())[None]
    started = time.perf_counter()
    with torch.no_grad():
        p = model(x)[-1][0,0].numpy()
    ms = (time.perf_counter()-started)*1000
    return (p, ms) if timed else p


def write_video(path, frames, fps):
    # Encode H.264 directly, no second lossy intermediate encode.
    h,w = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(str(path), (w,h), fps=fps,
        codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
        macro_block_size=1, quality=8, output_params=['-movflags','+faststart'])
    writer.send(None)
    try:
        for frame in frames:
            writer.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    decoded, actual_fps = read_video(path)
    if len(decoded) != len(frames) or abs(actual_fps-fps) > .02:
        raise RuntimeError(f'Output validation failed: {path}')
    return dict(frames=len(decoded), fps=actual_fps, bytes=path.stat().st_size, sha256=digest(path))


def evaluate(tag, frames, fps, teacher, student, pidi, out, osd=False, videos=True):
    names = ['baseline', 'thin_only', 'structural', 'adapted', 'pidinet']
    rendered = {n:[] for n in names}
    masks = {n:[] for n in names}
    timing = {n:[] for n in names}
    model_timing = {n:[] for n in names}
    config = SimpleNamespace(width=320,height=240,contrast_clip_limit=2.,
                             threshold=.75,edge_width=3,background_scale=.24)
    for index, frame in enumerate(frames):
        begin = time.perf_counter()
        base, base_mask, base_ms, _ = infer(teacher, frame, config, DEVICE, osd)
        model_timing['baseline'].append(base_ms)
        timing['baseline'].append((time.perf_counter()-begin)*1000)
        rendered['baseline'].append(base)
        masks['baseline'].append(base_mask > 0)
        for name in names[1:]:
            begin = time.perf_counter()
            if name == 'pidinet':
                p, model_ms = pidi_predict(pidi, frame, timed=True)
                mask = thin(select_contours(p, .2, .4))
            else:
                p, model_ms = predict(student if name == 'adapted' else teacher, frame,
                                      name != 'thin_only', timed=True)
                mask = thin((p >= .75).astype(np.uint8) if name == 'thin_only'
                            else select_contours(p, .60, .85))
            if osd:
                mask = clear_fire360_osd(mask)
            result = overlay(frame, mask)
            timing[name].append((time.perf_counter()-begin)*1000)
            model_timing[name].append(model_ms)
            rendered[name].append(result)
            masks[name].append(mask)
        if index % 60 == 0:
            print(f'{tag}: {index}/{len(frames)}', flush=True)
    result = {}
    for name in names:
        warm = timing[name][min(10,len(frames)-1):]
        metrics = dict(edge_occupancy_percent=100*float(np.mean(masks[name])),
            timing_samples=len(warm),
            cpu_pipeline_ms_median=float(np.median(warm)),
            cpu_pipeline_ms_p95=float(np.percentile(warm,95)) if len(warm)>1 else None,
            cpu_model_ms_median=float(np.median(model_timing[name][min(10,len(frames)-1):])))
        if videos:
            metrics['video'] = write_video(out/f'{tag}_{name}.mp4', rendered[name], fps)
        result[name] = metrics
    np.savez_compressed(out/f'{tag}_masks.npz', **{n:np.asarray(masks[n],dtype=bool) for n in names})
    # Backward optical flow aligns previous mask to current frame. No smoothing
    # is applied to displayed results, so a newly appearing obstacle is not delayed.
    if len(frames) > 1:
        flicker = {n:[] for n in names}
        coverage = []
        yy,xx = np.mgrid[:240,:320].astype(np.float32)
        for i in range(1,len(frames)):
            current = cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY)
            previous = cv2.cvtColor(frames[i-1],cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(current,previous,None,.5,3,15,3,5,1.2,0)
            mx,my = xx+flow[:,:,0], yy+flow[:,:,1]
            warped_gray = cv2.remap(previous,mx,my,cv2.INTER_LINEAR)
            valid = (mx>=0)&(mx<319)&(my>=0)&(my<239)&(np.abs(current.astype(float)-warped_gray)<20)
            coverage.append(float(valid.mean()))
            for n in names:
                warped = cv2.remap(masks[n][i-1].astype(np.uint8),mx,my,cv2.INTER_NEAREST)>0
                score = edge_disagreement(warped, masks[n][i], valid)
                if score is not None:
                    flicker[n].append(score)
        for n in names:
            result[n]['motion_aligned_edge_disagreement'] = float(np.mean(flicker[n])) if flicker[n] else None
            result[n]['temporal_usable_pairs'] = len(flicker[n])
            result[n]['flow_valid_pixel_fraction'] = float(np.mean(coverage))
    for i in sorted(set([0, len(frames)//2, len(frames)-1])):
        # Comparison image has no labels baked into scene pixels; report defines order.
        panels = [frames[i]]+[rendered[n][i] for n in names]
        sheet = cv2.vconcat([cv2.hconcat(panels[:3]),cv2.hconcat(panels[3:])])
        cv2.imwrite(str(out/f'{tag}_frame_{i:04d}.png'),sheet)
    if videos:
        write_video(out/f'{tag}_original.mp4',frames,fps)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-checkpoint',action='store_true')
    parser.add_argument('--suite',choices=['rgb','thermal'],default='thermal')
    args = parser.parse_args()
    torch.manual_seed(20260906)
    np.random.seed(20260906)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    out = HERE/'outputs'/args.suite
    out.mkdir(parents=True,exist_ok=True)
    teacher = load_model(str(ROOT/'edge/teed/models/5_model.pth'),DEVICE)
    thermal = args.suite == 'thermal'
    train_name = 'IFSI Video 8 (2).mp4' if thermal else 'sample_3.MP4'
    test_name = 'IFSI Video 4 (2).mp4' if thermal else 'sample_4.MP4'
    train, train_fps = read_video(HERE/'data'/train_name)
    test, test_fps = read_video(HERE/'data'/test_name)
    if args.reuse_checkpoint:
        training = json.loads((out/'training.json').read_text())
        if digest(out/'teed_fusion_adapted.pth') != training['checkpoint_sha256']:
            raise RuntimeError('Adapted checkpoint hash differs from training record')
        student = load_model(str(out/'teed_fusion_adapted.pth'),DEVICE)
    else:
        student,training = adapt(teacher,train,out,train_name,thermal, int(4*train_fps) if thermal else 0)
    pidi,pidi_path = load_pidinet()
    report = dict(environment=dict(python=sys.version,torch=torch.__version__,opencv=cv2.__version__,
        platform=platform.platform(),processor=platform.processor(),threads=4,cuda=False),
        parameters=dict(teed=sum(p.numel() for p in teacher.parameters()),pidinet=sum(p.numel() for p in pidi.parameters())),
        checkpoint_bytes=dict(teed=(ROOT/'edge/teed/models/5_model.pth').stat().st_size,pidinet=pidi_path.stat().st_size),
        checkpoint_sha256=dict(teed=digest(ROOT/'edge/teed/models/5_model.pth'),pidinet=digest(pidi_path)),
        pidinet_revision=subprocess.check_output(['git','-C',str(HERE/'third_party/pidinet'),'rev-parse','HEAD'],text=True).strip(),
        suite=args.suite,
        sources={name:dict(sha256=digest(HERE/'data'/name),
            url=f'https://uofi.app.box.com/v/fire360dataset/file/{fid}')
            for name,fid in ([(train_name,'1864227390988'),(test_name,'1864227378988')] if thermal else
                            [(train_name,'1864283301443'),(test_name,'1864286536357')])},
        training=training,results={})
    report['results']['heldout'] = evaluate('heldout',test,test_fps,teacher,student,pidi,out,osd=thermal)
    report['results']['training'] = evaluate('training',train,train_fps,teacher,student,pidi,out,osd=thermal)
    for path in sorted((ROOT/'edge/teed/samples').glob('*.jpg')):
        frame = cv2.resize(cv2.imread(str(path)),SIZE)
        report['results'][path.stem] = evaluate(path.stem,[frame],1,teacher,student,pidi,out,
                                               osd='ifsi' in path.stem,videos=False)
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FINISHED:',out,flush=True)


if __name__ == '__main__':
    main()
