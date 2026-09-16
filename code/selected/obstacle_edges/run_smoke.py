"""Primary smoke-condition experiment on actual thermal PNG sequences."""
import json
import platform
import subprocess

from run_experiment import (HERE, ROOT, DEVICE, SIZE, cv2, np, torch, digest,
                            load_model, load_pidinet, adapt, evaluate)


def main():
    torch.manual_seed(20260906)
    np.random.seed(20260906)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    manifest=json.loads((HERE/'data/smoke_manifest.json').read_text())
    clips={}
    for split in ['train','test']:
        frames=[]
        for member in manifest['splits'][split]['members']:
            path=HERE/'data/smoke_frames'/member
            if digest(path)!=manifest['files'][member]['sha256']:
                raise RuntimeError(f'Source frame hash mismatch: {member}')
            raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            if raw is None or raw.dtype!=np.uint8 or raw.shape!=(512,640,3):
                raise RuntimeError(f'Unexpected thermal format: {member}')
            # Preserve the dataset's actual 8-bit thermal display intensities.
            # No RGB->thermal synthesis, added smoke, or per-frame minmax stretch.
            frames.append(cv2.resize(raw,SIZE,interpolation=cv2.INTER_AREA))
        clips[split]=frames
    out=HERE/'outputs/smoke'
    out.mkdir(parents=True,exist_ok=True)
    teacher=load_model(str(ROOT/'edge/teed/models/5_model.pth'),DEVICE)
    student,training=adapt(teacher,clips['train'],out,'SmokeBasement run3 timestamped 24-second window')
    pidi,pidi_path=load_pidinet()
    report={'environment':{'python':platform.python_version(),'torch':torch.__version__,
            'opencv':cv2.__version__,'device':'cpu','threads':4,'processor':platform.processor()},
        'manifest':manifest,'training':training,
        'parameters':{'teed':sum(p.numel() for p in teacher.parameters()),'pidinet':sum(p.numel() for p in pidi.parameters())},
        'checkpoint_sha256':{'teed':digest(ROOT/'edge/teed/models/5_model.pth'),'pidinet':digest(pidi_path)},
        'pidinet_revision':subprocess.check_output(['git','-C',str(HERE/'third_party/pidinet'),'rev-parse','HEAD'],text=True).strip(),
        'results':evaluate('smoke_heldout',clips['test'],manifest['splits']['test']['fps'],teacher,student,pidi,out)}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SMOKE EXPERIMENT COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
