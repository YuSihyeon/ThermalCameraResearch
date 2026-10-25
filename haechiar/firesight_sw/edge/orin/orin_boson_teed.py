#!/usr/bin/env python3
"""Jetson Orin Nano 8GB + FLIR Boson 320 TEED runtime.

Features
- Boson 320x256, 50 deg HFOV, 4.3 mm, nominal 9 Hz defaults
- Y16/gray/BGR UVC input
- CUDA/FP16 TEED inference
- CUDA-event profiling: encoder, edge head, TEED total
- End-to-end latency + rolling FPS overlay
- Interactive recording: press r to start/stop, q/Esc to quit
- --record for automatic recording in headless or GUI mode

TEED is an edge detector. Y16 is contrast-normalized; no absolute-temperature
or fire classification is performed.
"""
import argparse, json, os, sys, time
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

W,H,FPS,HFOV,FOCAL = 320,256,9.0,50.0,4.3
MEAN_BGR=np.asarray((103.939,116.779,123.68),np.float32)
IMG_EXT=(".bmp",".jpeg",".jpg",".png",".tif",".tiff",".webp")

def _smish(x): return x*torch.tanh(torch.log1p(torch.sigmoid(x)))
class _Smish(nn.Module):
    def forward(self,x): return _smish(x)
class _DenseLayer(nn.Sequential):
    def __init__(self,i,o):
        super().__init__()
        self.add_module("conv1",nn.Conv2d(i,o,3,1,2,bias=True)); self.add_module("smish1",_Smish()); self.add_module("conv2",nn.Conv2d(o,o,3,1,bias=True))
    def forward(self,inputs):
        first,second=inputs; new=super().forward(_smish(first)); return .5*(new+second),second
class _DenseBlock(nn.Sequential):
    def __init__(self,n,i,o):
        super().__init__()
        for idx in range(n): self.add_module("denselayer{}".format(idx+1),_DenseLayer(i,o)); i=o
class _UpConvBlock(nn.Module):
    def __init__(self,i,s):
        super().__init__(); layers=[]; pads=[0,0,1,3,7]; k=2**s; pad=pads[s]
        for idx in range(s):
            o=1 if idx==s-1 else 16; layers += [nn.Conv2d(i,o,1),_Smish(),nn.ConvTranspose2d(o,o,k,2,pad)]; i=o
        self.features=nn.Sequential(*layers)
    def forward(self,x): return self.features(x)
class _SingleConvBlock(nn.Module):
    def __init__(self,i,o,s): super().__init__(); self.conv=nn.Conv2d(i,o,1,stride=s,bias=True)
    def forward(self,x): return self.conv(x)
class _DoubleConvBlock(nn.Module):
    def __init__(self,i,m,o=None,stride=1,use_activation=True):
        super().__init__(); self.use_activation=use_activation; o=m if o is None else o
        self.conv1=nn.Conv2d(i,m,3,padding=1,stride=stride); self.conv2=nn.Conv2d(m,o,3,padding=1); self.smish=_Smish()
    def forward(self,x):
        x=self.smish(self.conv1(x)); x=self.conv2(x); return self.smish(x) if self.use_activation else x
class _DoubleFusion(nn.Module):
    def __init__(self,i):
        super().__init__(); self.DWconv1=nn.Conv2d(i,i*8,3,1,1,groups=i); self.PSconv1=nn.PixelShuffle(1); self.DWconv2=nn.Conv2d(24,24,3,1,1,groups=24); self.AF=_Smish()
    def forward(self,x):
        a=self.PSconv1(self.DWconv1(self.AF(x))); b=self.PSconv1(self.DWconv2(self.AF(a))); return _smish((b+a).sum(1).unsqueeze(1))
class TED(nn.Module):
    def __init__(self):
        super().__init__(); self.block_1=_DoubleConvBlock(3,16,16,stride=2); self.block_2=_DoubleConvBlock(16,32,use_activation=False)
        self.dblock_3=_DenseBlock(1,32,48); self.maxpool=nn.MaxPool2d(3,2,1); self.side_1=_SingleConvBlock(16,32,2); self.pre_dense_3=_SingleConvBlock(32,48,1)
        self.up_block_1=_UpConvBlock(16,1); self.up_block_2=_UpConvBlock(32,1); self.up_block_3=_UpConvBlock(48,2); self.block_cat=_DoubleFusion(3)
    def encode(self,x):
        b1=self.block_1(x); side=self.side_1(b1); b2=self.block_2(b1); down=self.maxpool(b2); pre=self.pre_dense_3(down); b3,_=self.dblock_3([down+side,pre]); return b1,b2,b3
    def head(self,features):
        b1,b2,b3=features; outs=[self.up_block_1(b1),self.up_block_2(b2),self.up_block_3(b3)]; return self.block_cat(torch.cat(outs,1))
    def forward(self,x,single_test=False): return [self.head(self.encode(x))]

def parser():
    p=argparse.ArgumentParser(description="Orin Nano + Boson 320 TEED latency profiler")
    p.add_argument("--source",default="/dev/video0"); p.add_argument("--checkpoint",default="models/teed/5_model.pth")
    p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto"); p.add_argument("--half",action="store_true")
    p.add_argument("--width",type=int,default=W); p.add_argument("--height",type=int,default=H)
    p.add_argument("--capture-width",type=int,default=W); p.add_argument("--capture-height",type=int,default=H); p.add_argument("--capture-fps",type=float,default=FPS)
    p.add_argument("--input-format",choices=("auto","bgr","gray8","y16"),default="auto")
    p.add_argument("--threshold",type=float,default=.75); p.add_argument("--background-scale",type=float,default=.42); p.add_argument("--edge-width",type=int,default=1)
    p.add_argument("--thermal-low-percentile",type=float,default=1.0); p.add_argument("--thermal-high-percentile",type=float,default=99.0)
    p.add_argument("--warmup",type=int,default=5); p.add_argument("--max-frames",type=int); p.add_argument("--headless",action="store_true")
    p.add_argument("--record",action="store_true",help="start recording immediately")
    p.add_argument("--record-path",default="boson_teed_recording.mp4"); p.add_argument("--metrics",default="orin_boson_teed_metrics.json")
    return p

def device_for(s):
    if s=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return torch.device(s)
def load_model(path,dev,half):
    if not os.path.isfile(path): raise RuntimeError("missing checkpoint: "+path)
    m=TED().to(dev)
    try: st=torch.load(path,map_location=dev,weights_only=True)
    except TypeError: st=torch.load(path,map_location=dev)
    if isinstance(st,dict) and isinstance(st.get("state_dict"),dict): st=st["state_dict"]
    st={(k[7:] if k.startswith("module.") else k):v for k,v in st.items()}
    m.load_state_dict(st,strict=True); m.eval()
    if half:
        if dev.type!="cuda": raise RuntimeError("--half requires CUDA")
        m.half()
    return m

def open_capture(a):
    src=int(a.source) if a.source.isdigit() else a.source; v4l=isinstance(src,int) or str(src).startswith("/dev/video")
    cap=cv2.VideoCapture(src,cv2.CAP_V4L2) if v4l else cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError("unable to open source: "+a.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.capture_width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,a.capture_height); cap.set(cv2.CAP_PROP_FPS,a.capture_fps)
    if a.input_format=="y16": cap.set(cv2.CAP_PROP_CONVERT_RGB,0); cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*"Y16 "))
    return cap,"boson_uvc" if v4l else "video"
def unpack_y16(frame,w,h):
    x=np.asarray(frame)
    if x.dtype==np.uint16:
        if x.ndim==3 and x.shape[2]==1:x=x[:,:,0]
        if x.ndim==2:return x
    if x.dtype==np.uint8 and x.ndim==3 and x.shape[2]==2:return np.ascontiguousarray(x).view("<u2").reshape(x.shape[0],x.shape[1])
    if x.dtype==np.uint8 and x.ndim==2 and x.shape==(h,w*2):return np.ascontiguousarray(x).view("<u2").reshape(h,w)
    return None
def to_bgr(frame,a):
    if a.input_format in ("auto","y16"):
        y=unpack_y16(frame,a.capture_width,a.capture_height)
        if y is not None:
            lo,hi=np.percentile(y,[a.thermal_low_percentile,a.thermal_high_percentile]); hi=max(float(hi),float(lo)+1)
            g=np.clip((y.astype(np.float32)-lo)*255/(hi-lo),0,255).astype(np.uint8); return cv2.cvtColor(g,cv2.COLOR_GRAY2BGR),"y16"
        if a.input_format=="y16": raise RuntimeError("Y16 requested but input is not recognized as Y16")
    x=np.asarray(frame)
    if x.ndim==2:return cv2.cvtColor(x.astype(np.uint8),cv2.COLOR_GRAY2BGR),"gray8"
    if x.ndim==3 and x.shape[2]==1:return cv2.cvtColor(x[:,:,0].astype(np.uint8),cv2.COLOR_GRAY2BGR),"gray8"
    if x.ndim==3 and x.shape[2]>=3:return x[:,:,:3].astype(np.uint8),"bgr8"
    raise RuntimeError("unsupported frame shape")
def tensorize(frame,a,dev):
    r=cv2.resize(frame,(a.width,a.height),interpolation=cv2.INTER_AREA); x=r.astype(np.float32)-MEAN_BGR
    t=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).unsqueeze(0); pw=(8-a.width%8)%8; ph=(8-a.height%8)%8
    if pw or ph:t=F.pad(t,(0,pw,0,ph),mode="replicate")
    t=t.to(dev,non_blocking=True); return r,t.half() if a.half else t.float()

def profile_model(m,t,dev):
    if dev.type=="cuda":
        e0,e1,e2=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        e0.record(); f=m.encode(t); e1.record(); out=m.head(f); e2.record(); torch.cuda.synchronize(dev)
        enc=e0.elapsed_time(e1); head=e1.elapsed_time(e2); return out,float(enc),float(head),float(e0.elapsed_time(e2))
    s=time.perf_counter(); f=m.encode(t); mid=time.perf_counter(); out=m.head(f); end=time.perf_counter()
    return out,(mid-s)*1000,(end-mid)*1000,(end-s)*1000

def infer(m,frame,a,dev):
    e2e=time.perf_counter(); resized,t=tensorize(frame,a,dev); prep=(time.perf_counter()-e2e)*1000
    with torch.inference_mode(): logits,enc,head,model=profile_model(m,t,dev); prob=torch.sigmoid(logits[0,0,:a.height,:a.width])
    mask=(prob.float().detach().cpu().numpy()>=a.threshold).astype(np.uint8)*255
    if a.edge_width>1:
        k=a.edge_width if a.edge_width%2 else a.edge_width+1; mask=cv2.dilate(mask,np.ones((k,k),np.uint8))
    overlay=np.clip(resized.astype(np.float32)*a.background_scale,0,255).astype(np.uint8); overlay[mask>0]=(0,255,0)
    return overlay,{"preprocess":prep,"encoder":enc,"edge_head":head,"teed":model,"e2e":(time.perf_counter()-e2e)*1000}

def writer(path,a):
    w=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*"mp4v"),max(float(a.capture_fps),1.0),(a.width,a.height))
    if not w.isOpened(): raise RuntimeError("unable to create recording: "+path)
    return w
def pct(v,q): return float(np.percentile(np.asarray(v,np.float64),q)) if v else 0.0
def annotate(img,idx,t,fps,rec,p95):
    lines=[
      "TEED LATENCY | Encoder {:5.1f} ms | Head {:5.1f} ms".format(t["encoder"],t["edge_head"]),
      "TEED total {:5.1f} ms | End-to-end {:5.1f} ms | FPS {:4.1f}".format(t["teed"],t["e2e"],fps),
      "p95 Encoder {:5.1f} ms | TEED {:5.1f} ms | frame {}".format(p95[0],p95[1],idx),
      "Boson 320 | 50deg HFOV | 4.3mm | nominal 9Hz",
    ]
    for i,s in enumerate(lines):cv2.putText(img,s,(6,17+i*17),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,"GREEN = TEED EDGE | r: REC toggle | q/Esc: quit",(6,img.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1,cv2.LINE_AA)
    if rec:
        cv2.circle(img,(img.shape[1]-16,15),6,(0,0,255),-1); cv2.putText(img,"REC",(img.shape[1]-55,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1,cv2.LINE_AA)

def run(a):
    dev=device_for(a.device)
    if dev.type=="cuda": torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=True
    m=load_model(a.checkpoint,dev,a.half); image=(not a.source.startswith("/dev/video") and os.path.splitext(a.source)[1].lower() in IMG_EXT)
    cap=None; kind="image"
    if image: raw=cv2.imread(a.source,cv2.IMREAD_UNCHANGED); ok=raw is not None
    else: cap,kind=open_capture(a); ok,raw=cap.read()
    if not ok: raise RuntimeError("source produced no frame")
    bgr,fmt=to_bgr(raw,a)
    for _ in range(a.warmup): infer(m,bgr,a,dev)
    rec=a.record and not image; vw=writer(a.record_path,a) if rec else None
    hist={k:[] for k in ("preprocess","encoder","edge_head","teed","e2e")}; stamps=deque(maxlen=30); n=0; started=time.perf_counter()
    try:
      while True:
        bgr,fmt=to_bgr(raw,a); out,t=infer(m,bgr,a,dev)
        for k in hist:hist[k].append(t[k])
        stamps.append(time.perf_counter()); fps=(len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 and stamps[-1]>stamps[0] else 0.0
        annotate(out,n,t,fps,rec,(pct(hist["encoder"],95),pct(hist["teed"],95)))
        if vw:vw.write(out)
        key=-1
        if not a.headless:
            cv2.imshow("FireSight Edge - Boson 320 TEED",out); key=cv2.waitKey(1)&0xff
            if key in (27,ord("q")):break
            if key==ord("r") and not image:
                if rec:
                    vw.release(); vw=None; rec=False; print("Recording stopped:",a.record_path)
                else:
                    vw=writer(a.record_path,a); rec=True; print("Recording started:",a.record_path)
        n+=1
        if image or (a.max_frames and n>=a.max_frames):break
        ok,raw=cap.read()
        if not ok:break
    finally:
      if cap:cap.release()
      if vw:vw.release()
      if not a.headless:cv2.destroyAllWindows()
    elapsed=max(time.perf_counter()-started,1e-9)
    metrics={"protocol":"firesight_edge_latency_v2","device":str(dev),"torch_version":torch.__version__,"camera":{"family":"FLIR Boson 320","resolution":[W,H],"hfov_deg":HFOV,"focal_length_mm":FOCAL,"nominal_fps":FPS},"processed_frames":n,"effective_fps":n/elapsed,"record_path":a.record_path,"latency_ms":{k:{"mean":float(np.mean(v)) if v else 0.0,"p50":pct(v,50),"p95":pct(v,95),"max":max(v) if v else 0.0} for k,v in hist.items()}}
    if a.metrics:
        with open(a.metrics,"w",encoding="utf-8") as f:json.dump(metrics,f,indent=2);f.write("\n")
    print(json.dumps(metrics,indent=2)); return metrics

def main():
    try:return 0 if run(parser().parse_args()) is not None else 0
    except (OSError,RuntimeError,ValueError) as e:sys.stderr.write("error: {}\n".format(e));return 2
if __name__=="__main__":raise SystemExit(main())
