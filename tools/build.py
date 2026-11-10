import json,numpy as np,sys,io
from scipy.spatial import cKDTree
from PIL import Image, ImageFilter
sys.path.insert(0,'.'); import glbio,objio
import pygltflib as g
MB='/tmp/mh-build/'; SA='/tmp/mhassets/sys/'
cfg=json.load(open(sys.argv[1])) if len(sys.argv)>1 else {}
OUT=cfg.get('out','out/body.glb')
d=np.load(cfg.get('posed','posed.npz')); P=d['P']; T=list(d['T']); names=cfg.get('names',['breastIncr','breastDecr','underbustIncr','underbustDecr','waistIncr','waistDecr','bellyIncr','bellyDecr','hipsIncr','hipsDecr'])
V,VT,faces=objio.read_obj(MB+'base.obj','body')
B=V[:13380]
W=json.load(open(MB+'weights.default.json'))['weights']
# ---- adjacency
tri_full=[]
for f in faces:
  for i in range(1,len(f)-1): tri_full.append([f[0][0],f[i][0],f[i+1][0]])
tri_full=np.array(tri_full)
nb=[set() for _ in range(13380)]
for a,b,c in tri_full: nb[a]|={b,c}; nb[b]|={a,c}; nb[c]|={a,b}
nb=[np.array(sorted(x)) for x in nb]
def smooth_region(X,center_idx,radius,iters=0,strength=0.5):
  c=B[center_idx]; dist=np.linalg.norm(B-c,axis=1)
  inner=np.where(dist<radius)[0]; ann=np.where((dist>=radius)&(dist<radius*1.9))[0]
  allr=np.concatenate([inner,ann]); X=X.copy()
  cen=X[allr].mean(0); Q=X[allr]-cen
  U,S_,Vt=np.linalg.svd(Q,full_matrices=False); e1,e2,n=Vt[0],Vt[1],Vt[2]
  def uvh(Y): D=Y-cen; return D@e1,D@e2,D@n
  u,v,h=uvh(X[ann]); A=np.c_[np.ones_like(u),u,v,u*u,u*v,v*v,u**3,u*u*v,u*v*v,v**3]
  coef=np.linalg.lstsq(A,h,rcond=None)[0]
  u,v,h=uvh(X[inner]); A=np.c_[np.ones_like(u),u,v,u*u,u*v,v*v,u**3,u*u*v,u*v*v,v**3]; hf=A@coef
  t=np.clip((radius-dist[inner])/(radius*0.45),0,1); w=t*t*(3-2*t)
  X[inner]+=((hf-h)*w)[:,None]*n
  return X
def smooth_all(P,T,idxs,radius,iters):
  Pn=P.copy(); Tn=[]
  for ci in idxs: Pn=smooth_region(Pn,ci,radius,iters)
  for t in T:
    Q=P+t
    for ci in idxs: Q=smooth_region(Q,ci,radius,iters)
    Tn.append(Q-Pn)
  return Pn,Tn
if cfg.get('nipples',True):
  P,T=smooth_all(P,T,cfg.get('nipV',[8503,1831]),cfg.get('nipR',0.55),cfg.get('nipIt',150))
# ---- helper verts in posed space (local displacement)
s=(P[:,1].max()-P[:,1].min())/(B[:,1].max()-B[:,1].min()); tt=P.mean(0)-s*B.mean(0)
tree=cKDTree(B)
VP=np.zeros_like(V); VP[:13380]=P
dd,ii=tree.query(V[13380:],8); w=1/(dd+1e-3); w/=w.sum(1,keepdims=True)
disp=np.einsum('nk,nkd->nd',w,P[ii]-(s*B[ii]+tt)); VP[13380:]=s*V[13380:]+tt+disp
# ---- body UV split
vidx,tidx,tris=objio.split_uv(faces)
Nw=glbio.normals(P,tri_full)
BP=P[vidx]; BN=Nw[vidx]; BUV=VT[tidx].copy(); BUV[:,1]=1-BUV[:,1]
BT=[t[vidx] for t in T]
# ---- skin texture + painted underwear
skin=Image.open(SA+cfg['skin']).convert('RGB')
TEX=cfg.get('texsize',2048); skin=skin.resize((TEX,TEX),Image.LANCZOS)
sk=np.asarray(skin).astype(np.float32)
# rasterize rest positions into UV
def rasterize(attr):
  H=TEX; out=np.full((H,H,attr.shape[1]),np.nan,np.float32)
  uv=BUV*[H,H]; uv[:,1]=uv[:,1]
  for t in tris:
    p=uv[t]; mn=np.floor(p.min(0)).astype(int); mx=np.ceil(p.max(0)).astype(int)
    mn=np.clip(mn,0,H-1); mx=np.clip(mx,0,H-1)
    xs,ys=np.meshgrid(np.arange(mn[0],mx[0]+1)+0.5,np.arange(mn[1],mx[1]+1)+0.5)
    q=np.stack([xs.ravel(),ys.ravel()],1)
    a,b,c=p; v0=b-a; v1=c-a; den=v0[0]*v1[1]-v1[0]*v0[1]
    if abs(den)<1e-12: continue
    v2=q-a; l1=(v2[:,0]*v1[1]-v1[0]*v2[:,1])/den; l2=(v0[0]*v2[:,1]-v2[:,0]*v0[1])/den; l0=1-l1-l2
    m=(l0>=-0.02)&(l1>=-0.02)&(l2>=-0.02)
    if not m.any(): continue
    val=l0[m,None]*attr[t[0]]+l1[m,None]*attr[t[1]]+l2[m,None]*attr[t[2]]
    out[(q[m,1]-0.5).astype(int),(q[m,0]-0.5).astype(int)]=val
  return out
armw=np.zeros(13380)
for b in W:
  if any(k in b for k in ('upperarm','lowerarm','wrist','finger','metacarpal','shoulder01')):
    for vi,ww in W[b]:
      if vi<13380: armw[vi]+=ww
legw=np.zeros(13380)
for b in W:
  if any(k in b for k in ('upperleg','lowerleg','foot','toe')):
    for vi,ww in W[b]:
      if vi<13380: legw[vi]+=ww
REG=(np.load(cfg['regsrc'])['P'] if cfg.get('regsrc') else glbio.load('body_src.glb')[0]) if cfg.get('regspace')=='P0' else B
attr=np.concatenate([REG[vidx],armw[vidx,None]],1)
R=rasterize(attr)
X,Y,Zc,AW=R[...,0],R[...,1],R[...,2],R[...,3]
valid=~np.isnan(X)
mask=np.zeros(X.shape,np.float32)
for reg in cfg.get('regions',[]):
  m=valid&(AW<reg.get('armmax',0.35))
  if 'y' in reg: m&=(Y>=reg['y'][0])&(Y<=reg['y'][1])
  if 'absx' in reg: m&=(np.abs(X)>=reg['absx'][0])&(np.abs(X)<=reg['absx'][1])
  if 'zmin' in reg: m&=Zc>=reg['zmin']
  if 'zmax' in reg: m&=Zc<=reg['zmax']
  mask=np.maximum(mask,m.astype(np.float32))
# legs bottom of shorts: curved hem follows thigh — handled by y range
from scipy import ndimage
mimg=Image.fromarray((mask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))
mb=np.asarray(mimg).astype(np.float32)/255
# за пределами UV-островов — растянуть маску (без светлых/тёмных швов при mip-map)
idx=ndimage.distance_transform_edt(~valid,return_distances=False,return_indices=True)
mfill=mask[idx[0],idx[1]]
mb=np.where(valid,np.maximum(mb*(mask>0)+mb*(mask==0),0),mfill)
# внутри острова у края — не размывать в фон
mb=np.where(valid&(ndimage.binary_erosion(valid,iterations=2)==0),mask,mb)
mm=mb[...,None]
col=np.array(cfg.get('fabric',[70,70,76]),np.float32)
rng=np.random.default_rng(1); noise=(rng.normal(0,1,(TEX,TEX,1))*3).astype(np.float32)
shade=1.0
fab=np.clip(col*shade+noise,0,255)
outtex=sk*(1-mm)+fab*mm
# dilate texture across seams (fill empty with skin tex anyway)
skin_out=Image.fromarray(outtex.astype(np.uint8))
buf=io.BytesIO(); skin_out.save(buf,'JPEG',quality=cfg.get('jpgq',82)); skinjpg=buf.getvalue()
Image.fromarray((mask*255).astype(np.uint8)).save('out/mask.png')
# roughness via mask -> metallicRoughness texture (G=rough)
# ---- proxies
proxies=[]
for p in cfg.get('proxies',[]):
  dcl=objio.read_mhclo(SA+p['mhclo']); folder=SA+p['mhclo'].rsplit('/',1)[0]+'/'
  pv,pvt,pf=objio.read_obj(folder+dcl['obj'])
  fitted=objio.fit_mhclo(dcl,VP)
  vi,ti,tr=objio.split_uv(pf)
  pos=fitted[vi]; uv=pvt[ti].copy(); uv[:,1]=1-uv[:,1]
  n=glbio.normals(fitted,np.array([[f[0][0],f[i][0],f[i+1][0]] for f in pf for i in range(1,len(f)-1)]))[vi]
  im=Image.open(folder+p['tex'] if not p['tex'].startswith('/') else p['tex'])
  if p.get('texsize'): im=im.resize((p['texsize'],p['texsize']),Image.LANCZOS)
  if p.get('tint'):
    a=np.asarray(im.convert('RGBA')).copy(); a[...,:3]=p['tint']; a[...,3]=(a[...,3]*p.get('alpha',1.0)).astype(np.uint8); im=Image.fromarray(a,'RGBA')
  if p.get('lossless'):
    buf=io.BytesIO(); im.save(buf,'WEBP',lossless=True); data=buf.getvalue()
  buf=io.BytesIO()
  if im.mode=='RGBA': im.save(buf,'WEBP',quality=88,exact=True,lossless=bool(p.get('lossless'))); mime='image/webp'
  else: im.convert('RGB').save(buf,'JPEG',quality=85); mime='image/jpeg'
  proxies.append(dict(name=p['name'],pos=pos,n=n,uv=uv,tri=tr,img=buf.getvalue(),mime=mime,alpha=im.mode=='RGBA'))
  print(p['name'],len(pos),len(buf.getvalue()))
# ---- glTF
blobs=[];views=[];accs=[];images=[];textures=[];materials=[];meshes=[];nodes=[]
def addblob(data,target=None):
  off=sum(len(x) for x in blobs); pad=(-len(data))%4; blobs.append(data+b'\0'*pad)
  views.append(g.BufferView(buffer=0,byteOffset=off,byteLength=len(data),target=target)); return len(views)-1
def acc(arr,ctype,typ,target,mm=False):
  bv=addblob(arr.tobytes(),target); a=g.Accessor(bufferView=bv,componentType=ctype,count=len(arr),type=typ)
  if mm: a.min=arr.min(0).tolist(); a.max=arr.max(0).tolist()
  accs.append(a); return len(accs)-1
webp=False
def tex(data,mime):
  global webp
  bv=addblob(data); images.append(g.Image(bufferView=bv,mimeType=mime))
  if mime=='image/webp':
    webp=True; textures.append(g.Texture(extensions={'EXT_texture_webp':{'source':len(images)-1}}))
  else: textures.append(g.Texture(source=len(images)-1))
  return len(textures)-1
def addmesh(name,pos,n,uv,tri,texi,alpha=False,targets=None,tnames=None,rough=0.6,double=False,arm=None):
  pa=acc(pos.astype(np.float32),5126,'VEC3',34962,True); na=acc(n.astype(np.float32),5126,'VEC3',34962)
  ua=acc(uv.astype(np.float32),5126,'VEC2',34962)
  ia=acc(tri.astype(np.uint32).ravel(),5125,'SCALAR',34963)
  m=g.Material(name=name,pbrMetallicRoughness=g.PbrMetallicRoughness(baseColorTexture=g.TextureInfo(index=texi),metallicFactor=0,roughnessFactor=rough),
     alphaMode='MASK' if alpha else 'OPAQUE',alphaCutoff=0.4 if alpha else None,doubleSided=double)
  materials.append(m)
  at=g.Attributes(POSITION=pa,NORMAL=na,TEXCOORD_0=ua)
  if arm is not None: setattr(at,'_ARM',acc(arm.astype(np.float32),5126,'SCALAR',34962))
  prim=g.Primitive(attributes=at,indices=ia,material=len(materials)-1)
  ex=None
  if targets:
    prim.targets=[{'POSITION':acc(t.astype(np.float32),5126,'VEC3',34962,True)} for t in targets]
    ex={'targetNames':tnames}
  meshes.append(g.Mesh(name=name,primitives=[prim],weights=[0.0]*len(targets) if targets else None,extras=ex))
  nodes.append(g.Node(mesh=len(meshes)-1,name=name))
st=tex(skinjpg,'image/jpeg')
addmesh('Body',BP,BN,BUV,tris,st,targets=BT,tnames=names,rough=0.55,arm=np.clip(armw[vidx],0,1))
for p in proxies:
  ti=tex(p['img'],p['mime']); addmesh(p['name'],p['pos'],p['n'],p['uv'],p['tri'],ti,alpha=p['alpha'],rough=0.7 if 'hair' in p['name'].lower() else 0.4,double=p['alpha'])
m=g.GLTF2(asset=g.Asset(generator='looksy-build (MakeHuman CC0 assets)',version='2.0'),buffers=[g.Buffer(byteLength=sum(len(x) for x in blobs))],
  bufferViews=views,accessors=accs,images=images,textures=textures,materials=materials,meshes=meshes,nodes=nodes,scenes=[g.Scene(nodes=list(range(len(nodes))))],scene=0)
if webp: m.extensionsUsed=['EXT_texture_webp']; m.extensionsRequired=['EXT_texture_webp']
m.set_binary_blob(b''.join(blobs)); m.save_binary(OUT)
import os; print(OUT,os.path.getsize(OUT))
# landmarks for config tuning
print('breast tip y',B[8451,1],'navel?',)
