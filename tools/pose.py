import json,numpy as np,sys
from scipy.spatial import cKDTree
sys.path.insert(0,'.'); import glbio
MB='/tmp/mh-build/'
import os
if os.environ.get('POSE_IN'):
  _d=np.load(os.environ['POSE_IN']); P0=_d['P']; T0=list(_d['T']); I0=_d['I']
else:
  P0,N0,I0,T0,names=glbio.load('body_src.glb')
V=np.array([list(map(float,l.split()[1:4])) for l in open(MB+'base.obj') if l.startswith('v ')])
B=V[:13380]
rig=json.load(open(MB+'rig.default.json')); W=json.load(open(MB+'weights.default.json'))['weights']
groups=json.load(open(MB+'basemesh_vertex_groups.json'))
# similarity B->P
s=(P0[:,1].max()-P0[:,1].min())/(B[:,1].max()-B[:,1].min())
t=P0.mean(0)-s*B.mean(0)
tree=cKDTree(B)
def grpcenter(name):
  g=groups[name]; idx=[]
  for a in g:
    if isinstance(a,list): idx+=list(range(a[0],a[1]+1)) if len(a)==2 else a
    else: idx.append(a)
  return V[np.array(idx)].mean(0)
def toP(cB,K=16):
  d,ii=tree.query(cB,K); w=1/(d+1e-3); w/=w.sum()
  disp=((P0[ii]-(s*B[ii]+t))*w[:,None]).sum(0)
  return s*cB+t+disp
J={}
def joint(name):
  if name not in J: J[name]=toP(grpcenter(name))
  return J[name]
def head(b):
  h=rig[b]['head']
  if h['strategy']=='CUBE': return joint(h['cube_name'])
  if h['strategy']=='VERTEX': return P0[h['vertex_index']] if h['vertex_index']<13380 else toP(V[h['vertex_index']])
  return None
def rot(axis,deg):
  a=np.asarray(axis,float); a/=np.linalg.norm(a); th=np.radians(deg)
  K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
  return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K
def RT(axis,deg,piv):
  R=rot(axis,deg); M=np.eye(4); M[:3,:3]=R; M[:3,3]=piv-R@piv; return M
def build(local):
  # local: bone -> 4x4 ; FK in posed-rest world
  M={}
  def get(b):
    if b in M: return M[b]
    p=rig[b]['parent']; Mp=get(p) if p else np.eye(4)
    M[b]=Mp@local.get(b,np.eye(4)); return M[b]
  for b in rig: get(b)
  return M
def skin(M,P,T):
  bones=list(rig); A=np.zeros((13380,3,4)); ws=np.zeros(13380)
  for b in bones:
    for vi,w in W.get(b,[]):
      if vi<13380: A[vi]+=w*M[b][:3,:]; ws[vi]+=w
  A/=np.maximum(ws,1e-9)[:,None,None]; A[ws<1e-9]=np.eye(4)[:3]
  Pn=np.einsum('vij,vj->vi',A[:,:,:3],P)+A[:,:,3]
  Tn=[np.einsum('vij,vj->vi',A[:,:,:3],d) for d in T]
  return Pn,Tn
Z=np.array([0,0,1.]); X=np.array([1.,0,0]); Y=np.array([0,1.,0])
def nrm(v): return v/np.linalg.norm(v)
def make(a,TH1=2.0,TH2=5.0,YAW=8.0,ELB=12.0,cfg=None):
  L={}
  rank=joint('joint-r-ankle'); rhip=joint('joint-r-upper-leg'); lhip=joint('joint-l-upper-leg')
  L['root']=RT(Z,TH1,rank)@RT(Z,-TH2,rhip)
  L['upperleg01.R']=RT(Z,TH2,rhip)
  lk=joint('joint-l-knee'); la=joint('joint-l-ankle')
  L['upperleg01.L']=RT(Z,TH2,lhip)@RT(Y,YAW,lhip)@RT(-X,a,lhip)
  L['lowerleg01.L']=RT(-X,-2*a,lk)
  L['foot.L']=RT(-X,a,la)
  L['spine04']=RT(Z,2.0,joint('joint-spine-4')); L['spine03']=RT(Z,2.0,joint('joint-spine-3')); L['spine02']=RT(Z,1.5,joint('joint-spine-2'))
  L['neck01']=RT(Z,-1.5,joint('joint-neck')); L['head']=RT(Z,-1.0,joint('joint-head'))
  for sd,sg in (('L',1),('R',-1)):
    c=joint(f'joint-{sd.lower()}-clavicle'); sh=joint(f'joint-{sd.lower()}-shoulder'); el=joint(f'joint-{sd.lower()}-elbow'); hd=joint(f'joint-{sd.lower()}-hand')
    L[f'clavicle.{sd}']=RT(Z,-3*sg,c)
    L[f'upperarm01.{sd}']=RT(Z,(0.5 if sd=='L' else -5.5),sh)
    d=nrm(hd-el); L[f'lowerarm01.{sd}']=RT(np.cross(d,Z),ELB,el)
    # hand
    f2=joint(f'joint-{sd.lower()}-finger-2-1'); f5=joint(f'joint-{sd.lower()}-finger-5-1'); f3=joint(f'joint-{sd.lower()}-finger-3-1')
    aa=f2-f5; bb=f3-hd; n=nrm(np.cross(bb,aa) if sd=='L' else np.cross(aa,bb))
    L[f'wrist.{sd}']=RT(np.cross(nrm(f3-hd),n),6,hd)
    ang={2:(8,14,10),3:(12,18,12),4:(14,20,13),5:(16,22,15)}
    for f,(a1,a2,a3) in ang.items():
      for k,an in zip((1,2,3),(a1,a2,a3)):
        j0=joint(f'joint-{sd.lower()}-finger-{f}-{k}'); j1=joint(f'joint-{sd.lower()}-finger-{f}-{k+1}')
        L[f'finger{f}-{k}.{sd}']=RT(np.cross(nrm(j1-j0),n),an,j0)
    for k,an in ((2,6),(3,8)):
      j0=joint(f'joint-{sd.lower()}-finger-1-{k}'); j1=joint(f'joint-{sd.lower()}-finger-1-{k+1}')
      L[f'finger1-{k}.{sd}']=RT(np.cross(nrm(j1-j0),n),an,j0)
  return L
lfoot=np.array([vi for vi,w in W['foot.L'] if vi<13380 and w>0.3]+[vi for b in W if b.startswith('toe') and b.endswith('.L') for vi,w in W[b] if vi<13380])
rfoot=np.array([vi for vi,w in W['foot.R'] if vi<13380 and w>0.3])
def solve():
  lo,hi=0.,25.
  for it in range(18):
    a=(lo+hi)/2; M=build(make(a)); Pn,_=skin(M,P0,[])
    gap=Pn[lfoot,1].min()-Pn[rfoot,1].min()
    if gap<0: lo=a
    else: hi=a
  return a,gap
if __name__=='__main__':
  a,gap=solve(); print('knee a',a,'gap',gap)
  M=build(make(a)); Pn,Tn=skin(M,P0,T0)
  Pn[:,1]-=Pn[:,1].min()
  np.savez(os.environ.get('POSE_OUT','posed.npz'),P=Pn,T=np.array(Tn),I=I0)
  print(Pn.min(0),Pn.max(0))
