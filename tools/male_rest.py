# Мужская фигура: база hm08 + caucasian-male-young, руки вниз как у женской (stage 1), морфы мерок
import json,gzip,numpy as np,sys
sys.path.insert(0,'.'); import glbio,objio
MB='/tmp/mh-build/'
V=np.array([list(map(float,l.split()[1:4])) for l in open(MB+'base.obj') if l.startswith('v ')])
def tgt(path):
  op=gzip.open if path.endswith('.gz') else open
  D=np.zeros_like(V)
  for l in op(path,'rt'):
    if l.startswith('#') or not l.strip(): continue
    a=l.split(); D[int(a[0])]=list(map(float,a[1:4]))
  return D
Vm=V+tgt('/tmp/mh-male/caucasian-male-young.target.gz')+tgt('/tmp/mh-male/universal-male-young-averagemuscle-averageweight.target.gz')
TN=[('chestIncr','measure-bust-circ-incr'),('chestDecr','measure-bust-circ-decr'),('waistIncr','measure-waist-circ-incr'),('waistDecr','measure-waist-circ-decr'),
    ('bellyIncr','stomach-pregnant-incr'),('bellyDecr','stomach-pregnant-decr'),('hipsIncr','measure-hips-circ-incr'),('hipsDecr','measure-hips-circ-decr')]
TN+=[('massIncr','/tmp/mh-male/universal-male-young-averagemuscle-maxweight.target.gz'),('massDecr','/tmp/mh-male/universal-male-young-averagemuscle-minweight.target.gz')]
Td=[tgt(f if f.startswith('/') else MB+f+'.target') for _,f in TN]
rig=json.load(open(MB+'rig.default.json')); W=json.load(open(MB+'weights.default.json'))['weights']
groups=json.load(open(MB+'basemesh_vertex_groups.json'))
def gidx(name):
  idx=[]
  for a in groups[name]:
    if isinstance(a,list): idx+=list(range(a[0],a[1]+1))
    else: idx.append(a)
  return np.array(idx)
S=0.1
def J(name,X=Vm): return X[gidx(name)].mean(0)*S
X=Vm*S
# female reference directions (posed P0 of current female, before relaxed pose)
import importlib; import pose as F  # loads female P0 & toP joints
def rotmat(a,b):
  a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=a@b
  if np.linalg.norm(v)<1e-9: return np.eye(3)
  K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); return np.eye(3)+K+K@K*(1/(1+c))
def RTm(R,piv): M=np.eye(4); M[:3,:3]=R; M[:3,3]=piv-R@piv; return M
L={}
for sd in ('l','r'):
  S_=sd.upper()
  sh,el,hd=J(f'joint-{sd}-shoulder'),J(f'joint-{sd}-elbow'),J(f'joint-{sd}-hand')
  fsh,fel,fhd=F.joint(f'joint-{sd}-shoulder'),F.joint(f'joint-{sd}-elbow'),F.joint(f'joint-{sd}-hand')
  tgt_dir=F.rot(np.array([0,0,1.]),(-6 if sd=='l' else 6))@(fel-fsh)
  R1=rotmat(el-sh,tgt_dir); L[f'upperarm01.{S_}']=RTm(R1,sh)
  el2=sh+R1@(el-sh); hd2=sh+R1@(hd-sh)
  R2=rotmat(hd2-el2,F.rot(np.array([0,0,1.]),(-6 if sd=='l' else 6))@(fhd-fel)); L[f'lowerarm01.{S_}']=RTm(R2,el2)
def build(local):
  M={}
  def get(b):
    if b in M: return M[b]
    p=rig[b]['parent']; Mp=get(p) if p else np.eye(4); M[b]=Mp@local.get(b,np.eye(4)); return M[b]
  for b in rig: get(b)
  return M
M=build(L)
A=np.zeros((len(V),3,4)); ws=np.zeros(len(V))
for b in rig:
  for vi,w in W.get(b,[]): A[vi]+=w*M[b][:3,:]; ws[vi]+=w
A/=np.maximum(ws,1e-9)[:,None,None]; A[ws<1e-9]=np.eye(4)[:3]
P=np.einsum('vij,vj->vi',A[:,:,:3],X)+A[:,:,3]
T=[np.einsum('vij,vj->vi',A[:,:,:3],d*S) for d in Td]
off=np.array([-P[:13380,0].mean()*0, P[:13380,1].min(), 0]); P-=[0,P[:13380,1].min(),0]
P[:,0]-=P[:13380,0].mean()
np.savez('male_rest_full.npz',P=P,T=np.array(T))
F_I=F.I0
np.savez('male_rest.npz',P=P[:13380],T=np.array([t[:13380] for t in T]),I=F_I)
print('height',P[:13380,1].max(),'names',[n for n,_ in TN])
