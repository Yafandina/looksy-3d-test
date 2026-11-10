import json,struct,numpy as np
def load(path):
  b=open(path,'rb').read()
  l=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+l]); bin0=20+l+8
  def acc(i):
    a=j['accessors'][i]; bv=j['bufferViews'][a['bufferView']]; o=bin0+bv.get('byteOffset',0)+a.get('byteOffset',0)
    n={'VEC3':3,'SCALAR':1}[a['type']]; dt={5126:np.float32,5125:np.uint32,5123:np.uint16}[a['componentType']]
    x=np.frombuffer(b,dt,a['count']*n,o).copy()
    return x.reshape(-1,3) if n==3 else x
  prim=j['meshes'][0]['primitives'][0]
  P=acc(prim['attributes']['POSITION']).astype(np.float64)
  N=acc(prim['attributes']['NORMAL']).astype(np.float64)
  I=acc(prim['indices']).astype(np.int64)
  T=[acc(t['POSITION']).astype(np.float64) for t in prim['targets']]
  names=j['meshes'][0]['extras']['targetNames']
  return P,N,I,T,names
def save(path,P,N,I,T,names,color=(0.82,0.80,0.77,1.0)):
  import pygltflib as g
  blobs=[];views=[];accs=[]
  def add(arr,ctype,typ,target,minmax=False):
    data=arr.tobytes(); off=sum(len(x) for x in blobs)
    pad=(-len(data))%4; blobs.append(data+b'\0'*pad)
    views.append(g.BufferView(buffer=0,byteOffset=off,byteLength=len(data),target=target))
    a=g.Accessor(bufferView=len(views)-1,componentType=ctype,count=len(arr),type=typ)
    if minmax: a.min=arr.min(0).tolist(); a.max=arr.max(0).tolist()
    accs.append(a); return len(accs)-1
  pa=add(P.astype(np.float32),5126,'VEC3',34962,True)
  na=add(N.astype(np.float32),5126,'VEC3',34962)
  ia=add(I.astype(np.uint32),5125,'SCALAR',34963)
  ta=[add(t.astype(np.float32),5126,'VEC3',34962,True) for t in T]
  ia_acc=accs[ia]; ia_acc.min=[int(I.min())]; ia_acc.max=[int(I.max())]
  m=g.GLTF2(asset=g.Asset(generator='looksy-build',version='2.0'),
    buffers=[g.Buffer(byteLength=sum(len(x) for x in blobs))],bufferViews=views,accessors=accs,
    materials=[g.Material(name='mannequin',pbrMetallicRoughness=g.PbrMetallicRoughness(baseColorFactor=list(color),metallicFactor=0,roughnessFactor=0.6))],
    meshes=[g.Mesh(primitives=[g.Primitive(attributes=g.Attributes(POSITION=pa,NORMAL=na),indices=ia,material=0,targets=[{'POSITION':t} for t in ta])],weights=[0.0]*len(T),extras={'targetNames':names})],
    nodes=[g.Node(mesh=0,name='Body')],scenes=[g.Scene(nodes=[0])],scene=0)
  m.set_binary_blob(b''.join(blobs)); m.save_binary(path)
def normals(P,I):
  F=I.reshape(-1,3); n=np.cross(P[F[:,1]]-P[F[:,0]],P[F[:,2]]-P[F[:,0]])
  N=np.zeros_like(P)
  for k in range(3): np.add.at(N,F[:,k],n)
  return N/np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12)
