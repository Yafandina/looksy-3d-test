import numpy as np
def read_obj(path,group=None):
  v=[];vt=[];faces=[];cur=None
  for l in open(path,errors='ignore'):
    if l.startswith('v '): v.append(list(map(float,l.split()[1:4])))
    elif l.startswith('vt '): vt.append(list(map(float,l.split()[1:3])))
    elif l.startswith('g '): cur=l.split()[1] if len(l.split())>1 else None
    elif l.startswith('f '):
      if group and cur!=group: continue
      fs=[]
      for p in l.split()[1:]:
        a=p.split('/'); fs.append((int(a[0])-1, int(a[1])-1 if len(a)>1 and a[1] else -1))
      faces.append(fs)
  return np.array(v),np.array(vt) if vt else None,faces
def split_uv(faces):
  key={};vidx=[];tidx=[];tris=[]
  for f in faces:
    ids=[]
    for (vi,ti) in f:
      k=(vi,ti)
      if k not in key: key[k]=len(vidx); vidx.append(vi); tidx.append(ti)
      ids.append(key[k])
    for i in range(1,len(ids)-1): tris.append([ids[0],ids[i],ids[i+1]])
  return np.array(vidx),np.array(tidx),np.array(tris)
def read_mhclo(path):
  L=open(path).read().split('\n'); d={'rows':[]}
  inv=False
  for l in L:
    s=l.split()
    if not s or l.startswith('#'): continue
    if s[0] in('x_scale','y_scale','z_scale'): d[s[0]]=(int(s[1]),int(s[2]),float(s[3]))
    elif s[0]=='obj_file': d['obj']=s[1]
    elif s[0]=='verts': inv=True; continue
    elif inv and s[0].lstrip('-').replace('.','').isdigit(): d['rows'].append(s)
    elif inv and s[0] in ('delete_verts','weights'): inv=False
  return d
def fit_mhclo(d,V):
  sc=np.array([abs(V[d[k][0],i]-V[d[k][1],i])/d[k][2] for i,k in enumerate(('x_scale','y_scale','z_scale'))])
  out=[]
  for r in d['rows']:
    if len(r)==1: out.append(V[int(r[0])])
    else:
      ii=[int(x) for x in r[:3]]; w=np.array(list(map(float,r[3:6]))); o=np.array(list(map(float,r[6:9])))
      out.append(w@V[ii]+o*sc)
  return np.array(out)
