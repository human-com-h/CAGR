"""Metadata reader from the verified research projection utility."""
import struct

class Compact:
    def __init__(self,data=b''):self.data=data;self.i=0;self.out=bytearray()
    def byte(self):
        if self.i>=len(self.data):raise ValueError('Truncated compact metadata')
        v=self.data[self.i];self.i+=1;return v
    def varint(self):
        value=0
        for shift in range(0,70,7):
            x=self.byte();value|=(x&127)<<shift
            if not x&128:return value
        raise ValueError('Invalid varint')
    def putint(self,v):
        if v<0:raise ValueError('Unsigned varint')
        while v>127:self.out.append((v&127)|128);v>>=7
        self.out.append(v)
    @staticmethod
    def zz(v):return v<<1 if v>=0 else ((-v)<<1)-1
    @staticmethod
    def unzz(v):return (v>>1)^-(v&1)
    def value(self,t,depth=0):
        if depth>128:raise ValueError('Metadata nesting limit')
        if t in [1,2]:return t==1
        if t==3:return struct.unpack('b',bytes([self.byte()]))[0]
        if t in [4,5,6]:return self.unzz(self.varint())
        if t==7:
            b=bytes(self.byte() for _ in range(8));return struct.unpack('<d',b)[0]
        if t==8:
            n=self.varint()
            if n>len(self.data)-self.i:raise ValueError('Truncated binary')
            b=self.data[self.i:self.i+n];self.i+=n;return b
        if t in [9,10]:
            x=self.byte();n=x>>4;kind=x&15
            if n==15:n=self.varint()
            if n>10000000:raise ValueError('Container limit')
            a=[]
            for _ in range(n):a.append(self.byte()==1 if kind in [1,2] else self.value(kind,depth+1))
            return kind,a
        if t==11:
            n=self.varint();x=self.byte() if n else 0;kt,vt=x>>4,x&15
            return kt,vt,[(self.value(kt,depth+1),self.value(vt,depth+1)) for _ in range(n)]
        if t==12:
            result={};last=0
            while True:
                x=self.byte()
                if x==0:return result
                kind=x&15;delta=x>>4;fid=last+delta if delta else self.unzz(self.varint());last=fid
                if fid in result:raise ValueError('Duplicate compact field')
                result[fid]=(kind,self.value(kind,depth+1))
        raise ValueError('Unsupported compact type '+str(t))
    def put(self,t,v):
        if t in [1,2]:return
        if t==3:self.out.extend(struct.pack('b',v))
        elif t in [4,5,6]:self.putint(self.zz(v))
        elif t==7:self.out.extend(struct.pack('<d',v))
        elif t==8:self.putint(len(v));self.out.extend(v)
        elif t in [9,10]:
            kind,values=v;n=len(values);self.out.append((min(n,15)<<4)|kind)
            if n>=15:self.putint(n)
            for a in values:
                if kind in [1,2]:self.out.append(1 if a else 2)
                else:self.put(kind,a)
        elif t==11:
            kt,vt,values=v;self.putint(len(values))
            if values:self.out.append((kt<<4)|vt)
            for a,b in values:self.put(kt,a);self.put(vt,b)
        elif t==12:
            last=0
            for fid,(kind,a) in sorted(v.items()):
                kind=(1 if a else 2) if kind in [1,2] else kind;delta=fid-last
                if 0<delta<=15:self.out.append((delta<<4)|kind)
                else:self.out.append(kind);self.putint(self.zz(fid))
                self.put(kind,a);last=fid
            self.out.append(0)
        else:raise ValueError('Unsupported compact type')

def field(s,n):return s[n][1]