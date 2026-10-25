#pragma once
// Host-only, compiled and exercised locally without Torch/CUDA.
#include <array>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <time.h>
#endif
namespace cagr_state {
constexpr int K=5,V=151936;
constexpr uint64_t TIMER_MASK=(1ULL<<44)-1,TOKEN_MASK=(1ULL<<18)-1;
inline int64_t clock_ns(){
#ifdef _WIN32
  LARGE_INTEGER c,f;QueryPerformanceCounter(&c);QueryPerformanceFrequency(&f);
  return (c.QuadPart/f.QuadPart)*1000000000LL+(c.QuadPart%f.QuadPart)*1000000000LL/f.QuadPart;
#else
  timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))throw std::runtime_error("CLOCK_MONOTONIC failed");
  return int64_t(t.tv_sec)*1000000000LL+t.tv_nsec;
#endif
}
struct Record {int64_t a=0,b=0,z=0,s=0,e=0,decode=0,capture=0,work=0,lo=0,hi=0;};
inline int decode(uint64_t packet,int64_t start,int64_t& end){
  if(packet>=(1ULL<<62))throw std::runtime_error("Invalid packed token value");
  end=start+int64_t(((packet>>18)-(uint64_t(start)&TIMER_MASK))&TIMER_MASK);
  if(end-start>=10800LL*1000000000LL)throw std::runtime_error("Invalid GPU timing packet");
  return int(packet&TOKEN_MASK); // Nonfinite sentinel retained for explicit rejection.
}
inline int64_t union_at(const Record& r,int64_t offset){
  auto overlap=std::max<int64_t>(0,std::min(r.b,r.e+offset)-std::max(r.a,r.s+offset));
  return r.b-r.a+r.e-r.s-overlap;
}
inline void enclose(Record& r,int64_t capture_end){
  if(!(r.a<=r.b&&r.b<=r.z&&r.s<=r.e&&r.z+r.decode<=capture_end))
    throw std::runtime_error("Invalid per-dispatch causal timing order");
  r.lo=r.a-r.s;r.hi=r.z-r.e;
  if(r.lo>r.hi)throw std::runtime_error("GPU work exceeds its causal host read enclosure");
  r.work=std::max(union_at(r,r.lo),union_at(r,r.hi));r.capture=capture_end-r.z-r.decode;
}
struct Records {
  std::array<Record,K> rows{};int count=0;
  void reset(){count=0;}
  Record& next(int n,int limit){
    if(!(1<=n&&n<=limit&&limit<=K))throw std::runtime_error("K5 window exceeded");
    if(n==1)reset();
    if(n!=count+1)throw std::runtime_error("Non-sequential native signal step");
    count=n;rows[n-1]=Record{};return rows[n-1];
  }
};
}
